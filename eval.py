"""
python eval.py --model_path /path/to/yolov4_model.pth --test_dir /path/to/test/images --annotations_path /path/to/test/annotations.txt --num_classes 80 --resolution 416 416 --conf_thresh 0.5 --nms_thresh 0.4 --use_cuda

"""
import torch
from torch import nn
import torch.nn.functional as F
import argparse
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tool.utils import *


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size):
        assert (x.data.dim() == 4)
        _, _, H, W = target_size
        return F.interpolate(x, size=(H, W), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size())
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size())
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)
        # self.yolo1 = YoloLayer(anchor_mask=[0, 1, 2], num_classes=80,
        #                        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        #                        num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        # self.yolo2 = YoloLayer(anchor_mask=[3, 4, 5], num_classes=80,
        #                        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        #                        num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        # self.yolo3 = YoloLayer(anchor_mask=[6, 7, 8], num_classes=80,
        #                        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        #                        num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)
        # y1 = self.yolo1(x2)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        # y2 = self.yolo2(x10)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        return [x2, x10, x18]
        # y3 = self.yolo3(x18)
        # return [y1, y2, y3]
        # return y3


class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neck = Neck()
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        # head
        self.head = Yolov4Head(output_ch)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output


def load_data(test_dir, annotations_path):
    with open(annotations_path, 'r') as f:
        annotations = f.readlines()

    annotations = [line.strip().split() for line in annotations]
    data = []
    for ann in annotations:
        img_path = os.path.join(test_dir, ann[0])
        boxes = [list(map(int, box.split(','))) for box in ann[1:]]
        data.append((img_path, boxes))
    return data

def evaluate_model(model, test_data, device, resolution, conf_thresh, nms_thresh, num_classes, output_dir, class_names, iou_threshold=0.4):
    model.eval()
    # Ensure resolution is a tuple for the transformation
    transform = transforms.Compose([transforms.Resize(tuple(resolution)), transforms.ToTensor()])
    results = []
    all_gts, all_preds = [], []
    images_with_no_predictions = []
    wrong_pred_paths = []
    wh = tuple(resolution)
    width = wh[0]
    height=wh[1]

    # Ensure the output directory for wrong predictions exists
    wrong_preds_dir = os.path.join(output_dir, 'wrong_preds')
    os.makedirs(wrong_preds_dir, exist_ok=True)

    for img_path, gt_boxes in test_data:
        img = Image.open(img_path).convert('RGB')  # Make sure the image is RGB
        # img_tensor = transform(img).unsqueeze(0).to(device)
        preds = do_detect(model, img, conf_thresh, num_classes, nms_thresh, use_cuda=device.type == 'cuda')

        gt_image = plot_boxes_with_return(img.copy(), gt_boxes, class_names, normalized=False)
        pred_image = plot_boxes_with_return(img.copy(), preds, class_names, normalized=True)

        combined_image = Image.new('RGB', (gt_image.width * 2, gt_image.height))
        combined_image.paste(gt_image, (0, 0))
        combined_image.paste(pred_image, (gt_image.width, 0))

        # Process predictions to include only predictions with conf > conf_thresh
        filtered_preds = [pred for pred in preds if pred[4] >= conf_thresh]


        # Flatten and prepare ground truths and predictions for evaluation
        gt_labels = [x[-1] for x in gt_boxes]
        pred_labels = [x[-1] for x in preds]
        all_gts.extend(gt_labels)
        all_preds.extend(pred_labels)
        results.append((gt_boxes, preds))

        # Check predictions against ground truths using IoU
        correct_detected = 0
        for pred in filtered_preds:
            x1 = (pred[0] - pred[2] / 2.0) * width
            y1 = (pred[1] - pred[3] / 2.0) * height
            x2 = (pred[0] + pred[2] / 2.0) * width
            y2 = (pred[1] + pred[3] / 2.0) * height
            pred = [x1,y1,x2,y2, pred[4], pred[5], pred[6]]
            for gt in gt_boxes:
                # print("gt: ", gt)
                # print("pred: ", pred)
                if pred[-1] == gt[-1]:  # Compare class ids
                    iou = bbox_iou(pred[:4], gt[:4])  # Assuming boxes are [x1, y1, x2, y2, conf, class_id]
                    if iou >= iou_threshold:
                        correct_detected += 1
                        break
        
        if correct_detected != len(gt_boxes):
            wrong_pred_paths.append(img_path)
            plot_boxes(img, filtered_preds, None, class_names)
            save_path = os.path.join(wrong_preds_dir, os.path.basename(img_path))
            combined_image.save(save_path)
            print(f"Saved comparison image: {save_path}")

    # Assuming background or no-object as class_id = -1
    if len(all_gts) != len(all_preds):
        if len(all_gts) > len(all_preds):
            all_preds.extend([-1] * (len(all_gts) - len(all_preds)))  # Fill missing predictions with -1
        else:
            all_gts.extend([-1] * (len(all_preds) - len(all_gts)))  # Less common, but just in case

    # Output images with no predictions
    # for no_pred_image_path in images_with_no_predictions:
    #     image = Image.open(no_pred_image_path)
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     save_path = os.path.join(output_dir, os.path.basename(no_pred_image_path))
    #     image.save(save_path)
    #     print(f"Saved image with no predictions: {save_path}")

    # Compute confusion matrix and other metrics
    cm = confusion_matrix(all_gts, all_preds)
    accuracy = np.diag(cm).sum() / cm.sum()
    # Further calculations for mAP can be added here

    return cm, accuracy, wrong_pred_paths

def main(args):
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    model = Yolov4(n_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    test_data = load_data(args.test_dir, args.annotations_path)
    class_names = load_class_names(args.class_names)
    cm, accuracy, _ = evaluate_model(model, test_data, device, args.resolution, args.conf_thresh, args.nms_thresh, args.num_classes, args.no_pred_dir, class_names)

    print("Confusion Matrix:\n", cm)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    # Additional prints for mAP or other metrics can be added

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate YOLOv4 Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model .pth file')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--no_pred_dir', type=str, required=True, help='Directory containing test images with no predictions')
    parser.add_argument('--class_names', type=str, required=True, help='Path to class names file')
    parser.add_argument('--annotations_path', type=str, required=True, help='Path to the annotations .txt file')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--resolution', type=int, nargs=2, default=[416, 416], help='Resolution to resize images to, specified as two integers (height width). Default is 416 416')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--nms_thresh', type=float, default=0.4, help='NMS threshold for detection')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for the evaluation')

    args = parser.parse_args()
    main(args)