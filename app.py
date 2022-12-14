import math
import time
from collections import OrderedDict

# import cv2
import torch
from PIL import Image
from flask import Flask, request
from flask.json import jsonify
from torch import nn
from torchvision import transforms, models

from prediction import car_lrp_for_img

app = Flask(__name__)


# @app.route('/')
# def hello_world():
#     return 'Hello World!'

class MobileNetV3(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV3, self).__init__()
        self.model = MobileNetV31()

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:13](out3)
        out5 = self.model.features[13:16](out4)
        return out3, out4, out5


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups,
                           bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))


def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


# ---------------------------------------------------#
#   SPP???????????????????????????????????????????????????
#   ???????????????
# ---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


# ---------------------------------------------------#
#   ?????? + ?????????
# ---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


# ---------------------------------------------------#
#   ???????????????
# ---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


# ---------------------------------------------------#
#   ???????????????
# ---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(

                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),

                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),

                h_swish() if use_hs else nn.ReLU(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ---------------------------------------------------#
#   ????????????yolov4?????????
# ---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),

        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class MobileNetV31(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., ):
        super(MobileNetV31, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # `   k,   t,   c, SE,HS,s
            # 208,208,16 -> 208,208,16
            [3, 1, 16, 0, 0, 1],

            # 208,208,16 -> 104,104,24
            [3, 4, 24, 0, 0, 2],
            [3, 3, 24, 0, 0, 1],

            # 104,104,24 -> 52,52,40
            [5, 3, 40, 1, 0, 2],
            [5, 3, 40, 1, 0, 1],
            [5, 3, 40, 1, 0, 1],

            # 52,52,40 -> 26,26,80
            [3, 6, 80, 0, 1, 2],
            [3, 2.5, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],

            # 26,26,80 -> 26,26,112
            [3, 6, 112, 1, 1, 1],
            [3, 6, 112, 1, 1, 1],

            # 26,26,112 -> 13,13,160
            [5, 6, 160, 1, 1, 2],
            [5, 6, 160, 1, 1, 1],
            [5, 6, 160, 1, 1, 1]
        ]

        input_channel = _make_divisible(16 * width_mult, 8)
        # 416,416,3 -> 208,208,16
        layers = [conv_3x3_bn(3, input_channel, 2)]

        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class YoloBody(nn.Module):
    def __init__(self, num_classes=196, pretrained=False):
        super(YoloBody, self).__init__()
        # ---------------------------------------------------#
        #   ??????mobilnet????????????????????????????????????????????????
        # ---------------------------------------------------#

        self.backbone = MobileNetV3()
        in_filters = [40, 112, 160]

        self.conv1 = make_three_conv([512, 1024], in_filters[2])
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(in_filters[1], 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(in_filters[0], 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3 = yolo_head([256, 3 * (5 + num_classes)], 128)

        self.down_sample1 = conv_dw(128, 256, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2 = yolo_head([512, 3 * (5 + num_classes)], 256)

        self.down_sample2 = conv_dw(256, 512, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1 = yolo_head([1024, 3 * (5 + num_classes)], 512)
        self.classifier1 = nn.Sequential(
            nn.Linear(1809, 900),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(900, 196),
        )

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4, P5_upsample], axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = torch.cat([P3, P4_upsample], axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3)

        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample, P4], axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample, P5], axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)

        # ---------------------------------------------------#
        #   ??????????????????
        #   y3=(batch_size,75,52,52)
        # ---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        # ---------------------------------------------------#
        #   ??????????????????
        #   y2=(batch_size,75,26,26)
        # ---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        # ---------------------------------------------------#
        #   ??????????????????
        #   y1=(batch_size,75,13,13)
        # ---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        out2 = nn.functional.adaptive_avg_pool2d(out2, 1)
        out1 = nn.functional.adaptive_avg_pool2d(out1, 1)
        out0 = nn.functional.adaptive_avg_pool2d(out0, 1)
        # print("kkkkkkk", out2.shape)
        output1 = torch.cat((out2, out1), 1)
        output1 = torch.cat((out0, output1), 1).squeeze()
        output1 = self.classifier1(output1)

        return output1


from flask import render_template

filepath = 'user'

import pandas as pd

'''
???????????????????????????????????????????????????????????????
'''
import cv2
import numpy as np
import os
import copy

SZ = 20  # ??????????????????
MAX_WIDTH = 1000  # ????????????????????????
Min_Area = 2000  # ??????????????????????????????
PROVINCE_START = 1000


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def accurate_place(card_img_hsv, limit1, limit2, color, cfg):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    # col_num_limit = cfg["col_num_limit"]
    row_num_limit = cfg["row_num_limit"]
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # ???????????????
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl


def load_config():
    config = dict()

    for config_info in detect_config["config"]:
        if config_info["open"]:
            config = config_info
            break
    return config


def CaridDetect(detect_img):
    sign = 1
    # ????????????
    img = copy.deepcopy(detect_img)
    valid2 = True
    try:
        pic_hight, pic_width = img.shape[:2]
    except AttributeError:
        valid2 = False
    if valid2 == False:
        return 0, 0, 0, 0
    pic_hight, pic_width = img.shape[:2]

    if pic_width > MAX_WIDTH:
        resize_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
    # ????????????????????????????????????js??????????????????????????????????????????
    cfg = load_config()

    blur = cfg["blur"]
    # ????????????
    if blur > 0:
        img = cv2.GaussianBlur(img, (blur, blur), 0)  # ?????????????????????
    oldimg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # equ = cv2.equalizeHist(img)
    # img = np.hstack((img, equ))
    # ???????????????????????????????????????
    kernel = np.ones((20, 20), np.uint8)
    # morphologyEx ?????????????????????
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0);

    # ?????????????????? Canny????????????
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # ????????????????????????????????????????????????????????????
    kernel = np.ones((cfg["morphologyr"], cfg["morphologyc"]), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

    # ?????????????????????????????????????????????????????????????????????????????????????????????????????????
    # cv2.findContours()????????????????????????????????????
    try:
        contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
    # print('[ INFO ] len(contours): {}'.format(len(contours)))

    # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    car_contours = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        # ????????????????????????????????? cnt ??????????????????????????????4?????????????????????????????????4?????????
        # ?????? cv2.minAreaRect() ????????????Box2D??????rect????????????????????????????????????x???y????????????????????????????????????????????????
        # ???????????????????????????????????????????????????4???????????????box, ???????????? cv2.boxPoints() ?????????
        # ????????????[ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]???

        # ??????????????????????????????4??????????????????????????????????????????????????????????????????????????????????????????????????????
        # https://blog.csdn.net/lanyuelvyun/article/details/76614872

        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        # print(wh_ratio)
        # ??????????????????????????????2???5.5?????????2???5.5????????????????????????????????????????????? ??????????????????3.5
        if wh_ratio > 2 and wh_ratio < 5.5:
            car_contours.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
            # cv2.imshow("edge4", oldimg)
            # print(rect)
    # print("[ INFo ] len(car_contours): {}".format(len(car_contours)))
    # print("[ INFO ] ????????????.")

    card_imgs = []

    # ??????????????????????????????????????????????????????????????????????????????
    # ????????????????????????????????????YOLO,SSD???????????????????????????????????????????????????????????????????????????????????????????????? :) )
    for rect in car_contours:
        if rect[2] > -1 and rect[2] < 1:  # ????????????????????????????????????????????????????????????
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # ??????rect????????????????????????????????????

        box = cv2.boxPoints(rect)
        # ??????????????????????????????
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_hight]
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point

        if left_point[1] <= right_point[1]:  # ?????????
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # ??????????????????????????????
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)  # ????????????
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            card_imgs.append(card_img)
            # cv2.imshow("card", card_img)
            # cv2.waitKey(0)
        elif left_point[1] > right_point[1]:  # ?????????

            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # ??????????????????????????????
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            card_imgs.append(card_img)
            # cv2.imshow("card", card_img)
            # cv2.waitKey(0)

    # ?????????????????????????????????????????????????????????????????????????????????????????????

    sign, roi, labels, card_color = 0, 0, 0, 0

    colors = []
    rois = []
    for card_index, card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        try:
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        except BaseException:
            continue
        # ??????????????????????????????????????????????????????????????????
        if card_img_hsv is None:
            continue
        row_num, col_num = card_img_hsv.shape[:2]
        card_img_count = row_num * col_num

        temp_ = 0
        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if 11 < H <= 34 and S > 34:  # ?????????????????????
                    yello += 1
                elif 35 < H <= 99 and S > 34:  # ?????????????????????
                    green += 1
                elif 99 < H <= 124 and S > 34:  # ?????????????????????
                    blue += 1

                if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                    black += 1
                elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                    white += 1
        color = "no"

        limit1 = limit2 = 0
        if yello * 2 >= card_img_count:
            color = "yello"
            limit1 = 11
            limit2 = 34  # ???????????????????????????
        elif green * 2 >= card_img_count:
            color = "green"
            limit1 = 35
            limit2 = 99
        elif blue * 2 >= card_img_count:
            color = "blue"
            limit1 = 100
            limit2 = 124  # ???????????????????????????
        elif black + white >= card_img_count * 0.7:  # TODO
            color = "bw"
        # print("[ INFO ] color: {}".format(color))

        # print(blue, green, yello, black, white, card_img_count)
        # cv2.imshow("color", card_img)
        # cv2.waitKey(0)a
        if limit1 == 0:
            continue
        # ???????????????????????????

        # ??????????????????????????????????????????????????????????????????
        xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color, cfg)
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (
                yh - yl) // 4:yh, xl:xr]
        if need_accurate:  # ??????x???y????????????????????????????????????
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color, cfg)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (
                yh - yl) // 4:yh, xl:xr]

        sign = 1
        roi = card_img
        card_color = color
        colors.append(color)
        rois.append(roi)
        labels = (int(right_point[1]), int(heigth_point[1]), int(left_point[0]), int(right_point[0]))

    return sign, rois, labels, colors  # ???????????????????????????


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:

        dates = pd.read_csv(filepath, sep=' ')
        if request.form.get('username') in dates['user'].values:
            return render_template('/register.html')
        else:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(request.form.get('username') + ' ' + request.form.get('password') + '\n')
            return render_template('_05_web_page.html')


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'GET':
        return render_template('/login.html')
    else:
        dates = pd.read_csv(filepath, sep=' ')
        if request.form.get('username') in dates['user'].values:
            if request.form.get('password') in dates[dates['user'] == request.form.get('username')]['password'].values:
                return render_template('_05_web_page.html')
            else:
                return render_template('/login.html')
        else:
            return render_template('/register.html')


@app.route('/zhuye', methods=['POST', 'GET'])
def zhuye():
    return render_template('_05_web_page.html')


def predict_image_resnet18(imageFilePath):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = imageFilePath
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    resnet18 = models.resnet18(pretrained=False)
    fc_inputs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Sequential(
        torch.nn.Linear(fc_inputs, 196),
    )

    model = resnet18.to('cpu')
    # load model weights
    # print("loading")
    weights_path = "./resNet18.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    # print("loaded")
    class_list = []
    f = open('class.txt', encoding='utf-8')
    line = f.readline()
    while line:
        # print(line, end='\n')  # ??? Python 3 ?????????
        line = f.readline()
        class_list.append(line)
    f.close()
    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to('cpu'))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        # print(class_list[predict_cla])
    return class_list[predict_cla]


def predict_image_yolo(imageFilePath):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = imageFilePath
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    resnet18 = models.resnet18(pretrained=False)
    fc_inputs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Sequential(
        torch.nn.Linear(fc_inputs, 196),
    )

    yolo = YoloBody().to('cpu')
    # model_path = 'yolo.pth'
    # load model weights
    # print("loading")
    weights_path = "./yolo.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    yolo.load_state_dict(torch.load(weights_path, map_location='cpu'))
    # print("loaded")
    class_list = []
    f = open('class.txt', encoding='utf-8')
    line = f.readline()
    while line:
        # print(line, end='\n')  # ??? Python 3 ?????????
        line = f.readline()
        class_list.append(line)
    f.close()
    # prediction
    yolo.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(yolo(img.to('cpu'))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        # print(class_list[predict_cla])
    return class_list[predict_cla]


def predict_image_alexnet(imageFilePath):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = imageFilePath
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    alexnet = models.alexnet(pretrained=False)
    alexnet.classifier = nn.Sequential(
        nn.Linear(9216, 196),
    )

    model = alexnet.to('cpu')
    # load model weights
    # print("loading")
    weights_path = "./alexnet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    # print("loaded")
    class_list = []
    f = open('class.txt', encoding='utf-8')
    line = f.readline()
    while line:
        # print(line, end='\n')  # ??? Python 3 ?????????
        line = f.readline()
        class_list.append(line)
    f.close()
    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to('cpu'))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        # print(class_list[predict_cla])
    return class_list[predict_cla]


def predict_image_vgg16(imageFilePath):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = imageFilePath
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    vgg16 = models.vgg16(pretrained=False)
    # print(vgg16)
    # for param in resnet152.parameters():
    #     param.requires_grad = False
    # fc_inputs = vgg16.classifier.in_features
    vgg16.fc = nn.Sequential(
        nn.Linear(25088, 196),
    )

    # ???GPU???????????????
    vgg16 = vgg16.to('cpu')
    # print("loading")
    weights_path = "./vgg16.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    vgg16.load_state_dict(torch.load(weights_path, map_location='cpu'))
    # print("loaded")
    class_list = []
    f = open('class.txt', encoding='utf-8')
    line = f.readline()
    while line:
        # print(line, end='\n')  # ??? Python 3 ?????????
        line = f.readline()
        class_list.append(line)
    f.close()
    # prediction
    vgg16.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(vgg16(img.to('cpu'))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        # print(class_list[predict_cla])
    return class_list[predict_cla]


def predict_image_googlenet(imageFilePath):
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = imageFilePath
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    googlenet = models.googlenet(pretrained=True)
    # print(googlenet)
    # for param in resnet152.parameters():
    #     param.requires_grad = False
    # fc_inputs = vgg16.classifier.in_features
    googlenet.fc = nn.Sequential(
        nn.Linear(1024, 196),
    )

    # ???GPU???????????????
    googlenet = googlenet.to('cpu')

    # ???GPU???????????????
    # vgg16 = vgg16.to('cpu')
    # print("loading")
    weights_path = "./googlenet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    googlenet.load_state_dict(torch.load(weights_path, map_location='cpu'))
    # print("loaded")
    class_list = []
    f = open('class.txt', encoding='utf-8')
    line = f.readline()
    while line:
        # print(line, end='\n')  # ??? Python 3 ?????????
        line = f.readline()
        class_list.append(line)
    f.close()
    # prediction
    googlenet.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(googlenet(img.to('cpu'))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        # print(class_list[predict_cla])
    return class_list[predict_cla]


@app.route("/predict_image_resnet18", methods=['POST'])
def anyname_you_like_resnet18():
    startTime = time.time()
    # ????????????????????????
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        # ???????????????????????????????????????
        received_dirPath = 'resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('???????????????????????????????????????%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('????????????????????????????????????%.2f???' % usedTime)
        # ??????????????????????????????????????????????????????????????????????????????????????????
        startTime = time.time()
        predict_className = predict_image_resnet18(imageFilePath)
        usedTime = time.time() - startTime
        print('???????????????????????????????????????????????????%.2f???\n' % usedTime)
        return jsonify(predict_className=predict_className)


@app.route("/predict_image_alexnet", methods=['POST'])
def anyname_you_like_alexnet():
    startTime = time.time()
    # ????????????????????????
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        # ???????????????????????????????????????
        received_dirPath = 'resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('???????????????????????????????????????%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('????????????????????????????????????%.2f???' % usedTime)
        # ??????????????????????????????????????????????????????????????????????????????????????????
        startTime = time.time()
        predict_className = predict_image_alexnet(imageFilePath)
        usedTime = time.time() - startTime
        print('???????????????????????????????????????????????????%.2f???\n' % usedTime)
        return jsonify(predict_className=predict_className)


@app.route("/predict_image_googlenet", methods=['POST'])
def anyname_you_like_googlenet():
    startTime = time.time()
    # ????????????????????????
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        # ???????????????????????????????????????
        received_dirPath = 'resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('???????????????????????????????????????%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('????????????????????????????????????%.2f???' % usedTime)
        # ??????????????????????????????????????????????????????????????????????????????????????????
        startTime = time.time()
        predict_className = predict_image_googlenet(imageFilePath)
        print(predict_className)
        usedTime = time.time() - startTime
        print('???????????????????????????????????????????????????%.2f???\n' % usedTime)
        return jsonify(predict_className=predict_className)


@app.route("/predict_image_vgg16", methods=['POST'])
def anyname_you_like_vgg16():
    startTime = time.time()
    # ????????????????????????
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        # ???????????????????????????????????????
        received_dirPath = 'resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('???????????????????????????????????????%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('????????????????????????????????????%.2f???' % usedTime)
        # ??????????????????????????????????????????????????????????????????????????????????????????
        startTime = time.time()
        predict_className = predict_image_vgg16(imageFilePath)
        usedTime = time.time() - startTime
        print('???????????????????????????????????????????????????%.2f???\n' % usedTime)
        return jsonify(predict_className=predict_className)


@app.route("/predict_image_yolo", methods=['POST'])
def anyname_you_like_yolo():
    startTime = time.time()
    # ????????????????????????
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        # ???????????????????????????????????????
        received_dirPath = 'resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('???????????????????????????????????????%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('????????????????????????????????????%.2f???' % usedTime)
        # ??????????????????????????????????????????????????????????????????????????????????????????
        startTime = time.time()
        predict_className = predict_image_yolo(imageFilePath)
        usedTime = time.time() - startTime
        print('???????????????????????????????????????????????????%.2f???\n' % usedTime)
        return jsonify(predict_className=predict_className)


@app.route('/', methods=['POST', 'GET'])
def kong():
    return render_template('/login.html')


@app.route('/predict_chepai', methods=['POST', 'GET'])
def chepai():
    startTime = time.time()
    # ????????????????????????
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        # ???????????????????????????????????????
        received_dirPath = 'resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('???????????????????????????????????????%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('????????????????????????????????????%.2f???' % usedTime)
        # ??????????????????????????????????????????????????????????????????????????????????????????
        startTime = time.time()
        img = cv2.imread(imageFilePath, cv2.IMREAD_COLOR)
        str = car_lrp_for_img(img)
        print(str)
        # usedTime = time.time() - startTime
        # print('???????????????????????????????????????????????????%.2f???\n' % usedTime)
        return jsonify(predict_className=str)


if __name__ == '__main__':
    app.run()
