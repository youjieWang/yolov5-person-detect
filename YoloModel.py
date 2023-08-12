# import logging
import torch
import torch.nn as nn
from yolo_utils import check_img_size, non_max_suppression, scale_coords, letterbox
import numpy as np
import cv2

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class YoloModel():
    def __init__(self, configs):
        # set_logging()
        self.configs = configs
        # logging.basicConfig(format="%(message)s", level=logging.INFO)
        self.device = torch.device(self.configs['device'])  # 'cuda:0'
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = self._load_model(self.configs['weights'], map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.configs['imgsz'], s=self.stride)  # check img_siz e
        if self.half:
            self.model.half()  # to FP16

    def predict(self, image):
        # 传入一张图片
        # Padded resize
        img = letterbox(image, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # Run inference
        if self.device.type != 'cpu':  # 如果设备是GPU的运行
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        # img,
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=self.configs['augment'])[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.configs['conf_thres'], self.configs['iou_thres'], classes=self.configs['classes'],
                                   agnostic=self.configs['agnostic_nms'])
        # 框的过滤和转换
        res = []
        for det in pred:
            print(pred)
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # 这里其实还要判断取最大值的那个
                    if cls == 0.0:  # 如果是person类，那么就裁剪图像
                        w = xyxy[2] - xyxy[0]
                        h = xyxy[3] - xyxy[1]
                        if w < self.configs['size3'][0] or h < self.configs['size3'][1]:
                            continue
                        res = xyxy
        return res

    def _load_model(self, weights, map_location=None):
        model = Ensemble()  # ?
        ckpt = torch.load(weights, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        return model[-1]  # return model