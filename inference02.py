#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/28 17:03
# @Author  : yj.wang
# @File    : inference02.py
# -*- coding: UTF-8 -*-

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 这边直接在这里指定一张卡
import base64
import json
import os
import cv2
import numpy as np
import time
from pathlib import Path
import copy

import torch
from utils.general import increment_path, set_logging, check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.datasets import LoadImages
from utils.torch_utils import select_device, time_synchronized
from models.experimental import attempt_load

from utils.plots import plot_one_box

configs = {
    'name': 'exp',
    'device': 'cpu',
    'weights': r'./weights/yolov5s.pt',
    'imgsz': 640,
    'augment': False,  # 是否增强推理
    'conf_thres': 0.25,
    'iou_thres': 0.45,
    'classes': None,
    'agnostic_nms': False,
    'exist_ok': False,
    'alpha': 1.1  # 预测框扩大的比例
}


# 单张图片的多个boxes进行处理
def make_prelabel(path, img_name, img, boxes, h, w, img64):
    # json格式
    two_point_label = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
        ],
        "imagePath": img_name,
        "imageData": img64,
        "imageHeight": h,
        "imageWidth": w
    }

    for b in boxes:
        # box的前四个值表示
        box = np.array(b[:4]).astype(np.int32)
        box = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
        box = np.array(box)
        # 保存原始输入图片中的与标签
        two_point_label["shapes"].append({
            "label": str(int(box[-1])),
            "points": [
                [
                    b[0],
                    b[1]
                ],
                [
                    b[2],
                    b[3]
                ]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })
        with open(os.path.join(path, img_name.replace('.jpg', '.json')), 'w') as f:
            json.dump(two_point_label, f)

        cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)  # font thickness
        cv2.putText(
            img,
            str(b[-1]),
            (int(b[0]), int(b[1]) - 2),
            0,
            tl / 3,
            [225, 0, 0],
            thickness=tf,
            lineType=cv2.LINE_AA
        )
        cv2.imwrite(path + img_name, img)


def get_xywh(xyxy):
    cx = (xyxy[0] + xyxy[2]) / 2
    cy = (xyxy[1] + xyxy[3]) / 2
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    return [cx, cy, w, h]


def get_xyxy(xywh):
    # 转换成xyxy格式
    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = xywh[0] + xywh[2] / 2
    y2 = xywh[1] + xywh[3] / 2
    return [x1, y1, x2, y2]


def box_same_process(H, W, xyxy, alpha=1.1):
    '''
    1:1 和 440*852方法共同的处理部分
    H, W：原始图像宽高
    box：预测框的宽高
    return:
    box:扩大后的box的xyxy和hw
    '''
    box = []
    for x in xyxy:
        box.append(int(x))

    w = box[2] - box[0]
    h = box[3] - box[1]
    # 太小了就扩大一点点比例
    if int((w * 852) / 440) < h:
        alpha = 1.05
    box, w, h = box_expand_2(H, W, box, alpha=alpha) # 返回的元素都是int类型

    return box, w, h,


def box_expand(H, W, box, alpha=1.1):
    # 这个是按照比例扩大
    xywh = get_xywh(box)
    w1 = xywh[2] * alpha
    h1 = xywh[3] * alpha
    # 如果高度刚好大于边界的话就设置添加的长度和w一样的比例
    if xywh[1] + h1 > H:
        h1 = xywh[3] + (w1 - xywh[2])
    box = get_xyxy([xywh[0], xywh[1], w1, h1])
    # 有可能expand或者没有expand的时候就已经有下面的边界到图像的下边界，这时候就直接取下边界就好了
    box[0] = int(max(box[0], 0))
    box[1] = int(max(box[1], 0))
    box[2] = int(min(box[2], W))
    box[3] = int(min(box[3], H))


    return box, int(w1), int(h1)

def box_expand_2(H, W, box, alpha=1.1):
    # 四边都扩大同样的大小的像素，扩大的像素按照宽的比例
    xywh = get_xywh(box)
    w1 = xywh[2] * alpha
    h1 = xywh[3] + (w1 - xywh[2])
    box = get_xyxy([xywh[0], xywh[1], w1, h1])
    # TODO 这边expand有点问题需要处理
    # 有可能expand或者没有expand的时候就已经有下面的边界到图像的下边界
    box[0] = int(max(box[0], 0))
    box[1] = int(max(box[1], 0))
    box[2] = int(min(box[2], W))
    box[3] = int(min(box[3], H))

    return box, int(w1), int(h1)

def box_process(H, W, box, alpha=1.1):
    '''
    H:原始图片的高
    W：原始图片的宽
    box:[x,y,x,y]->int
    '''
    print('---\nbox expand---------')
    print(box)
    pad = 0
    w = box[2] - box[0]
    # h = box[3] - box[1]
    # TODO 行人框肯定是W最短，有可能出现W>H的情况，这种后面考虑
    r_h = int((w * 852) / 440)
    # y + r_h < H，不超过边界框就直接裁剪，其他出现意外情况先不考虑
    # 判断是否是要裁剪，如果是false表示要增加东西，如果是True表示要crop
    # crop = True if (w / h) < (440/852) else False
    if box[1] + r_h < H:
        box[3] = box[1] + r_h
    else:
        box[3] = H
        res = (box[1] + r_h) - H
        # 判断box[0]-res是否小于0
        if box[1] - res > 0:
            box[1] = box[1] - res
        else:
            # 否则需要上下pad
            print(int((res - box[1]) / 2))
            pad = 0 if int((res - box[1]) / 2) < 60 else int((res - box[1]) / 2)
            box[1] = 0
    # if crop:
    #     # 直接裁剪
    #     box[3] = box[1] + r_h
    # else:
    #     # <H直接裁剪
    #     if box[1] + r_h < H:
    #         box[3] = box[1] + r_h
    #     else:
    #         box[3] = H
    #         res = (box[1] + r_h) - H
    #         # 判断box[0]-res是否小于0
    #         box[1] = box[1] - res if box[1] - res > 0 else 0
    # TODO 有一个问题就是你裁剪之后他并不是有和原来比例是一样的
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])], pad


# 裁剪图片
def crop_resize_imgs(path1, img, box, index, size=(440, 852), save_img=False, pad=0):
    '''
    box:[x,y,x,y]->Torch.tensor
    labeled_path: only for test
    '''

    # img1 = img[int(box[1]): int(box[3]), int(box[0]): int(box[2]), :]
    img1 = img[box[1]: box[3], box[0]: box[2], :]
    if pad:
        img1 = cv2.copyMakeBorder(img1, pad, pad, 0, 0, cv2.BORDER_REPLICATE)
    img1 = cv2.resize(img1, size)

    if save_img:
        cv2.imwrite(f"{path1.split('.')[-2]}_resize{size}_{str(index)}.jpg", img1)

    return img1


def right_crop(img, box, path1, index, save_img=False):
    '''
    裁剪成440*852
    input
    img: 原始图片->np
    box：expand之后的预测框坐标->[int,int,...] xyxy格式
    path：裁剪图片存储的路径
    index：图片中第几个人
    save：是否保存图片
    return
    result['imgb64'] = imgb64  # base64类型的数据
    result['box'] = box  # 调整后的框
    result['id'] = index # 第几个人
    '''
    result = {'size': '440*852'}
    H, W, _ = img.shape
    # # 先框做处理在裁剪
    box, pad = box_process(H, W, box, alpha=configs['alpha'])  # 处理之后的裁剪框
    # 裁剪
    crop_img = crop_resize_imgs(path1, img, box, index, size=(440, 852), save_img=save_img, pad=pad)
    # # TODO 2、设计一个返回的json类型（完成）
    imgb64 = str(base64.b64encode(np.ascontiguousarray(crop_img)), 'utf-8')
    result['imgb64'] = imgb64
    result['box'] = box
    result['id'] = index
    return result


def box_to_square(box, H, W):
    '''
    裁剪框处理
    input
    box：预测框坐标->[int,int,...] xyxy格式
    # 1、predict的框肯定是在图像内部，没有超出图像的外的，所以这里不做预处理
    # 2、这里只考虑预测框h>w的情况，其他情况先不考虑
    '''
    w1 = box[0]
    w2 = W - box[2]
    pad_w = w1 if w2 > w1 else w2
    box[0] = box[0] - pad_w  # TODO 这个会不会超出边界，要不要+1
    box[2] = box[2] + pad_w
    w_e = box[2] - box[0]
    h = box[3] - box[1]
    if w_e < h:
        # TODO 扩大多少比例取决于(H-box[1]) / w_e 的比例，或者直接扩大比例再做处理，
        #  那就不用加像素，反正两个函数都是要expand，那只做一次就好了
        # box[1] = max(box[1] - 20, 0) # 这边在之前已经处理了，所以就不用添加像素了
        box[3] = box[1] + (box[2] - box[0])
    else:
        # TODO
        # 这个是在原始图片H>W 的情况下做调整
        # 1、上下调整同样的尺寸(w_e-h)/2
        # 2、min(h1, h2)>(w_e-h)/2 就上下调整(w_e-h)/2
        # 3、min(h1, h2)<(w_e-h)/2 短边到顶，长边延伸(w_e-h)/2+(h1-h2)
        h1 = box[1]
        h2 = H - box[3]
        pad_h = int((w_e-h)/2)  # 这里需要整型
        if min(h1, h2) > pad_h:
            box[1] = max(box[1] - pad_h, 0)
            box[3] = min(box[3] + pad_h, H)
        else:
            if h1 < h2:
                pad_down = 2 * pad_h - h1
                box[1] = 0
                box[3] = box[3] + pad_down
            else:
                pad_up = 2 * pad_h - h2
                box[1] = box[1] - pad_up
                box[3] = H

    return box


def square_crop(img, box, path1, index, save_img=False):
    '''
    1:1的方式裁剪
    input
    img: 原始图片->np
    box：expand之后的预测框坐标->[int,int,...] xyxy格式
    path：裁剪图片存储的路径
    index：图片中第几个人
    save：是否保存图片
    return
    result['imgb64'] = imgb64  # base64类型的数据
    result['box'] = box  # 调整后的框
    result['id'] = index # 第几个人
    '''
    box1 = copy.deepcopy(box)  # 你传入的时候这个框变量就改变了
    result = {'size': '1040*1040'}
    H, W, _ = img.shape
    box1 = box_to_square(box1, H, W)
    crop_img = crop_resize_imgs(path1, img, box1, index, size=(1024, 1024), save_img=save_img)
    imgb64 = str(base64.b64encode(np.ascontiguousarray(crop_img)), 'utf-8')
    result['imgb64'] = imgb64
    result['box'] = box1
    result['id'] = index
    return result




def infer(source, save_path, labeled_path, save_img=True):
    '''
    source: 原始数据的目录
    return :# 首先是多人，每个人返回的是什么操作
            # results = []
            # results.append(result{'imgb64': '裁剪之后imgb64的图像', 'xyxy': '坐标', id: '表示这张图像中第几个人'})
    '''
    results = []
    result = {}
    save_dir = Path(increment_path(Path(save_path) / configs['name'], exist_ok=configs['exist_ok']))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    set_logging()
    device = select_device(configs['device'])
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(configs['weights'], map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(configs['imgsz'], s=stride)  # check img_siz e
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':  # 如果设备是GPU的运行
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=configs['augment'])[0]

        # Apply NMS
        pred = non_max_suppression(pred, configs['conf_thres'], configs['iou_thres'], classes=configs['classes'],
                                   agnostic=configs['agnostic_nms'])
        t2 = time_synchronized()
        # print('------pred--------------')
        # print(pred)

        # Process detections
        for det in pred:  # detections per image
            p, s, im0, frame = path, path.split('\\')[-1], im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += ': %gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # print(save_path)  # data\crop_imgs\exp\bus.jpg
            # print(type(save_path))
            # print(txt_path)
            # TODO 判断是否检测到图片
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print('-----dect---------')
                # print(det)  # [x1, y1, x2, y2, conf, cls]
                index = 0
                for *xyxy, conf, cls in reversed(det):
                    # print(type(xyxy))
                    # print(xyxy)
                    # print(conf)
                    # print(cls)
                    if cls == 0.0:  # 如果是person类，那么就裁剪图像
                        # TODO 这里做一个判断就是非常小的人就不用裁剪
                        _, _, w, h = get_xywh(xyxy)  # 裁剪框的宽高
                        if w < 100 or h < 100:
                            continue

                        H, W, _ = im0.shape
                        # 统一处理成int类型， 并且扩大box
                        box, w, h = box_same_process(H, W, xyxy, alpha=1.1)

                        # 预测框-------------------------
                        # plot_one_box(xyxy, im0, color=[0, 0, 0], label='predict', line_thickness=3)
                        # # 扩大框
                        # plot_one_box(box, im0, color=[0, 225, 0], label='box_expand', line_thickness=3)
                        # -------------------------------
                        # 两个分支
                        # 1、处理1:1的框, im0原始图片，xyxy：tensor类型的坐标-》转换成int类型
                        res1 = square_crop(im0, box, save_path, index, save_img=save_img)
                        print(res1['box'])
                        result['1040_1040'] = res1

                        # # 结果框-------------------------
                        # plot_one_box(res1['box'], im0, color=[0, 225, 225], label='result1', line_thickness=3)
                        # -------------------------------
                        # 2、处理440 * 852
                        res2 = right_crop(im0, box, save_path, index, save_img=save_img)
                        # plot_one_box(res2['box'], im0, color=[225, 225, 0], label='result2', line_thickness=3)
                        # cv2.imwrite(save_path.split('.')[-2] + '_' + 'plot.jpg', im0)
                        result['440_852'] = res2
                        index += 1
                        results.append(result)
                    else:
                        print('输入的图片中找不到人')

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')
        print('number of person: ', len(results))
        print(f'Done. ({time.time() - t0:.3f}s)')

    return results


if __name__ == '__main__':
    path = './data/test_data/'  # 路径
    # labeled_path = './data/labeled'  # 存储预标签的图片
    crop_store_path = './data/crop_imgs'  # 裁剪的图片
    save_cropped = True

    # 如果文件不存在的话就创建
    # if not os.path.exists(labeled_path):
    #     os.mkdir(labeled_path)

    # 这里单张图片和多张图片都可以处理
    t = time.time()
    res = infer(path, crop_store_path, save_cropped)
    print(f'infer. ({time.time() - t:.3f}s)')
