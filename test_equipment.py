#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/28 17:03
# @Author  : lx-rookie
# @File    : test_01.py
# -*- coding: UTF-8 -*-
import base64
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 这边直接在这里指定一张卡
from make_labelme import two_point
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


import cv2
import numpy as np
import torch

from clean_classifier import infer
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF

sim_parse = {
    'cfg': r'./models/yolov5s.yaml',
    'weights': r'./runs/train/idcard_text_detect5/weights/best.pt',
    'sim_gpu_memeory': 0.1,
    'device': '0',
    'conf_thres': 0.4,
    'iou_thres': 0.3
}


class SimDetectModel():

    def __init__(self, weights=sim_parse['weights'], sim_gpu_memeory=sim_parse['sim_gpu_memeory'],
                 device=sim_parse['device'], new_size=(800,640), stride=32):
        # self.g = tf.Graph()
        # soft_config = tf.ConfigProto()
        # soft_config.gpu_options.per_process_gpu_memory_fraction = sim_gpu_memeory  # 设置显存占用不超过50%
        # self.sess = tf.Session(config=soft_config, graph=self.g)
        # KTF.set_session(self.sess)
        self.new_size = new_size
        self.stride = stride
        self.device = select_device(device, batch_size=1)


        # self.model = Model(cfg=cfg, ch=3, nc=12).cuda()
        # self.model.load_state_dict(torch.load(weights), map_location='cuda:0')
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model

        # 如果设备不是cpu并且gpu数目为1，则将模型由Float32转为Float16，提高前向传播的速度
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        self.model.eval()
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, 800, 640).to(self.device).type_as(next(self.model.parameters())))

    def pred(self, img_ori, augment=False, conf_thres=sim_parse['conf_thres'], iou_thres=sim_parse['iou_thres'],
             lb=None):
        # resultImgs = []
        h, w = img_ori.shape[:2]
        img, ratio, dw = letterbox(img_ori, self.new_size, stride=self.stride)  # 这一段就是使用边框填充
        # img = cv2.resize(img_ori, tuple(self.new_size))
        img = np.stack([img, ], 0)
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)  # 内存连续存储的数组
        img = torch.from_numpy(img).type(torch.FloatTensor)  # 将数组转换成numpy
        img = img.to(self.device, non_blocking=True)  # 加载到GPU锁页缓存中
        img = img.half() if self.half else img.float()  # 图片也由Float32->Float16
        img /= 255.0  # 归一化： 0 - 255 to 0.0 - 1.0
        # 前向传播
        # 第一个值：out为预测结果, 第二个值_训练结果
        out, _ = self.model(img, augment=augment)
        out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
        pred = out[0]
        scale_coords(img[0].shape[1:], pred[:, :4], (h, w), (ratio, dw))
        return pred.tolist()


print('*****************加载YOLOV5检测模型********************')
sim_detect_model = SimDetectModel()


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


# 裁剪图片
def crop_imgs(path1, img_name, img, box, is_save=False):
    # 这边可能有多个图片
    imgs_cls = []

    h = int(box[3]) - int(box[1])
    w = int(box[2]) - int(box[0])
    # [0.0, 240.625, 713.75, 550.0, 0.80810546875, 1.0]]
    img1 = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
    if is_save:
        # 这个是原始图片上的坐标数值
        # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        # cv2.imwrite(path1 + str(int(box[-1])) + '_box_' + img_name, img)
        cv2.imwrite(path1 + str(int(box[-1])) + '_' + img_name, img1)
    imgs_cls.append((img1, box[-1]))
    return imgs_cls






if __name__ == '__main__':
    path = './imgs/'  # 路径
    path1 = ''  # 存储预标签的图片
    path2 = './imgs/'  # 裁剪的图片
    weight_path = ''  # 分类模型权重的路径
    pre_label = False  # 是否要预标签操作
    is_save_cropped = True
    is_test = False

    if not os.path.exists(path1):
        os.mkdir(path1)

    img_name_list = os.listdir(path)
    # print(img_name_list)
    for img_name in img_name_list:
        # 不是jpg的图片就不处理
        if 'jpg' not in img_name:
            continue

        img = cv2.imread(path + img_name)
        with open(path + img_name, 'rb') as f:
            imgb64 = str(base64.b64encode(f.read()), 'utf-8')

        h, w, _ = img.shape

        img1 = cv2.resize(img, (800, 640))  # 这里转换成模型输入的大小

        boxes = sim_detect_model.pred(img1)
        # boxes = [[0.0, 240.625, 713.75, 550.0, 0.80810546875, 1.0]]

        crop_imgs_list = []
        cls_res_list = []  # 考虑一张图片可能有多个终端设备物体
        # 如果使用预标注，那么就执行这个功能
        if pre_label:
            make_prelabel(path1, img_name, img, boxes, h, w, imgb64)
        else:
            # TODO 判断是否有检测到东西
            if len(boxes) == 0:

                print('检测不到任何东西，是虚假照片')
            else:

                # 否则就是裁剪图片然后进行后面的分类
                for box in boxes:
                    if box[-1] == 1:
                        crop_imgs_list = crop_imgs(path2, img_name, img, box, is_save=is_save_cropped)
                        # 如果检测后需要后面的分类操作
                        for crop_img, cls in crop_imgs_list:
                            # 裁剪的图片传入分类网络
                            cls_res = infer(crop_img, weight_path, 1)   # str
                            cls_res_list.append(cls_res)
                            # crop_img[0] 是图片， crop_img[1]是类别
                            # if crop_img[-1] == 1:
                            #     # TODO 对设备清洁度做分类
                            #     # 训练一个终端分类模型
                            #     pass
                            # 这边的返回应该是一个list
        if len(boxes) == 0:
            print('虚假照片检测不到东西，还是本来不是虚假照片检测不到东西')
        else:
            print(cls_res_list)




    # # import pandas as pd
    # # a = pd.read_excel('base.xlsx')
    # # num = len(a)
    # # for i in range(num):
    # #     # print()
    # #     if a.iloc[i, 2] == 1:
    # #         if 'IMG' in a.iloc[i, 0]:
    # #             img = cv2.imread(r'test/fht/' + a.iloc[i, 0])
    # #             boxes = sim_detect_model.pred(img)
    # #             if len(boxes) > 0:
    # #                 # print(boxes[0])
    # #                 if boxes[0][-1] == 0:
    # #                     a.iloc[i, 2] = 0
    # #         else:
    # #             img = cv2.imread(r'test/ht/' + a.iloc[i, 0])
    # #             boxes = sim_detect_model.pred(img)
    # #             if len(boxes) > 0:
    # #                 # print(boxes[0])
    # #                 if boxes[0][-1] == 0:
    # #                     a.iloc[i, 2] = 0
    # # a.to_excel('base_1.xlsx', index=False)
    #
    # path = './test_img/'
    # path1 = './test_img1/'
    # # path2 = '/home/ctff/Lx/data/sim/labels_4/'
    # a = os.listdir(path)
    # print(len(a))
    # d = 0
    # for j in a:
    #     if 'jpg' not in j:
    #         continue
    #     # if j != '1589794863.jpg':
    #     #     continue
    #     print(j)
    #     img = cv2.imread(path + j)
    #     # img = cv2.resize(img, (640, 640))
    #     with open(path + j, 'rb') as f:
    #         imgb64 = str(base64.b64encode(f.read()), 'utf-8')
    #     h,w,_ = img.shape
    #     # print(ss)
    #     img = cv2.resize(img, (640, 416))
    #     print(img.shape)
    #     boxes = sim_detect_model.pred(img)
    #     print(boxes)
    #     two_point_label = {
    #         "version": "5.0.1",
    #         "flags": {},
    #         "shapes": [
    #         ],
    #         "imagePath": "1评标索引表_0.png",
    #         "imageData": "",
    #         "imageHeight": 1684,
    #         "imageWidth": 1191
    #     }
    #     # try:
    #     d1 = 0
    #     for i in boxes:
    #         # print(i)
    #         if i[-1] != 4:
    #             continue
    #         box = np.array(i[:4]).astype(np.int32)
    #         box = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    #         box = np.array(box)
    #         cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
    #         tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    #         tf = max(tl - 1, 1)  # font thickness
    #         cv2.putText(
    #             img,
    #             str(i[-1]),
    #             (int(i[0]), int(i[1]) - 2),
    #             0,
    #             tl / 3,
    #             [225, 0, 0],
    #             thickness=tf,
    #             lineType=cv2.LINE_AA,
    #         )
    #     cv2.imwrite(path1 + j, img)
        # for box in boxes:

            # if box[-1] == 1:
            #     d1 += 1
            #     h = int(box[3]) - int(box[1])
            #     w = int(box[2]) - int(box[0])
            #     img1 = img[(int(box[1]) - h // 10):(int(box[3]) + h // 10), (int(box[0]) - w // 20):(int(box[2]) + w // 20), :]
            #     print(img1.shape)
            #     cv2.imwrite(path1 + str(d1) + '_' + j, img1)
        #     two_point_label['shapes'].append({
        #         "label": str(int(box[-1])),
        #         "points": [
        #             [
        #                 box[0],
        #                 box[1]
        #             ],
        #             [
        #                 box[2],
        #                 box[3]
        #             ]
        #         ],
        #         "group_id": None,
        #         "shape_type": "rectangle",
        #         "flags": {}
        #     })
        #
        # two_point_label['imagePath'] = j
        # two_point_label['imageData'] = imgb64
        # two_point_label['imageHeight'] = h
        # two_point_label['imageWidth'] = w
        # with open(os.path.join(path, j.replace('.jpg', '.json')), 'w') as f:
        #     json.dump(two_point_label, f)
        # except:
        #     pass

