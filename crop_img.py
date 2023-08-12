'''
该文件仅仅是支持裁剪用的。
调用get_img(pred, img0)， 传入预测框预处理好的预测框，
检测代码需要外部传入
'''
import os.path

import cv2
import numpy as np
import copy
import yaml


# 输入预测框，和图像，给到裁剪后的图片
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


def box_same_process(H, W, xyxy, size, alpha=1.1):
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
    if int((w * size[1]) / size[0]) < h:
        alpha = 1.05
    box, w, h = box_expand_2(H, W, box, alpha=alpha)  # 返回的元素都是int类型

    return box, w, h,


def box_expand(H, W, box, alpha=1.1):
    xywh = get_xywh(box)
    w1 = xywh[2] * alpha
    h1 = xywh[3] * alpha
    if xywh[1] + h1 > H:
        h1 = xywh[3] + (w1 - xywh[2])
    box = get_xyxy([xywh[0], xywh[1], w1, h1])
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
    box[0] = int(max(box[0], 0))
    box[1] = int(max(box[1], 0))
    box[2] = int(min(box[2], W))
    box[3] = int(min(box[3], H))

    return box, int(w1), int(h1)


def right_box(H, W, box, size, pad=60):
    '''
    H:原始图片的高
    W：原始图片的宽
    box:[x,y,x,y]->int
    '''
    pad = 0
    w = box[2] - box[0]
    r_h = int((w * size[1]) / size[0])
    if box[1] + r_h < H:
        box[3] = box[1] + r_h
    else:
        box[3] = H
        res = (box[1] + r_h) - H
        if box[1] - res > 0:
            box[1] = box[1] - res
        else:
            # 否则需要上下pad
            pad = 0 if int((res - box[1]) / 2) < pad else int((res - box[1]) / 2)
            box[1] = 0
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])], pad


def square_box(box, H, W):
    '''
    1:1 裁剪框处理
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
        box[3] = box[1] + (box[2] - box[0])
    else:
        h1 = box[1]
        h2 = H - box[3]
        pad_h = int((w_e - h) / 2)  # 这里需要整型
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


def crop_resize(img, box, size=(440, 852), save_path='/', save_img=False, pad=0):
    '''
    裁剪并且resize图像
    box:[x,y,x,y]->Torch.tensor
    labeled_path: only for test
    '''
    img1 = img[box[1]: box[3], box[0]: box[2], :]
    if pad:
        # 纯色填充
        img1 = cv2.copyMakeBorder(img1, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(225, 225, 225))
    img1 = cv2.resize(img1, size)
    img1 = np.ascontiguousarray(img1)
    if save_img and len(save_path) != 0:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cv2.imwrite(f"{save_path}_resize{size}.jpg", img1)
    return img1


def get_right_img(img, box, size=(440, 852), pad=60, save_path="/", save_img=False):
    '''
    裁剪成440*852
    input
    img: 原始图片->np
    box：expand之后的预测框坐标->[int,int,...] xyxy格式
    path：裁剪图片存储的路径
    index：图片中第几个人
    size：默认设置为440*852
    save_img：是否保存图片，默认魏False
    return
    result['imgb64'] = imgb64  # base64类型的数据
    result['box'] = box  # 调整后的框
    result['id'] = index # 第几个人
    '''
    H, W, _ = img.shape
    # 先框做处理在裁剪
    box, pad = right_box(H, W, box, size=size, pad=pad)  # 处理之后的裁剪框
    # 裁剪
    crop_img = crop_resize(img, box, size=size, save_path=save_path, save_img=save_img, pad=pad)
    return crop_img


def get_square_img(img, box, size=(1024, 1024), save_path="/", save_img=False):
    '''
    1:1的方式裁剪
    input
    img: 原始图片->np
    box：expand之后的预测框坐标->[int,int,...] xyxy格式
    path：裁剪图片存储的路径
    index：图片中第几个人
    size：默认设置为1040*1040
    save_img：是否保存图片，默认魏False
    return
    result['imgb64'] = imgb64  # base64类型的数据
    result['box'] = box  # 调整后的框
    result['id'] = index # 第几个人
    '''
    box1 = copy.deepcopy(box)  # 你传入的时候这个框变量就改变了
    H, W, _ = img.shape
    box1 = square_box(box1, H, W)
    crop_img = crop_resize(img, box1, size=size, save_path=save_path, save_img=save_img)
    return crop_img


def get_img(pred, img, configs):
    '''
    input
        pred：预测框 [x, y, x, y]
        img:原始图片
    return
        [img1, img2]
    '''
    results = []
    # 这里做一个判断就是非常小的人就不用裁剪
    # _, _, w, h = get_xywh(pred)  # 裁剪框的宽高
    H, W, _ = img.shape
    # 统一处理成int类型， 并且扩大box
    box, w, h = box_same_process(H, W, pred, size=configs['size2'], alpha=configs['alpha'])
    # 两个分支
    # 1、处理1:1的框, img原始图片，xyxy：tensor类型的坐标-》转换成int类型
    res1 = get_square_img(img, box, size=configs['size1'], save_path=configs['save_path'], save_img=configs['save_img'])
    results.append(res1)
    # 2、处理440 * 852
    res2 = get_right_img(img, box, size=configs['size2'], pad=configs['pad'], save_path=configs['save_path'],  save_img=configs['save_img'])
    results.append(res2)

    return results






if __name__ == "__main__":
    # 使用外部的配置文件，导入参数,这一块也要用作全局的
    with open('crop_img.yaml', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)  # 读取yaml文件
    # 这边直接实例化
    from YoloModel import YoloModel
    yolo_model = YoloModel(configs)


    def inference(path):
        img = cv2.imread(path)
        pred = yolo_model.predict(img)
        # 这边的判断是否有检测到物体需要在检测这部分做
        # 过滤小人的检测框也是在这部分做
        res = get_img(pred, img, configs)
        return res

    # 预测
    res = inference(configs['path'])

