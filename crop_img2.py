import cv2
import numpy as np
import copy
import yaml
from utils.plots import plot_one_box
import base64

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


def right_box(H, W, box, alpha=1.1):
    '''
    H:原始图片的高
    W：原始图片的宽
    box:[x,y,x,y]->int
    '''
    pad = 0
    w = box[2] - box[0]
    size = configs['size2']
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
            pad = 0 if int((res - box[1]) / 2) < 60 else int((res - box[1]) / 2)
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
    box[0] = box[0] - pad_w
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


def crop_resize(path1, img, box, index, size=(440, 852), save_img=False, pad=0):
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
    if save_img:
        cv2.imwrite(f"{path1}_resize{size}_{str(index)}.jpg", img1)
    return img1


def get_right_img(img, box, path1, index, size=(440, 852), save_img=False):
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
    box, pad = right_box(H, W, box, alpha=configs['alpha'])  # 处理之后的裁剪框
    # 裁剪
    crop_img = crop_resize(path1, img, box, index, size=size, save_img=save_img, pad=pad)
    # crop_img = np.ascontiguousarray(crop_img)
    # return crop_img
    result = {'size': configs['size2']}
    imgb64 = str(base64.b64encode(np.ascontiguousarray(crop_img)), 'utf-8')
    result['imgb64'] = imgb64
    result['box'] = box
    result['id'] = index
    return result

def get_square_img(img, box, path1, index, size=(1024, 1024), save_img=False):
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
    crop_img = crop_resize(path1, img, box1, index, size=size, save_img=save_img)
    # crop_img = np.ascontiguousarray(crop_img)
    # return crop_img
    result = {'size': configs['size1']}
    imgb64 = str(base64.b64encode(np.ascontiguousarray(crop_img)), 'utf-8')
    result['imgb64'] = imgb64
    result['box'] = box1
    result['id'] = index
    return result

def get_img(pred, img):
    '''
    input
        pred：预测框 [x, y, x, y, conf, class]
        img:原始图片
    return
        [{img1, img2}, {}, {}...]
    '''
    results = []
    result = {}
    for det in pred:
        index = 0
        for *xyxy, conf, cls in reversed(det):
            if cls == 0.0:  # 如果是person类，那么就裁剪图像
                # 这里做一个判断就是非常小的人就不用裁剪
                _, _, w, h = get_xywh(xyxy)  # 裁剪框的宽高
                if w < 100 or h < 100:
                    continue
                H, W, _ = img.shape
                # 统一处理成int类型， 并且扩大box
                box, w, h = box_same_process(H, W, xyxy, alpha=configs['alpha'])
                # 两个分支
                # 1、处理1:1的框, im0原始图片，xyxy：tensor类型的坐标-》转换成int类型
                res1 = get_square_img(img, box, configs['save_path'], index, size=configs['size1'], save_img=configs['save_img'])
                result['square_img'] = res1
                # 2、处理440 * 852
                res2 = get_right_img(img, box, configs['save_path'], index, size=configs['size2'], save_img=configs['save_img'])
                result['right_img'] = res2
                index += 1
                results.append(result)
                # 画预测框-------------------------
                plot_one_box(xyxy, img0, color=[0, 0, 0], label='predict', line_thickness=3)
                # 画扩大框
                plot_one_box(box, img0, color=[0, 225, 0], label='box_expand', line_thickness=3)
                # # 画结果框-------------------------
                plot_one_box(res1['box'], img0, color=[0, 225, 225], label='result1', line_thickness=3)
                # -------------------------------
                plot_one_box(res2['box'], img0, color=[225, 225, 0], label='result2', line_thickness=3)
                cv2.imwrite(configs['save_path'] + '_' + 'plot.jpg', img0)
                print(configs['save_path'].split('.')[-2] + '_' + 'plot.jpg')
            else:
                print('输入的图片中找不到人')
                # ----------------找不到图片报错
        return result


import torch
from utils.general import set_logging, check_img_size, non_max_suppression
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.datasets import letterbox


def detect(source):
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

    # ---------load img and preprocess-------------
    img0 = cv2.imread(source)

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # Run inference
    if device.type != 'cpu':  # 如果设备是GPU的运行
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # img, img0是需要的
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=configs['augment'])[0]

    # Apply NMS
    pred = non_max_suppression(pred, configs['conf_thres'], configs['iou_thres'], classes=configs['classes'],
                               agnostic=configs['agnostic_nms'])
    return pred, img, img0

if __name__ == "__main__":
    # 使用外部的配置文件，导入参数
    with open('inference.yaml', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)  # 读取yaml文件

    from utils.general import scale_coords
    pred, img, img0 = detect(configs['path'])
    # 这边的判断是否哟检测到物体需要在检测这部分做
    print(len(pred))
    print(pred)
    if len(pred):
        for det in pred:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        get_img(pred, img0)
