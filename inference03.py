from crop_img import get_img
from YoloModel import YoloModel
import cv2
import yaml

# 使用外部的配置文件，导入参数,这一块也要用作全局的
with open('crop_img.yaml', encoding='utf-8') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)  # 读取yaml文件

# 全局变量
yolo_model = YoloModel(configs)

def inference(path):
    img = cv2.imread(path)
    pred = yolo_model.predict(img)
    # 这边的判断是否有检测到物体需要在检测这部分做
    # 过滤小人的检测框也是在这部分做
    res = get_img(pred, img, configs)
    return res
if __name__ == '__main__':

    res = inference(configs['path'])