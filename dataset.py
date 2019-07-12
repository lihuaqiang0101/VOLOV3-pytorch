import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import cfg
import os

from PIL import Image
from PIL import ImageDraw
import math

LABEL_FILE_PATH = r"C:\Users\34801\Desktop\YPLOV3_esay\label.txt"
IMG_BASE_DIR = r'C:\images'

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b

def Iou(box,boxs,isMin=False):
    w = box[2]
    h = box[3]
    x1 = box[0] - w / 2
    y1= box[1] - h / 2
    x2 = x1 + w
    y2 = y1 + h
    w_ = boxs[2]
    h_ = boxs[3]
    x1_ = boxs[0] - w_ / 2
    y1_ = boxs[1] - h_ / 2
    x2_ = x1_ + w_
    y2_ = y1_ + h_
    x_left = np.maximum(x1,x1_)
    y_left = np.maximum(y1,y1_)
    x_right = np.minimum(x2,x2_)
    y_right = np.minimum(y2,y2_)
    intener = (x_right - x_left) * (y_right - y_left)
    if isMin:
        union = np.minimum(w * h,w_ * h_)
    else:
        union = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intener
    return intener / union

class MyDataset(Dataset):

    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()[:8]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}

        line = self.dataset[index]
        strs = line.split()
        try:
            _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))
        except:
            return self.__getitem__(index + 1)
        # W,H = _img_data.size
        try:
            img_data = transforms(_img_data)
        except:
            return self.__getitem__(index + 1)
        # img_data = (img_data / 255 - 0.5)*2
        # imagedraw = ImageDraw.Draw(_img_data)
        try:
            _boxes = np.array([float(x) for x in strs[1:]])#得到框:
        except:
            return self.__getitem__(index + 1)
        #_boxes = [ 23. 320. 203. 140. 290.  23.  77. 375.  86.  54.]
        # _boxes = np.array(list(map(float, strs[1:])))
        try:
            boxes = np.split(_boxes, len(_boxes) // 5)#将每一个框分成一个array
        except:
            return self.__getitem__(index + 1)
        #遍历每一种尺寸的输出的anchors，它会将每种尺寸的特征层及其所对应的三种建议框封装成一个元组
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():#feature_size是尺寸大小，anchors是建议框尺寸列表
            #为每种尺寸的特征层构造其输出向量
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            for box in boxes:
                try:
                    box = list(box)
                except TypeError:
                    box = [0,0]
                if len(box) != 5:
                    continue
                cls, cx, cy, w, h = box
                # x1 = int(cx - w/2)
                # y1 = int(cy - h/2)
                # x2 = x1 + w
                # y2 = y1 + h
                # imagedraw.rectangle((x1,y1,x2,y2),outline='red')
                #将小数部分作为偏移，整数部分作为起始位置
                cx_offset, cx_index = math.modf(cx / cfg.IMG_WIDTH * feature_size )#相当于先让原始的框在原图上归一化，在返回到各个尺寸的特征图上
                #也就是相当于我这个点占了原位置上的%多少
                cy_offset, cy_index = math.modf(cy / cfg.IMG_HEIGHT * feature_size )
                for i, anchor in enumerate(anchors):#{13: [100, 91, 91], 26: [225, 216, 216], 52: [400, 450, 450]}
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    #p_w，p_h是为了后面计算偏移量，偏移量就是在此基础上取对数
                    p_w, p_h = w / anchor[0], h / anchor[1]#w是实际框的宽，anchor[0]是建议框的宽，h是实际框的高，anchor[1]是建议框的高
                    p_area = w * h
                    boxs = []
                    boxs.extend(box[1:3])
                    boxs.extend(anchor)
                    iou = Iou(np.array(box[1:]),np.array(boxs),isMin=False)
                    # iou = min(p_area, anchor_area) / max(p_area, anchor_area)#置信度必须小于1
                    #这是一个85维的向量表示每个建议框的置信度、四个偏移、类别的one-hot形式
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))])
        # _img_data.show()
        # print(labels[13].shape, labels[26].shape, labels[52].shape, img_data.shape)
        if img_data.shape[0] != 3:
            return self.__getitem__(index+1)
        else:
            return labels[13], labels[26], labels[52], img_data

# d = MyDataset()
# for i in range(8):
#     d.__getitem__(i)