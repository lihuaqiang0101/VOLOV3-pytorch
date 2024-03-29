
IMG_HEIGHT = 416
IMG_WIDTH = 416

CLASS_NUM = 80

#定义每个特征层上的每种建议框的宽和高
# 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
ANCHORS_GROUP = {
    13: [[116,90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}
#得到每种尺寸下特征图上的每一个建议框的面积
ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
