# VOLOV3-pytorch

![image](https://github.com/lihuaqiang0101/VOLOV3-pytorch/blob/master/2019040211084050.jpg)


损失的设计：
分四个损失：正样本iou使用kl散度，负样本使用二值交叉熵损失，位置使用mse，分类使用交叉熵，由于pytorch的交叉熵集成了softmax所以不使用softmax，而且pytorch的交叉熵损失不支持one-hot形式。
