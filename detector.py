# from darknet import *
import cfg
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from module import *
import torch

class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = Darknet53()
        self.net.eval()
        self.net.cuda()
        self.net.load_state_dict(torch.load(r'C:\Users\34801\Desktop\YPLOV3_esay\netkl_param.pt'))

    def forward(self, input, thresh, anchors):
        output_13, output_26, output_52 = self.net(input)
        output_13 = output_13.cpu()
        output_52 = output_52.cpu()
        output_26 = output_26.cpu()
        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = torch.Tensor(self._parse(idxs_13, vecs_13, 32, anchors[13]))

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = torch.Tensor(self._parse(idxs_26, vecs_26, 16, anchors[26]))

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = torch.Tensor(self._parse(idxs_52, vecs_52, 8, anchors[52]))

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        # output[1,13,13,3,85]
        mask = output[..., 0] > thresh
        #
        idxs = mask.nonzero()#是一个二维的0维表示被选出来的框1维表示[图片的索引，y,x,框的索引（最多到2，因为只有3个）]
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors)
        # n = idxs[:, 0]  # 所属的图片
        n = vecs[:, 0]#置信度
        a = idxs[:, 3]  # 建议框

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x

        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])
        if vecs.size(0) != 0:
            cls = torch.argmax(vecs[:,5:],dim=1).detach().numpy()
            return np.stack([n.detach().numpy(), cx.detach().numpy(), cy.detach().numpy(), w.detach().numpy(), h.detach().numpy(),cls],axis=1)
        else:
            return np.zeros(shape=(0,6))

def Iou(box,boxs,isMin=True):
    w = box[3].detach().numpy()
    h = box[4].detach().numpy()
    x1 = box[1].detach().numpy() - w / 2
    y1= box[2].detach().numpy() - h / 2
    x2 = x1 + w
    y2 = y1 + h
    w_ = boxs[:,3].detach().numpy()
    h_ = boxs[:,4].detach().numpy()
    x1_ = boxs[:,1].detach().numpy() - w_ / 2
    y1_ = boxs[:,2].detach().numpy() - h_ / 2
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

def nms(boxs,isMin=True):
    iou = boxs[:,0]
    index = (-iou).argsort()
    boxs = boxs[index]
    boxes = []
    while boxs.shape[0] > 0:
        max_box = boxs[0]
        boxes.append(max_box)
        boxs = boxs[1:]
        IOU = Iou(max_box,boxs,isMin)
        idx = np.where(IOU < 0.4)
        boxs = boxs[idx]
    return boxes

def nms1(boxs,isMin=True):
    iou = boxs[:,0]
    index = (-iou).argsort()
    boxs = boxs[index]
    boxes = []
    while boxs.shape[0] > 0:
        max_box = boxs[0]
        boxes.append(max_box)
        boxs = boxs[1:]
        IOU = Iou(max_box,boxs,isMin)
        idx = np.where(IOU < 0.4)
        boxs = boxs[idx]
    return boxes

if __name__ == '__main__':
    f = open(r'coco.names')
    names = f.read()
    names = names.split('\n')
    detector = Detector()
    image = Image.open(r'C:\Users\34801\Desktop\YPLOV3_esay\data\images\COCO_train2014_000000000025.jpg')
    image = image.resize((416,416))
    imgdraw = ImageDraw.Draw(image)
    imgfont = ImageFont.truetype(r'C:\Users\34801\Desktop\YPLOV3_esay\arial.ttf',25)
    img = np.array(image) / 255
    img = np.reshape(img, [1, 416, 416, 3])
    img = np.transpose(img, [0, 3, 1, 2])
    img = torch.Tensor(img)
    img = img.cuda()
    boxs = detector(img, 0.377, cfg.ANCHORS_GROUP)
    boxs = nms(boxs,isMin=False)
    # bboxs = []
    # for box in boxs:
    #     box = box.detach().numpy()
    #     bboxs.append(box)
    # bboxs = torch.Tensor(bboxs)
    # boxs = nms1(bboxs, isMin=True)
    # bboxs = []
    # for box in boxs:
    #     box = box.detach().numpy()
    #     bboxs.append(box)
    # bboxs = torch.Tensor(bboxs)
    # boxs = nms(bboxs, isMin=False)
    for box in boxs:
        w = int(box[3])
        h = int(box[4])
        x1 = int(box[1]) - w // 2
        y1 = int(box[2]) - h // 2
        x2 = x1 + w
        y2 = y1 + h
        imgdraw.rectangle((x1, y1, x2, y2), outline='red')
        imgdraw.text((x1, y1),names[int(box[5].item())], "yellow",font=imgfont)
        imgdraw.text((x1, y1), str(box[0].item()), "black")
    image.show()