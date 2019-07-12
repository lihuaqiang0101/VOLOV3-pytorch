import dataset
from Net import *
from module import *
from torch.utils.data import DataLoader
import torch
import os
import time
import torch.nn.functional as F

#在求损失的时候没有考虑类别
def loss_fn(output, target, alpha):#[2, 13, 13, 3, 85]target
    output = output.permute(0, 2, 3, 1)#换轴将C换在最后（因为网络输出的通道是15，表示3个框，每个框5个输出）
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)#将3个框和5个输出分开
    # target = target[...,:5]
    #选出有目标的框
    mask_obj = target[..., 0] > 0#...代表前面的除最后一个维度以外的所有维度
    #选出没有目标的框
    # mask_noobj = target[..., 0] == 0
    output = output.double()
    target = target.double()
    iou_out = output[mask_obj][:,:1]
    mask1 = iou_out < 0
    mask2 = iou_out > 0
    iou_target = target[mask_obj][:,:1]
    # 负的
    iou_out1 = iou_out[mask1]
    iou_target1 = iou_target[mask1]
    #正的
    iou_out2 = iou_out[mask2]
    iou_target2 = iou_target[mask2]
    posi_out = output[mask_obj][:,1:5]
    posi_target = target[mask_obj][:,1:5]
    # posi_out = output[mask_obj][:, :5]
    # posi_target = target[mask_obj][:,:5]
    cls_out = output[mask_obj][:,5:]
    cls_target = target[mask_obj][:,5:]
    # loss_position = torch.mean((posi_out - posi_target) ** 2)
    # loss_cls = torch.mean((cls_out - cls_target) ** 2)
    # loss_obj = torch.mean((iou_out - iou_target) ** 2)
    # print(loss_cls.item())
    LOSS1 = nn.BCELoss()
    LOSS2 = nn.MSELoss()
    LOSS3 = nn.KLDivLoss()
    # LOSS3 = F.cross_entropy()#nn.CrossEntropyLoss()
    loss_position = LOSS2(posi_out,posi_target)
    loss_obj = LOSS3(torch.log(F.sigmoid(iou_out2)), iou_target2 + LOSS3(torch.log(1 - F.sigmoid(iou_out2)),(1 - iou_target2)))
    loss_noobj = LOSS1(F.sigmoid(iou_out1), iou_target1)
    cls_out = cls_out.cpu()
    cls_target = cls_target.cpu()
    # cls_out = cls_out.argmax(dim=1)
    # cls_out = torch.reshape(cls_out,[-1,1])
    cls_target = cls_target.argmax(dim=1)
    # cls_target = torch.reshape(cls_target,[-1,1])
    # print(cls_out.shape,cls_target.long().shape)
    # print(cls_out , cls_target)
    # loss_cls = LOSS3(cls_out , cls_target)#
    loss_cls = F.cross_entropy(input=cls_out , target=cls_target.long())
    # loss_noobj = torch.mean((output[mask_noobj] - target[mask_noobj]) ** 2)
    # loss = 0.5 * loss_obj + 0.28 * loss_position + 0.22 * loss_cls#+ (1 - alpha) * loss_noobj这个不要加权
    # loss = 0.8 * loss_position + 0.2 * loss_cls
    # loss = loss_position + loss_cls
    # print(loss_cls.item(),loss_obj.item(),loss_position.item())
    loss = loss_obj + loss_position + loss_cls + loss_noobj
    return loss


if __name__ == '__main__':

    net = Darknet53()
    net = net.cuda()
    # net.load_state_dict(torch.load(r'C:\Users\34801\Desktop\YPLOV3_esay\params\6398\net_param.pkl'))
    myDataset = dataset.MyDataset()
    net.train()
    opt = torch.optim.Adam(net.parameters())
    starttime = time.time()
    day = 0
    for epoch in range(2000):
        train_loader = DataLoader(myDataset, batch_size=8, shuffle=True)
        for target_13, target_26, target_52, img_data in train_loader:
            img_data = img_data.cuda()
            # target_13 = target_13.cuda()
            # target_26 = target_26.cuda()
            # target_52 = target_52.cuda()
            output_13, output_26, output_52 = net(img_data)
            # torch.Size([2, 15, 13, 13]),output_13.shape
            # torch.Size([2, 13, 13, 3, 85]),target_13.shape
            # torch.Size([2, 15, 26, 26])
            # torch.Size([2, 26, 26, 3, 85])
            # torch.Size([2, 15, 52, 52])
            # torch.Size([2, 52, 52, 3, 85])
            output_13 = output_13.cpu()
            output_26 = output_26.cpu()
            output_52 = output_52.cpu()
            loss_13 = loss_fn(output_13, target_13, 0.9)
            loss_26 = loss_fn(output_26, target_26, 0.9)
            loss_52 = loss_fn(output_52, target_52, 0.9)
            loss = loss_13 + loss_26 + loss_52
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(epoch, loss.item())
        # if epoch > 450 and loss.item() < 0.0001:
    torch.save(net.state_dict(),'netklll_param.pt')
            # break
        #     endtime = time.time()
        #     T = int(endtime-starttime)
        #     #每3个小时保存一次权重
        #     if T % 10800 == 0:
        #         day += 1
        #         if not os.path.exists(r'params\第{}天'.format(day)):
        #             os.mkdir(r'params\第{}天'.format(day))
        #             torch.save(net.state_dict(), r'params\第{}天\net_param.pt'.format(day))
        # if epoch > 20:
        #     if not os.path.exists(r'params\{}'.format(epoch)):
        #         os.mkdir(r'params\{}'.format(epoch))
        #     torch.save(net.state_dict(), r'params\{}\net_param.pt'.format(epoch))
        # else:
        #     if (epoch + 1) % 200 == 0:
        #         if not os.path.exists(r'params\{}'.format(epoch)):
        #             os.mkdir(r'params\{}'.format(epoch))
        #         torch.save(net.state_dict(), r'params\{}\net_param.pt'.format(epoch))