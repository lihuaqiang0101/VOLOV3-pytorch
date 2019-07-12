import os
from PIL import Image
import cfg
from multiprocessing import Pool

def create():
    base_dir = r'D:\数据集\COCO\labels\val2014'
    img_basedir = r'E:\images\val2014'
    lables_dir = r'D:\数据集\COCO\labels\val2014'
    labels = os.listdir(lables_dir)
    file = open('label.txt','a')
    for label in labels:
        img = Image.open(os.path.join(img_basedir,label.replace('txt','jpg')))
        img = img.resize((cfg.IMG_WIDTH,cfg.IMG_HEIGHT))
        img.save(os.path.join(r'C:\images',label.replace('txt','jpg')))
        print(label)
        file.write(label.replace('txt', 'jpg') + ' ')
        w,h = img.size
        # w_scale = w / cfg.IMG_WIDTH
        # h_scale = h / cfg.IMG_HEIGHT
        with open(os.path.join(base_dir,label)) as f:
            texts = [line.strip() for line in f.readlines()]
        for text in texts:
            boxs = text.split(' ')
            cls = boxs[0]
            x1 = int(cfg.IMG_WIDTH * (float(boxs[1]) - float(boxs[3]) / 2))
            y1 = int(cfg.IMG_HEIGHT * (float(boxs[2]) - float(boxs[4]) / 2))
            x2 = int(cfg.IMG_WIDTH * (float(boxs[1]) + float(boxs[3]) / 2))
            y2 = int(cfg.IMG_HEIGHT * (float(boxs[2]) + float(boxs[4]) / 2))
            # x1 = int(w * (float(boxs[1]) - float(boxs[3]) / 2))
            # y1 = int(h * (float(boxs[2]) - float(boxs[4]) / 2))
            # x2 = int(w * (float(boxs[1]) + float(boxs[3]) / 2))
            # y2 = int(h * (float(boxs[2]) + float(boxs[4]) / 2))
            w1 = int(x2 - x1)
            h1 = int(y2 - y1)
            cx = int(x1 + w1//2)
            cy = int(y1 + h1//2)
            file.write(cls+' '+str(cx)+' '+str(cy)+' '+str(w1)+' '+str(h1))
            file.write(' ')
        file.write('\n')
    file.close()

if __name__ == '__main__':
    pool = Pool(50)  # 创建25个线程
    pool.apply_async(create)  # 让每个线程都去执行downloadmovie函数，传递的参数为(i,)
    pool.close()  # 任务执行完毕以后就关闭线程
    pool.join()  # 等待线程结束