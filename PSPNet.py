'''
修改自以下代码：
版权声明：本文为CSDN博主「樊川_Minus Type」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/boycetoon29/article/details/117263163
'''


# 导入需要使用的包
import os
import mxnet as mx
from mxnet import image, gpu
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete,plot_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

# 忽略警告
import warnings; warnings.filterwarnings(action='once') 
warnings.filterwarnings("ignore")

# 设定使用GPU或者CUP进行计算，没有安装GPU版本的MXNet请使用CPU
ctx = mx.gpu(0)

# 定义函数对单张图片进行图像分割，并将结果存为pd.Series
def get_seg(imagepath):
    # 下载模型，这里使用的是在Cityscapes数据集上预训练的PSPnet模型
    model = gluoncv.model_zoo.get_model('psp_resnet101_citys', ctx=ctx, pretrained=True)
    img = image.imread(imagepath)
    img = test_transform(img,ctx=ctx)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    # 定义Cityscapes数据集分割标签字典
    col_map = {0:'road', 1:'sidewalk', 2:'building', 3:'wall', 4:'fence', 5:'pole', 6:'traffic light',
               7:'traffic sign', 8:'vegetation', 9:'terrain', 10:'sky', 11:'person', 12:'rider',
               13:'car', 14:'truck', 15:'bus', 16:'train', 17:'motorcycle', 18:'bicycle'}
    pred = []
    for i in range(19):
        pred.append((len(predict[predict==i])/(predict.shape[0]*predict.shape[1])))
    pred = pd.Series(pred).rename(col_map)
    seg_value = pred [0:5] + pred[10:]
    return seg_value



