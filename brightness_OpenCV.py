# URL: https://blog.csdn.net/weixin_42357472/article/details/115011587

import cv2
import numpy as np

## 色调（H），饱和度（S），明度（V）

def image_v(imagepath):
    image = cv2.imread (imagepath)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    #print(H, S, V)
    v = V.ravel()[np.flatnonzero(V)]   #亮度非零的值
    average_v  = sum(v)/len(v)
    return average_v
