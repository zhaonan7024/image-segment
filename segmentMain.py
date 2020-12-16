# -*- ecoding: utf-8 -*-
# @ModuleName: segmentMain.m
# @Author: Nancy
# @Time: 2020/12/13 21:37
import scipy.io
import PIL.Image as image
from numpy import *
import numpy as np
from machineLearning.ImageSegment.kmeans import compareSegmentations
import time
import ImageSegment.kmeans as ks

print("=====================")
ks.createTextons(1, 2, 3)
print("=====================")
img_name = 'planets'
origIm = array(image.open('pics/' + img_name + '.jpg'), 'f')
data = scipy.io.loadmat('./pics/imStack.mat')  # 读取mat文件
imStack = data['imStack']
data = scipy.io.loadmat('./pics/filterBank.mat')  # 读取mat文件
bank = data['F']
# 生成纹理基元编码集
textons = createTextons(imStack, bank, 10)
winSize = 6
numColorRegions = 6
numTextureRegions = 6
colorLabelIm, textureLabelIm = compareSegmentations(origIm, bank, textons, 10, numColorRegions, numTextureRegions)
uuid_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
print('colorLabelIm', colorLabelIm)
print('textureLabelIm', textureLabelIm)
np.savetxt('./pics/result_color_' + img_name + '_w10_' + uuid_str + '.txt', colorLabelIm)
np.savetxt('./pics/result_texton_' + img_name + '_w10_' + uuid_str + '.txt', textureLabelIm)
