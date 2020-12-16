# -*- ecoding: utf-8 -*-
# @ModuleName: kmeans
# @Author: Nancy
# @Time: 2020/12/2 20:30
from numpy import *
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
import time
from scipy import signal
import scipy.io

"""
按照颜色切割 
按照纹理切割
"""

# 读取图片
def load_data(file_path):
    f = open(file_path, 'rb')  # 二进制打开
    data = []
    img = image.open(f)  # 以列表形式返回图片像素值
    h, w = img.size  # 活的图片大小,height,weight
    data = array(img.getdata())
    featIm = data.reshape(h, w, 3)
    f.close()
    return featIm, h, w  # 以矩阵型式返回data，图片大小


def distEclud(vecA, vecB):
    """
    :param vecA:
    :param vecB:
    :return: 两向量之间的欧式距离
    """
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(dataSet, k):
    """
    :param dataSet:
    :param k: k,cluster centers
    :return:meanFeats,为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
    """
    n = dataSet.shape[2]  # 列的数量，即数据的特征个数
    centroids = mat(zeros((k, n)))  # 创建k个质心矩阵
    for j in range(n):  # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataSet[:, :, j].reshape(-1, 1))  # 最小值
        rangeJ = float(max(dataSet[:, :, j].reshape(-1, 1)) - minJ)  # 范围 = 最大值 - 最小值
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  # 随机生成，mat为numpy函数，需要在最开始写上 from numpy import *
    return centroids


def rgb2gray(rgb):
    """
    :param rgb: RGB图像
    :return:灰度图像
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def quantizeFeats(featIm, meanFeats):
    """
    :param featIm: 特征矩阵,[]hwd,h:原始图片的height,w：原始图片的weight,d：d denotes the dimensionality of the feature vector already computed for each of its pixels
    :param meanFeats: 均值矩阵，[]dk，k,cluster centers,each of which is a d-dimensional vector (a row in the matrix)
    :return:labelIm,[]hw,matrix of integers indicating the cluster membership (1…k) for each pixel,隶属度矩阵
    """
    h = featIm.shape[0]
    w = featIm.shape[1]
    k = meanFeats.shape[0]
    labelIm = ones((h, w))
    for i in range(h):
        for j in range(w):
            min = 0
            minDist = distEclud(featIm[i][j][:], meanFeats[min][:])
            for t in range(k):
                dist = distEclud(featIm[i][j][:], meanFeats[t][:])
                if (minDist > dist):
                    min = t
                    minDist = dist
            labelIm[i][j] = min
    return labelIm


def createTextons(imStack, bank, k):
    """
    :param imStack:n,包含n个二值图像
    :param bank:[]mmd,滤波器的大小是mm，d个滤波器,d个特征
    :param k:cluster
    :return:textons,生成纹理基元编码集,[]kd,每一行代表一个纹理特征
    """
    row = imStack.shape[1]
    d = bank.shape[2]  # 滤波器个数
    all_image_features = []
    for i in range(row):
        im = imStack[:, i][0]
        d_features = np.zeros([im.shape[0], im.shape[1], d])
        for j in range(d):
            filter_result = signal.convolve2d(im, bank[:, :, j], mode="same")
            d_features[:, :, j] = filter_result
        x, y, d = d_features.shape[0], d_features.shape[1], d_features.shape[2]
        d_features = d_features.reshape([x * y, d])
        if (all_image_features == []):
            all_image_features = d_features
        else:
            all_image_features = np.r_[all_image_features, d_features]  # 沿着矩阵行拼接
    print('all_image_features', all_image_features.shape)
    clf = KMeans(n_clusters=k)
    clf.fit_predict(all_image_features)
    textons = clf.cluster_centers_
    print('textons', textons.shape)
    return textons


def extractTextonHists(origIm, bank, textons, winSize):
    """
    :param origIm:将原始灰度图像origIm
    :param bank::[]mmd,滤波器的大小是mm，d个滤波器,d个特征
    :param textons: k*d ,有 k 个纹理基元,每一行代表一个纹理特征
    :param winSize:定义在固定大小的winSize 内的局部窗口
    :return:featIm ,r*c*k,将原始灰度图像origIm，使用滤波器组进行过滤，得到一个r*c*38 的矩阵
    """
    row, col = origIm.shape[0], origIm.shape[1]
    k = textons.shape[0]
    d = bank.shape[2]
    responses = zeros([row, col, d])
    print('origIm', origIm.shape)
    print('bank', bank.shape)
    for r in range(d):
        responses[:, :, r] = signal.convolve2d(origIm, bank[:, :, r], mode="same")
        # responses[:, r] = np.max(data11, axis=1)
    # 计算 滤波器组响应 与 纹理基元编码集 的距离，并生成隶属度矩阵 feattexton
    print('responses', responses.shape)
    feattexton = zeros([row, col])
    for i in range(row):
        for j in range(col):
            minD = distEclud(responses[i, j, :], textons[0, :])
            min = 0
            for t in range(k):
                if (minD > distEclud(responses[i, j, :], textons[t, :])):
                    min = t
            feattexton[i, j] = min
    # 图像边界处理
    # 生成柱状纹理图
    featIm = zeros([row, col, k])
    if winSize > 1:
        colNumLeft = math.floor((winSize - 1) / 2)
        colNumRight = math.ceil((winSize - 1) / 2)
    else:
        colNumLeft = 0
        colNumRight = 0
    # 生成柱状纹理图
    for i in range(row):
        for j in range(col):
            left = (i - colNumLeft) if (i - colNumLeft) > 0 else 0
            down = (j - colNumLeft) if (j - colNumLeft) > 0 else 0
            right = (i + colNumRight) if ((i + colNumRight) < row) else row
            up = (j + colNumRight) if ((j + colNumRight) < col) else col
            data = feattexton[left:right, down:up]
            for t in np.unique(data):
                a = np.sum(data == t)
                featIm[i, j, int(t)] = a
    return featIm


def compareSegmentations(origIm, bank, textons, winSize, numColorRegions, numTextureRegions):
    """
    :param origIm:
    :param bank:
    :param textons:
    :param winSize:
    :param numColorRegions:
    :param numTextureRegions:
    :return:colorLabelIm 和textureLabelIm 是h*w 的矩阵
    """
    # 生成颜色分割的标签矩阵
    x, y, z = origIm.shape[0], origIm.shape[1], origIm.shape[2]
    colordata = origIm.reshape(x * y, z)
    clf = KMeans(n_clusters=numColorRegions)
    clf.fit_predict(colordata)
    meanFeats = clf.cluster_centers_
    colorLabelIm = quantizeFeats(origIm, meanFeats)
    # 获取纹理柱状图，计算纹理特征，生成基于纹理分割的标签矩阵
    origIm_gray = rgb2gray(origIm)
    texton_histogram = extractTextonHists(origIm_gray, bank, textons, winSize)
    h, w, k = texton_histogram.shape[0], texton_histogram.shape[1], texton_histogram.shape[2]
    texton = texton_histogram.reshape(h * w, k)
    clf2 = KMeans(n_clusters=numTextureRegions)
    clf2.fit_predict(texton)
    textureCenter = clf2.cluster_centers_
    textureLabelIm = quantizeFeats(texton_histogram, textureCenter)
    textureLabelIm = textureLabelIm.reshape([h, w])
    return colorLabelIm, textureLabelIm


# img = array(image.open('pics/coins.jpg').convert('1'), 'f')
# img_L = array(image.open('pics/coins.jpg').convert('L'), 'f')

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

colorLabelIm, textureLabelIm = compareSegmentations(origIm, bank, textons, 20, numColorRegions, numTextureRegions)
uuid_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
print('colorLabelIm', colorLabelIm)
print('textureLabelIm', textureLabelIm)
np.savetxt('./pics/result_color_' + img_name + '_w20_' + uuid_str + '.txt', colorLabelIm)
np.savetxt('./pics/result_texton_' + img_name + '_w20_' + uuid_str + '.txt', textureLabelIm)

colorLabelIm, textureLabelIm = compareSegmentations(origIm, bank, textons, 30, numColorRegions, numTextureRegions)
uuid_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
print('colorLabelIm', colorLabelIm)
print('textureLabelIm', textureLabelIm)
np.savetxt('./pics/result_color_' + img_name + '_w30_' + uuid_str + '.txt', colorLabelIm)
np.savetxt('./pics/result_texton_' + img_name + '_w30_' + uuid_str + '.txt', textureLabelIm)
