import numpy as np
import cv2
import PIL.Image as image
"""
matlab 里使用了labels2rgb进行了标签矩阵与图片的转化，python没找到，网上找到个类似的函数，进行了调用展示
我希望我是个色盲
参考：
https://www.pythonheidong.com/blog/article/168788/ef29f48539ba89a411ce/
"""
def gen_lut():
  """
  Generate a label colormap compatible with opencv lookup table, based on
  Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
  appendix C2 `Pseudocolor Generation`.
  :Returns:
    color_lut : opencv compatible color lookup table
  """
  tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
  arr = np.arange(256)
  r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
  g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
  b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
  return np.concatenate([[[b]], [[g]], [[r]]]).T

def labels2rgb(labels, lut):
  """
  Convert a label image to an rgb image using a lookup table
  :Parameters:
    labels : an image of type np.uint8 2D array
    lut : a lookup table of shape (256, 3) and type np.uint8
  :Returns:
    colorized_labels : a colorized label image
  """
  return cv2.LUT(cv2.merge((labels, labels, labels)), lut)


if __name__ == '__main__':
  #labels = np.arange(256).astype(np.uint8)[np.newaxis, :]
  labels = np.loadtxt('./pics/result_texton_coins_texture3_20201215002820.txt',dtype=np.uint8)
  lut = gen_lut()
  print(labels.shape)
  rgb = labels2rgb(labels, lut)
  image = image.fromarray(rgb)
  print(rgb.shape)
  image.show()
