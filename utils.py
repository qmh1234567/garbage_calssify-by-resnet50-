## Cutout

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self,img):
        h = img.shape[0]
        w = img.shape[1]
        c = img.shape[2]

        mask = np.ones((h,w,c),np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            # 截取函数 遮挡部分不能超过图片的一半
            y1 = np.clip(y - self.length//2,0,h)
            y2 = np.clip(y+self.length//2,0,h)
            x1 = np.clip(x-self.length//2,0,w)
            x2 = np.clip(x+self.length//2,0,w)

            mask[y1:y2,x1:x2,:] = 0.
        
        # mask = tf.convert_to_tensor(mask)
        # mask = tf.reshape(mask,img.shape)
        img = img*mask

        return img
        