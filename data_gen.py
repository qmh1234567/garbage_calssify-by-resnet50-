# -*- coding: utf-8 -*-
import os
import math
import codecs
import random
import numpy as np
from glob import glob
from PIL import Image

from keras.utils import np_utils, Sequence
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
from skimage import exposure
from models.resnet50 import preprocess_input
from keras.preprocessing import image
from utils import Cutout
from keras.preprocessing.image import ImageDataGenerator



class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels, batch_size, img_size,is_train):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        ## (?,41)
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_train = is_train
        if self.is_train:
            train_datagen = ImageDataGenerator(
                rotation_range = 30,  # 图片随机转动角度
                width_shift_range = 0.2, #浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
                height_shift_range = 0.2, #浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
                shear_range = 0.2, # 剪切强度（逆时针方向的剪切变换角度）
                zoom_range = 0.2,  # 随机缩放的幅度，
                horizontal_flip = True, # 随机水平翻转
                vertical_flip = True, # 随机竖直翻转
                fill_mode = 'nearest'
            )
            self.train_datagen = train_datagen

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)
    

    @staticmethod
    def center_img(img, size=None, fill_value=255):
        """
        center img in a square background
        """
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        # h,w,channel
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = Image.open(img_path)
        img = img.resize((256,256))
        img = img.convert('RGB')
        img = np.array(img)
        img = img[16:16+224,16:16+224]
        return img


    def cutout_img(self,img):
        cut_out = Cutout(n_holes=1,length=40)
        img = cut_out(img)
        return img


    def __getitem__(self, idx):

        # 图片路径
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        # 图片标签
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        # 这里是像素数组 （224，224，3）
        
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        # smooth labels
        batch_y = np.array(batch_y).astype(np.float32)*(1-0.05)+0.05/40

        # # 训练集数据增强
        if self.is_train:
            indexs = np.random.choice([0,1,2],batch_x.shape[0],replace=True,p=[0.4,0.4,0.2])
            mask_indexs = np.where(indexs==1)
            multi_indexs = np.where(indexs==2)
            
            if len(multi_indexs):
                # 数据增强
                multipy_batch_x = batch_x[multi_indexs]
                multipy_batch_y = batch_y[multi_indexs]
                
                train_datagenerator = self.train_datagen.flow(multipy_batch_x,multipy_batch_y,batch_size=self.batch_size)
                (multipy_batch_x,multipy_batch_y) = train_datagenerator.next()
                
                batch_x[multi_indexs] = multipy_batch_x
                batch_y[multi_indexs] = multipy_batch_y

            if len(mask_indexs[0]):
                # 随机遮挡
                mask_batch_x = batch_x[mask_indexs]
                mask_batch_y = batch_y[mask_indexs]
                mask_batch_x = np.array([self.cutout_img(img) for img in mask_batch_x])
                
                batch_x[mask_indexs] = mask_batch_x
                batch_y[mask_indexs] = mask_batch_y
            
        
        # 预处理
        batch_x =np.array([preprocess_input(img) for img in batch_x])

        #  # plt 绘制图像时需要将其换成整型
        # for index,label in enumerate(batch_y):
        #     print(np.argmax(label))
        #     plt.subplot(2,8,index+1)
        #     plt.imshow(batch_x[index].astype(int))
        # plt.show()
        # exit()

        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)


def data_flow(train_data_dir, batch_size, num_classes, input_size):  
    label_files = glob(os.path.join(train_data_dir, '*.txt'))
    random.shuffle(label_files)
    img_paths = []
    labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(train_data_dir, img_name))
        labels.append(label)
    
    labels = np_utils.to_categorical(labels, num_classes)
    train_img_paths, validation_img_paths, train_labels, validation_labels = \
        train_test_split(img_paths, labels, stratify=labels,test_size=0.15, random_state=0)
    print('total samples: %d, training samples: %d, validation samples: %d' % (len(img_paths), len(train_img_paths), len(validation_img_paths)))
    # 训练集随机增强图片
    train_sequence = BaseSequence(train_img_paths, train_labels, batch_size, [input_size, input_size],is_train=True)
    validation_sequence = BaseSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size],is_train=False)
    return train_sequence,validation_sequence

# def data_flow(data_dir, batch_size, num_classes, input_size,is_train=False):  
#     label_files = glob(os.path.join(data_dir, '*.txt'))  # 14802
#     random.shuffle(label_files)
#     img_paths = []
#     labels = []
#     for index, file_path in enumerate(label_files):
#         with codecs.open(file_path, 'r', 'utf-8') as f:
#             line = f.readline()
#         line_split = line.strip().split(', ')
#         if len(line_split) != 2:
#             print('%s contain error lable' % os.path.basename(file_path))
#             continue
#         img_name = line_split[0]
#         label = int(line_split[1])
#         img_paths.append(os.path.join(data_dir, img_name))
#         labels.append(label)
#     print(Counter(labels))
#     ## 图片增强
#     labels = np_utils.to_categorical(labels, num_classes)
#     # # # # #  数据流生成器
#     batch_sequence = BaseSequence(img_paths, labels, batch_size, [input_size, input_size],is_train)
#     # validation_sequence = BaseSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size])
#     return batch_sequence


########### eval #############

def preprocess_img(img_path,img_size):
    """
    image preprocessing
    you can add your special preprocess method here
    """
    img = Image.open(img_path)
    # resize_scale = img_size / max(img.size[:2])
    # img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
    img = img.resize((256,256))
    img = img.convert('RGB')
    img = np.array(img)
    imgs = []
    for _ in range(10):
        i = random.randint(0,32)
        j = random.randint(0,32)
        imgg = img[i:i+224,j:j+224]
        imgg = preprocess_input(imgg)
        imgs.append(imgg)
    return imgs


def load_test_data(FLAGS):
    label_files = glob(os.path.join(FLAGS.test_data_local,"*.txt"))
    test_data = []
    img_names = []
    test_labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path,'r','utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(',')
        img_names.append(line_split[0])
        # 处理图片
        img_path = os.path.join(FLAGS.test_data_local,line_split[0])
        img = preprocess_img(img_path,FLAGS.input_size)
        test_data.append(preprocess_img(img_path,FLAGS.input_size))
        test_labels.append(int(line_split[1]))
    print(Counter(test_labels))
    # test_data = np.array(test_data)
    return img_names,test_data,test_labels




if __name__ == '__main__':
    # train_enqueuer, validation_enqueuer, train_data_generator, validation_data_generator = data_flow(dog_cat_data_path, batch_size)
    # for i in range(10):
    #     train_data_batch = next(train_data_generator)
    # train_enqueuer.stop()
    train_data_dir = './garbage_classify/train_data/'
    batch_size = 16
    num_classes = 40
    input_size = 224
    # shape= (224,224,3)  label=(16,40)
    train_sequence, validation_sequence = data_flow(train_data_dir, batch_size,num_classes,input_size)
    batch_data, bacth_label = train_sequence.__getitem__(5)
    # print(train_sequence.shape)
    print(batch_data[0])

    # label_name = ['cat', 'dog']
    # for index, data in enumerate(batch_data):
    #     img = Image.fromarray(data[:, :, ::-1])
    #     img.save('./debug/%d_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    # train_sequence.on_epoch_end()
    # batch_data, bacth_label = train_sequence.__getitem__(5)
    # for index, data in enumerate(batch_data):
    #     img = Image.fromarray(data[:, :, ::-1])
    #     img.save('./debug/%d_2_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    # train_sequence.on_epoch_end()
    # batch_data, bacth_label = train_sequence.__getitem__(5)
    # for index, data in enumerate(batch_data):
    #     img = Image.fromarray(data[:, :, ::-1])
    #     img.save('./debug/%d_3_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    # print('end')
