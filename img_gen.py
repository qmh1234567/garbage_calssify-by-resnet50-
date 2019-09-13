import os
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from glob import glob
from numpy import random
import codecs
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import Counter
from tqdm import tqdm
import numpy as np
import shutil
import matplotlib.pyplot as plt


def center_img(img,size=None,fill_value=255):
    h,w = img.shape[:2]
    if size is None:
        size = max(h,w)
    shape = (size,size) + img.shape[2:]
    background = np.full(shape,fill_value,np.uint8)
    center_x = (size-w)//2
    center_y = (size-h)//2
    background[center_y:center_y+h,center_x:center_x+w] = img
    return background

def precess_imge(img_path,img_size):
    img = Image.open(img_path)
    resize_scale = img_size / max(img.size[:2])
    img = img.resize((int(img.size[0]*resize_scale),int(img.size[1]*resize_scale)))
    img = img.convert('RGB')
    img = np.array(img)
    img = img[:,:,::-1]
    img = center_img(img,img_size)
    return img

# 加载图片和标签
def load_dataset(train_data_dir):
    label_files = glob(os.path.join(train_data_dir,'*.txt'))
    labels = []
    img_data = []
    for file_path in tqdm(label_files):
        with codecs.open(file_path,'r','utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        label = int(line_split[1])
        img_path = os.path.join(train_data_dir,line_split[0])
        # img = precess_imge(img_path,FLAGS.input_size)
        img_data.append(img_path)
        labels.append(label)
    print(sorted(Counter(labels).items(),key=lambda d:d[0],reverse=False))
    # print(Counter(labels))
    
    return img_data,labels


# 将图片归类到文件夹中
def write_list_to_dir(dstPath,img_paths,labels,is_split = False):
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    index = 0
    for label in tqdm(labels):
        label = str(label)
        savedir = os.path.join(dstPath,label)
        if not is_split:
            savedir = os.path.join(savedir,label)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        img = Image.open(img_paths[index])
        imgsavePath = os.path.join(savedir,os.path.basename(img_paths[index]))
        img.save(imgsavePath)
        index += 1      

def write_split_dataset(dstPath,img_paths,labels):
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    index = 0
    for label in tqdm(labels):
        # 写入图片
        img_name = os.path.basename(img_paths[index])[:-4]
        img = Image.open(img_paths[index])
        imgsavePath = os.path.join(dstPath,os.path.basename(img_paths[index]))
        img.save(imgsavePath)
        # 写入标签
        txt = img_name + '.txt'
        with open(os.path.join(dstPath,txt),'w') as f:
            f.write(img_name+'.jpg'+', '+str(labels[index]))
        index += 1
    



# amplify 表示扩充的倍数 class_num表示需要扩充的类别
def increase_img(img_path,class_num,save_path,amplify_ratio=1):
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
    # # 验证集合不用增强
    # val_datagen = ImageDataGenerator(rescale=1./255)
    # 原图片统计
    imgs = os.listdir(os.path.join(img_path,str(class_num)))
    print(len(imgs))
    # 清空保存文件夹下的所有文件
    shutil.rmtree(save_path)
    os.makedirs(save_path)
    # 迭代器
    train_datagenator = train_datagen.flow_from_directory(img_path,shuffle=True,
    save_to_dir=save_path,batch_size=1,target_size=(224,224),save_prefix='img',save_format='jpg')
    # 生成图片和txt
    for i in range(int(len(imgs)*amplify_ratio)):
        train_datagenator.next()
    img_names = os.listdir(save_path)
    img_names_txt_list = [x[:-4]+'.txt' for x in img_names]
    for index,txt in enumerate(img_names_txt_list):
        with open(os.path.join(save_path,txt),'w') as f:
            f.write(img_names[index]+', '+str(class_num))
    
    # train_datagenerator = train_datagen.flow_from_directory('./garbage_classify/data_set/train_dir/',target_size=(224,224),classes=classes,batch_size=16,seed=0)
    # # print(train_datagenerator.class_indices)
    # for data_batch,label_batch in train_datagenerator:
    #     for index,label in enumerate(label_batch):
    #         print(np.argmax(label))
    #         plt.subplot(1,16,index+1)
    #         plt.imshow(data_batch[index])    
    #     plt.show()
    #     exit()

    # for i in range(5):
    #     train_datagenator.next()


def increase_train_img(train_path,class_num,amplify_ratio):
    img_paths,labels = load_dataset(train_path)
    img_path = f'./garbage_classify/data_set/{class_num}/'
    save_path = './garbage_classify/increase_img/'
    increase_img(img_path,class_num,save_path,amplify_ratio)
    




if __name__ == "__main__":
    train_data_dir = './garbage_classify/train_data_save/'
    # dstPath = './garbage_classify/data_set'
    # num_classes = 40
    img_paths,labels = load_dataset(train_data_dir)
    # 不划分数据集
    # write_list_to_dir(dstPath,img_paths,labels)

    # 划分验证集和训练集
    # train_img_paths, validation_img_paths, train_labels, validation_labels = train_test_split(img_paths,labels,stratify=labels,test_size=0.25,random_state=0)
    # print('total samples: %d, training samples: %d, validation samples: %d' % (len(img_paths), len(train_img_paths), len(validation_img_paths)))
    # write_split_dataset('./garbage_classify/splitDataset/train',train_img_paths,train_labels)
    # write_split_dataset('./garbage_classify/splitDataset/val',validation_img_paths,validation_labels)
    # train_dstPath = './garbage_classify/splitDataset/train'
    # val_dstPath = './garbage_classify/splitDataset/val'
    # write_list_to_dir(train_dstPath,train_img_paths,train_labels,True)
    # write_list_to_dir(val_dstPath,validation_img_paths,validation_labels,True)


    # # 增广数据集
    # train_path = './garbage_classify/splitDataset/train'
    # class_num = 0
    # amplify_ratio = 4
    # img_paths,labels = load_dataset(train_path)
    # increase_train_img(train_path,class_num,amplify_ratio)





