#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__: Qmh
# __file_name__: models.py
# __time__: 2019:06:27:19:51

import keras.backend as K
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, Dropout,Activation
from keras.layers import Dense,Lambda,Add,GlobalAveragePooling2D,ZeroPadding2D,Multiply
from keras.regularizers import l2
from keras import Model
from keras.layers.core import Permute
from keras import regularizers
from keras.layers import Conv1D,MaxPool1D,LSTM,ZeroPadding2D
from keras import initializers
from keras.layers import GlobalMaxPool1D,Permute,MaxPooling2D
from keras.layers import GRU,TimeDistributed,Flatten, LeakyReLU,ELU

# MODEL
WEIGHT_DECAY = 0.0001 #0.00001
REDUCTION_RATIO = 4
BLOCK_NUM = 1
# DROPOUT= 0.5
# Resblcok
def res_conv_block(x,filters,strides,name):
    filter1,filer2,filter3 = filters
    # block a
    x = Conv2D(filter1,(1,1),strides=strides,kernel_initializer='he_normal',
                kernel_regularizer = regularizers.l2(l=WEIGHT_DECAY),name=f'{name}_conva')(x)
    x = BatchNormalization(name=f'{name}_bna')(x)
    x = Activation('relu',name=f'{name}_relua')(x)
    # block b
    x = Conv2D(filer2,(3,3),padding='same',kernel_regularizer = regularizers.l2(l=WEIGHT_DECAY),name=f'{name}_convb')(x)
    x = BatchNormalization(name=f'{name}_bnb')(x)
    x = Activation('relu',name=f'{name}_relub')(x)
    # block c
    x = Conv2D(filter3,(1,1),name=f'{name}_convc',kernel_regularizer = regularizers.l2(l=WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bnc')(x)
    # shortcut
    shortcut = Conv2D(filter3,(1,1),strides=strides,name=f'{name}_shcut',kernel_regularizer = regularizers.l2(l=WEIGHT_DECAY))(x)
    shortcut = BatchNormalization(name=f'{name}_stbn')(x)
    x = Add(name=f'{name}_add')([x,shortcut])
    x = Activation('relu',name=f'{name}_relu')(x)
    return x


# ResNet
def ResNet_50(input_shape):
    x_in = Input(input_shape,name='input')

    x = Conv2D(64,(7,7),strides=(2,2),padding='same',name='conv1',kernel_regularizer = regularizers.l2(l=WEIGHT_DECAY))(x_in)
    x = BatchNormalization(name="bn1")(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3,3),strides=(2,2),padding='same',name='pool1')(x)

    x = res_conv_block(x,(64,64,256),(1,1),name='block1')
    x = res_conv_block(x,(64,64,256),(1,1),name='block2')
    x = res_conv_block(x,(64,64,256),(1,1),name='block3')

    x = res_conv_block(x,(128,128,512),(1,1),name='block4')
    x = res_conv_block(x,(128,128,512),(1,1),name='block5')
    x = res_conv_block(x,(128,128,512),(1,1),name='block6')
    x = res_conv_block(x,(128,128,512),(2,2),name='block7')

    x = Conv2D(512,(x.shape[1].value,1),name='fc6')(x)
    x = BatchNormalization(name="bn_fc6")(x)
    x = Activation('relu',name='relu_fc6')(x)
    # avgpool
    # x = GlobalAveragePooling2D(name='avgPool')(x)
    x = Lambda(lambda y: K.mean(y,axis=[1,2]),name='avgpool')(x)

    model = Model(inputs=[x_in],outputs=[x],name='ResCNN')
    # model.summary()
    return model

def squeeze_excitation(x,reduction_ratio,name):
    out_dim = int(x.shape[-1].value)
    x = GlobalAveragePooling2D(name=f'{name}_squeeze')(x)
    x = Dense(out_dim//reduction_ratio,activation='relu',name=f'{name}_ex0')(x)
    x = Dense(out_dim,activation='sigmoid',name=f'{name}_ex1')(x)
    return x

def conv_block(x,filters,kernal_size,stride,name,stage,i,padding='same'):
    x = Conv2D(filters,kernal_size,strides=stride,padding=padding,name=f'{name}_conv{stage}_{i}',
        kernel_regularizer = regularizers.l2(l=WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bn{stage}_{i}')(x)
    if stage != 'c':
        # x = ELU(name=f'{name}_relu{stage}_{i}')(x)
        x = Activation('relu',name=f'{name}_relu{stage}_{i}')(x)
    return x 


def residual_block(x,outdim,stride,name):
    input_dim = int(x.shape[-1].value)
    shortcut = Conv2D(outdim,kernel_size=(1,1),strides=stride,name=f'{name}_scut_conv',
    kernel_regularizer = regularizers.l2(l=WEIGHT_DECAY))(x)
    shortcut = BatchNormalization(name=f'{name}_scut_norm')(shortcut)

    for i in range(BLOCK_NUM):
        if i>0 :
           stride = 1
        #    x = Dropout(DROPOUT,name=f'{name}_drop{i-1}')(x)
        x = conv_block(x,outdim//4,(1,1),stride,name,'a',i,padding='valid')
        x = conv_block(x,outdim//4,(3,3),(1,1),name,'b',i,padding='same')
        x = conv_block(x,outdim,(1,1),(1,1),name,'c',i,padding='valid')
    # add SE
    x = Multiply(name=f'{name}_scale')([x,squeeze_excitation(x,REDUCTION_RATIO,name)])
    x = Add(name=f'{name}_scut')([shortcut,x])
    x = Activation('relu',name=f'{name}_relu')(x)
    return x 


# proposed model v4.0 timit libri
def SE_ResNet(input_shape):
    # first layer
    x_in =Input(input_shape,name='input')

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x_in)
    x = Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer= regularizers.l2(WEIGHT_DECAY),
                      name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu',name='relu1')(x)
    # x = ELU(name=f'relu1')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)


    x = residual_block(x,outdim=256,stride=(2,2),name='block1')
    x = residual_block(x,outdim=256,stride=(2,2),name='block2')
    x = residual_block(x,outdim=256,stride=(2,2),name='block3')
    # x = residual_block(x,outdim=256,stride=(2,2),name='block4')

    x = residual_block(x,outdim=512,stride=(2,2),name='block5')
    x = residual_block(x,outdim=512,stride=(2,2),name='block6')
    # x = residual_block(x,outdim=512,stride=(2,2),name='block7')
 

    x = Flatten(name='flatten')(x)

    x = Dropout(0.5,name='drop1')(x)
    x = Dense(512,kernel_regularizer= regularizers.l2(WEIGHT_DECAY),name='fc1')(x)
    x = BatchNormalization(name='bn_fc1')(x)
    # x = ELU(name=f'relu_fc1')(x)
    x = Activation('relu',name=f'relu_fc1')(x)


    return Model(inputs=[x_in],outputs=[x],name='SEResNet')


# vggvox1
def conv_pool(x,layerid,filters,kernal_size,conv_strides,pool_size=None,pool_strides=None,pool=None):
    x = Conv2D(filters,kernal_size,strides= conv_strides,padding='same',name=f'conv{layerid}')(x)
    x = BatchNormalization(name=f'bn{layerid}')(x)
    x = Activation('relu',name=f'relu{layerid}')(x)
    if pool == 'max':
        x = MaxPool2D(pool_size,pool_strides,name=f'mpool{layerid}')(x)
    return x

def vggvox1_cnn(input_shape):
    x_in = Input(input_shape,name='input')
    x = conv_pool(x_in,1,96,(7,7),(2,2),(3,3),(2,2),'max')
    x = conv_pool(x,2,256,(5,5),(2,2),(3,3),(2,2),'max')
    x = conv_pool(x,3,384,(3,3),(1,1))
    x = conv_pool(x,4,256,(3,3),(1,1))
    x = conv_pool(x,5,256,(3,3),(1,1),(5,3),(3,2),'max')
    # fc 6
    x = Conv2D(256,(9,1),name='fc6')(x)
    # apool6
    x = GlobalAveragePooling2D(name='avgPool')(x)
    # fc7
    x = Dense(512,name='fc7',activation='relu')(x)
    model = Model(inputs=[x_in],outputs=[x],name='vggvox1_cnn')
    return model

# def vggvox1_cnn(input_shape):
#     x_in = Input(input_shape,name='input')
#     x = conv_pool(x_in,1,96,(7,7),(1,1),(3,3),(2,2),'max')
#     x = conv_pool(x,2,256,(5,5),(1,1),(3,3),(2,2),'max')
#     x = conv_pool(x,3,384,(3,3),(1,1))
#     x = conv_pool(x,4,256,(3,3),(1,1))
#     x = conv_pool(x,5,256,(3,3),(1,1),(5,3),(2,2),'max')
#     # fc 6
#     x = Conv2D(256,(x.shape[1].value,1),name='fc6')(x)
#     # apool6
#     x = GlobalAveragePooling2D(name='avgPool')(x)
#     # fc7
#     x = Dense(512,name='fc7',activation='relu')(x)
#     model = Model(inputs=[x_in],outputs=[x],name='vggvox1_cnn')
#     return model

# deep speaker
def clipped_relu(inputs):
    return Lambda(lambda y:K.minimum(K.maximum(y,0),20))(inputs)

def identity_block(x_in,kernel_size,filters,name):
    x = Conv2D(filters,kernel_size=kernel_size,strides=(1,1),
    padding='same',kernel_regularizer=regularizers.l2(l=WEIGHT_DECAY),
    name=f'{name}_conva')(x_in)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = clipped_relu(x)
    x = Conv2D(filters,kernel_size=kernel_size,strides=(1,1),
    padding='same',kernel_regularizer = regularizers.l2(l=WEIGHT_DECAY),
    name=f'{name}_convb')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Add(name=f'{name}_add')([x,x_in])
    x = clipped_relu(x)
    return x

def Deep_speaker_model(input_shape):
    def conv_and_res_block(x_in,filters):
        x = Conv2D(filters,kernel_size=(5,5),strides=(2,2),
        padding='same',kernel_regularizer=regularizers.l2(l=WEIGHT_DECAY),
        name=f'conv_{filters}-s')(x_in)
        x = BatchNormalization(name=f'conv_{filters}-s_bn')(x)
        x = clipped_relu(x)
        for i in range(3):
            x = identity_block(x,kernel_size=(3,3),filters=filters,name=f'res{filters}_{i}')
        return x
    
    x_in = Input(input_shape,name='input')
    x = Permute((2,1,3),name='permute')(x_in)
    x = conv_and_res_block(x,64)
    x = conv_and_res_block(x,128)
    x = conv_and_res_block(x,256)
    x = conv_and_res_block(x,512)
    # average
    x = Lambda(lambda y: K.mean(y,axis=[1,2]),name='avgpool')(x)
    # affine
    x = Dense(512,name='affine')(x)
    x = Lambda(lambda y:K.l2_normalize(y,axis=1),name='ln')(x)
    model = Model(inputs=[x_in],outputs=[x],name='deepspeaker')
    return model

# proposed model
def Baseline_GRU(input_shape):
    # first layer
    x_in = Input(input_shape, name='input')
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='bn1')(x)
    x = ELU(name='relu1')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = ELU(name='relu2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    x = TimeDistributed(Flatten(),  name='timedis1')(x)
    x = GRU(512,  return_sequences=True,  name='gru1')(x)
    x = GRU(512,  return_sequences=True,  name='gru2')(x)
    x = GRU(512,  return_sequences=False,  name='gru4')(x)
  
    x = Dense(512, name='fc2', activation='relu')(x)
    x = BatchNormalization(name='fc_norm')(x)
    x = ELU(name='relu3')(x)

    return Model(inputs=[x_in], outputs=[x], name='Baseline_GRU')





if __name__ == "__main__":
    
    # model = ResNet(c.INPUT_SHPE)
    model = vggvox1_cnn((299,40,1))
    # model = Deep_speaker_model(c.INPUT_SHPE)
    # # model = SE_ResNet(c.INPUT_SHPE)
    # model = RWCNN_LSTM((59049,1))
    print(model.summary())
   