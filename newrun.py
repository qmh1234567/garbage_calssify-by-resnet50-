import tensorflow as tf
import os
import sys
from keras import optimizers
from keras.optimizers import adam,SGD
from data_gen import data_flow,load_test_data
from models.resnet50 import ResNet50
from keras.layers import Flatten,Dense,Dropout,BatchNormalization,Activation,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.callbacks import TensorBoard,Callback,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
import keras.backend as K
from glob import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import img_gen
from keras import regularizers
import matplotlib.pyplot as plt
from keras.utils import np_utils
from models.model import SE_ResNet
from keras.applications.imagenet_utils import decode_predictions


# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)

# run
tf.app.flags.DEFINE_string('mode', 'train', 'optional: train, save_pb, eval')
tf.app.flags.DEFINE_string('test_data_local', '', 'the test data path on local')
tf.app.flags.DEFINE_string('data_local', '', 'the train data path on local')


tf.app.flags.DEFINE_integer('num_classes', 0, 'the num of classes which your task should classify')
tf.app.flags.DEFINE_integer('input_size', 224, 'the input image size of the model')
tf.app.flags.DEFINE_integer('batch_size', 16, '')
tf.app.flags.DEFINE_float('learning_rate',1e-4, '')
tf.app.flags.DEFINE_integer('max_epochs', 30, '')

# train
tf.app.flags.DEFINE_string('train_local', '', 'the training output results on local')
tf.app.flags.DEFINE_integer('keep_weights_file_num', 20,
                            'the max num of weights files keeps, if set -1, means infinity')


FLAGS = tf.app.flags.FLAGS


## test_acc = 0.78
def add_new_last_layer(base_model,num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5,name='dropout1')(x)
    # x = Dense(1024,activation='relu',kernel_regularizer= regularizers.l2(0.0001),name='fc1')(x)
    # x = BatchNormalization(name='bn_fc_00')(x)
    x = Dense(512,activation='relu',kernel_regularizer= regularizers.l2(0.0001),name='fc2')(x)
    x = BatchNormalization(name='bn_fc_01')(x)
    x = Dropout(0.5,name='dropout2')(x)
    x = Dense(num_classes,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=x)
    return model


def setup_to_finetune(FLAGS,model,layer_number=149):
    # K.set_learning_phase(0)
    for layer in model.layers[:layer_number]:
        layer.trainable = False
    # K.set_learning_phase(1)
    for layer in model.layers[layer_number:]:
        layer.trainable = True
    # Adam = adam(lr=FLAGS.learning_rate,clipnorm=0.001)
    Adam = adam(lr=FLAGS.learning_rate,decay=0.0005)
    model.compile(optimizer=Adam,loss='categorical_crossentropy',metrics=['accuracy'])

    

def model_fn(FLAGS):
    # K.set_learning_phase(0)
    # setup model
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    for layer in base_model.layers:
        layer.trainable = False

    # if FLAGS.mode == 'train':
        # K.set_learning_phase(1)
    model = add_new_last_layer(base_model,FLAGS.num_classes)

    # print(model.summary())
    # print(model.layers[84].name)
    # exit()

    # Adam = adam(lr=FLAGS.learning_rate,clipnorm=0.001)
    model.compile(optimizer="adam",loss = 'categorical_crossentropy',metrics=['accuracy'])
    return model



def train_model(FLAGS):
     # data flow generator
    train_sequence, validation_sequence = data_flow(FLAGS.data_local, FLAGS.batch_size,
                                                    FLAGS.num_classes, FLAGS.input_size)

    
    model = model_fn(FLAGS)

    history_tl = model.fit_generator(
        train_sequence,
        steps_per_epoch = len(train_sequence),
        epochs = FLAGS.max_epochs,
        verbose = 1,
        validation_data = validation_sequence,
        max_queue_size = 10,
        shuffle=True
    )

    # ## finetune
    setup_to_finetune(FLAGS,model)

    history_tl = model.fit_generator(
        train_sequence,
        steps_per_epoch = len(train_sequence),
        epochs = FLAGS.max_epochs*2,
        verbose = 1,
        callbacks = [
            ModelCheckpoint(f'output/best.h5',
                            monitor='val_loss', save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                            patience=10, mode='min'),
            EarlyStopping(monitor='val_loss', patience=10),
            ],
        validation_data = validation_sequence,
        max_queue_size = 10,
        shuffle=True
    )

    print('training done!')


## eval 
def test_single_h5(FLAGS,h5_weights_path):
    # model
    model = model_fn(FLAGS)

    model.load_weights(h5_weights_path,by_name=True)


    img_names,test_datas,test_labels = load_test_data(FLAGS)

    prediction_list =[]
    tta_num = 5
    for test_data in test_datas:
         ## tta
        predictions = [0*tta_num]
        for i in range(tta_num):
            x_test= test_data[i]
            x_test = x_test[np.newaxis,:,:,:]
            prediction = model.predict(x_test)[0]
            predictions += prediction
        prediction_list.append(predictions)

    right_count = 0
    error_infos = []
    
    for index,pred in enumerate(prediction_list):
        pred_label = np.argmax(pred,axis=0)
        test_label = test_labels[index]
        if pred_label == test_label:
            right_count += 1
        else:
            error_infos.append('%s,%s,%s' % (img_names[index],test_label,pred_label))

    accuracy = right_count / len(test_labels)
    print('accuracy: %s' % accuracy)
    result_file_name = os.path.join(os.path.dirname(h5_weights_path),"%s_accuracy.txt" % os.path.basename(h5_weights_path))

    with open(result_file_name,'w') as f:
        f.write('#predict error files\n')
        f.write('#####################\n')
        f.write('filename,true_label,pred_label\n')
        f.writelines(line + '\n' for line in error_infos)
        f.write("#####################\n")
        f.write('accuracy:%s\n'%accuracy)
    print(f'result save at {result_file_name}')



def main(FLAGS):
    if FLAGS.mode == 'train':
        train_model(FLAGS)
    else:
        h5_weights_path = './output/best.h5'
        test_single_h5(FLAGS,h5_weights_path)
    
if __name__ == "__main__":
    main(FLAGS)
