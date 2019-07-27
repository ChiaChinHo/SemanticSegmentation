import os
import numpy as np
from keras import backend
from keras.models import Model
from keras.layers import *
#from keras.utils.vis_utils import plot_model

backend.set_image_dim_ordering('tf')

def vgg16_cnn(img_input, path=None):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block3_pool')(x)
    b3 = x
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block4_pool')(x)
    b4 = x
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name = 'block5_pool')(x)
    b5 = x

    vgg = Model(img_input, x, name='vgg16')
    if path:
        vgg16_model_path = os.path.join(path, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        vgg.load_weights(vgg16_model_path, by_name=True)
    for layer in vgg.layers:
        layer.trainable = False

    return b3, b4, b5

def crop(o, o2, img_input):
    o_shape = Model(img_input, o).output_shape    
    o_height, o_width = o_shape[1], o_shape[2]

    o2_shape = Model(img_input, o2).output_shape    
    o2_height, o2_width = o2_shape[1], o2_shape[2]
    
    cx = abs(o_width - o2_width)
    cy = abs(o_height - o2_height)

    if o_width > o2_width:
        o = Cropping2D(((0, int(cx/2))))(o)
    else:
        o2 = Cropping2D(((0, int(cx/2))))(o2)

    if o_height > o2_height:
        o = Cropping2D((int(cy/2), 0))(o)
    else:
        o2 = Cropping2D((int(cy/2), 0))(o2)

    return o, o2


def FCN32s(n_class = 7, path=None, width=512, height=512):
    img_input = Input(shape = (width, height, 3))

    _, _, b5 = vgg16_cnn(img_input, path)
    o = b5
    
    o = Conv2D(4096, (7, 7), activation='relu', padding='same', name='conv6')(o)
    o = Dropout(0.5)(o)
    o = Conv2D(4096, (1, 1), activation='relu', padding='same', name='conv7')(o)
    o = Dropout(0.5)(o)
    o = Conv2D(n_class, (1, 1), name='conv8')(o)
    
    o = Conv2DTranspose(n_class, kernel_size=(64, 64), strides=(32, 32), padding='same')(o)

    o = Activation('softmax')(o)
    
    fcn32s= Model(img_input, o)
    #plot_model(fcn32s, to_file='FCN32s.png')
    return fcn32s


if __name__ == '__main__':
    path = '../model/'
    model = FCN32s(path=path)
    #model = Segnet(path=path)
    #model = VGGSegnet(path=path)
    #opt = Adam()
    #model.compile(loss='categorical_crossentropy', optimizer=opt)
    print(model.summary())

    del model
