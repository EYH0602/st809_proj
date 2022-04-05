import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import\
    Layer, Activation, Input, Add, BatchNormalization, Conv2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD


class IdBlock(Layer):
    def __init__(self, filter_num):
        super(IdBlock, self).__init__()
        fn1, fn2, fn3 = filter_num
        
        self.conv1 = Conv2D(fn1, (1, 1), padding='same')
        self.conv2 = Conv2D(fn2, (3, 3), padding='same')
        self.conv3 = Conv2D(fn3, (1, 1), padding='same')
    
    def call(self, inputs):
        '''forward propagation'''
        
        # will be add back to the output of the layers
        X = inputs
        X_shortcut = X
        
        # 1x1
        X = self.conv1(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        # 3x3
        X = self.conv2(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        # 1x1
        X = self.conv3(X)
        X = BatchNormalization(axis=3)(X)
        
        # add the residuals to kept X values
        h = Add()([X, X_shortcut])
        h = Activation('relu')(h)
        return h


class ConvBlock(Layer):
    def __init__(self, filter_num, s=2):
        super(ConvBlock, self).__init__()
        self.fn1, self.fn2, self.fn3 = filter_num
        
        self.conv1 = Conv2D(self.fn1, (1, 1), strides=(s, s), padding='same')
        self.conv2 = Conv2D(self.fn2, (3, 3), padding='same')
        self.conv3 = Conv2D(self.fn3, (1, 1), padding='same')
        self.conv_shortcut = Conv2D(self.fn3, (1, 1), strides=(s, s), padding='same')
    
    def call(self, inputs):
        '''forward propagation'''
        
        # will be add back to the output of the layers
        X = inputs
        X_shortcut = X
        
        # 1x1
        X = self.conv1(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        # 3x3
        X = self.conv2(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
        # 1x1
        X = self.conv3(X)
        X = BatchNormalization(axis=3)(X)
        
        # conv X shortcut if the in shape and out shape does not match
        X_shortcut = self.conv_shortcut(X_shortcut)
        X_shortcut = BatchNormalization()(X_shortcut)
        
        # add the residuals to kept X values
        h = Add()([X, X_shortcut])
        h = Activation('relu')(h)
        return h
