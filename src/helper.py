import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt


def getModel(input_shape):
    model = Sequential()
    # Your code here
    p = 0.3

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(p))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(p))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(7, activation='softmax'))

    opt = Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def getDatasetLength(dataset):
    return dataset.cardinality().numpy()

def getData(batch_size, val_split, path='data/'):
    train = keras.utils.image_dataset_from_directory(
        directory=path+'train/',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(48, 48)
    )

    test = keras.utils.image_dataset_from_directory(
        directory=path+'test/',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(48, 48)
    )
    
    total_size = train.cardinality().numpy()
    train_size = int((1 - val_split) * total_size)
    
    train = train.shuffle(batch_size)
    train_data = train.take(train_size)
    val_data = train.skip(train_size)
    return train_data, val_data, test

def plotHistory(history, model_name='Neural Network Training History'):
    fig, ax = plt.subplots(1,2,figsize = (16,4))
    ax[0].plot(history.history['loss'],color='#EFAEA4',label = 'Training Loss')
    ax[0].plot(history.history['val_loss'],color='#B2D7D0',label = 'Validation Loss')
    ax[1].plot(history.history['accuracy'],color='#EFAEA4',label = 'Training Accuracy')
    ax[1].plot(history.history['val_accuracy'],color='#B2D7D0',label = 'Validation Accuracy')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs');
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Accuracy %');
    fig.suptitle('NN_model Training', fontsize = 24)
