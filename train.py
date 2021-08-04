

from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import os


def load_data(datadir):
    imagelist = os.listdir(datadir)
    data_list = []
    label_list = []
    i = 0
    for image in imagelist:
        inner_path = os.path.join(datadir, image)
        img = load_img(inner_path,target_size=(224,224))
        x = img_to_array(img)
        data_list.append(x)
        label_list.append(int(image[0])-1)
        i = i+1
        print("%d loaded images"%(i))
    print("All images have been loaded")
    return np.array(data_list),np.array(label_list)

###################################################################################
###################################################################################
from keras.models import Sequential
from keras import utils as np_utils
from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, MaxPooling2D, GlobalMaxPooling2D
import keras

def quality_classify_model():
    ### net1
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3)))  # convolution layer
    model.add(Activation('relu'))  # activation layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # maxpooling layer
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # Fully connected layer
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Randomly discarded
    model.add(Dense(6))
    model.add(Activation('softmax'))

    model.summary()

    # compile the model
    # opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    opt = keras.optimizers.Adam(lr=0.001,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


    return model
###################################################################################
###################################################################################
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import utils as np_utils
from keras.preprocessing.image import ImageDataGenerator

def train():
    traindir = "./train"
    x_train, y_train = load_data(traindir)
    valdir = "./val"
    x_val, y_val = load_data(valdir)

    y_train = keras.utils.np_utils.to_categorical(y_train, 6)
    y_val = keras.utils.np_utils.to_categorical(y_val, 6)

    x_train = x_train.astype('float32')
    train_datagan = ImageDataGenerator(rescale=1. / 255, rotation_range=15, width_shift_range=0.15,
                                       height_shift_range=0.15, fill_mode='wrap')
    x_val = x_val.astype('float32') / 255

    model = quality_classify_model()
    hist = model.fit(train_datagan.flow(x_train, y_train, batch_size=64), steps_per_epoch = x_train.shape[0] // 64, epochs=10, validation_data=(x_val, y_val), shuffle=True)
    # hist = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val), shuffle=True)

    # save the model
    model.save('C:/Users/sudha/OneDrive/Desktop/anjii/work_dirs/leaves8_model11.hdf5')
    model.save_weights('C:/Users/sudha/OneDrive/Desktop/anjii/work_dirs/leaves8_model_weight11.hdf5')

    hist_dict = hist.history
    print("train acc:")
    print(hist_dict['accuracy'])
    print("validation acc:")
    print(hist_dict['val_accuracy'])

    train_acc = hist_dict['accuracy']
    val_acc = hist_dict['val_accuracy']
    train_loss = hist_dict['loss']
    val_loss = hist_dict['val_loss']

    # Drawing
    epochs = range(1, len(train_acc)+1)
    plt.plot(epochs, train_acc, 'bo', label = 'Training acc')
    plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("accuracy11.png")
    plt.figure() # Create a new graph
    plt.plot(epochs, train_loss, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("loss11.png")

train()


##############################################
#############################################
##########################################
############################################
