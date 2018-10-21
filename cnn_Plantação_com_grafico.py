'''
Autor: Anderson Alves Schinaid
Data: 12/09/2018
@Base no curso python DeepLearning de A - Z
@livro Simon Haykin Redes neurais Principios e pratica 
'''
import warnings# leia abaixo
warnings.filterwarnings("ignore")# no momento não tenho tempo para ficar tratando as merdas dos avisos então vamos simplificar essa merda
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.layers import Conv2D
from keras import applications
from keras.models import model_from_json
import numpy as np
import sys
import argparse
from keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='5'

_train_dir = 'data/imagens_tratadas/sem fundo/train'
_val_dir = 'data/imagens_tratadas/sem fundo/teste'
    
## Part 1: Building CNN
classifier = Sequential()

# 1.Convolution layer
classifier.add(Convolution2D(32,3,3, input_shape = (128,128,3), activation = 'relu', border_mode = 'same'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.5))
# 2.Convolution layer
classifier.add(Convolution2D(64,3,3,  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#classifier.add(Dropout(0.2))

# 3.Convolution layer
classifier.add(Convolution2D(64,3,3,  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#classifier.add(Dropout(0.2))

# 4.Convolution layer
#classifier.add(Convolution2D(128,3,3,  activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2,2)))
#classifier.add(Dropout(0.2))


# 5.Convolution layer
#classifier.add(Convolution2D(256,3,3,  activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2,2)))
#classifier.add(Dropout(0.2))

# 6.Convolution layer
#classifier.add(Convolution2D(512,3,3,  activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2,2)))
#classifier.add(Dropout(0.5))

# Flatten
classifier.add(Flatten())

# Full connection
    # rede completamente conectada
classifier.add(Dense(units = 2048, activation = 'relu'))
classifier.add(Dropout(rate = 0.5))
#classifier.add(Dense(units = 512, activation = 'relu'))
#classifier.add(Dropout(rate = 0.2))
classifier.add(Dense(units = 10, activation = 'softmax'))
classifier.summary()

 ## Part 2: Fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.5,
                                    zoom_range=0.5,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(_train_dir,
                                                    target_size = (128,128),
                                                    batch_size = 32,
                                                    class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(_val_dir,
                                                        target_size=(128,128),
                                                        batch_size = 32,
                                                        class_mode='categorical')




classifier.add(Dense(output_dim = len(train_generator.class_indices), activation = 'softmax'))

classifier.summary()
# Compiling CNN
classifier.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_history = classifier.fit_generator(train_generator,steps_per_epoch=400,epochs=50,
                        validation_data=validation_generator,validation_steps=20)

model_json = classifier.to_json()
with open("model/model_teste_02.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model/model_weight_teste02.h5')  
classifier.save('model/model_teste02.h5')
print("Modelo Salvo com sucesso!!")

import matplotlib.pyplot as plt
def show_train_history(train_history, train_acc, test_acc):
    
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Treinamento')
    plt.ylabel('precisão')
    plt.xlabel('Epoca')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
show_train_history(train_history, 'acc', 'val_acc')   

