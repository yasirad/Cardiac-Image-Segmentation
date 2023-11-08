import tensorflow as tf
import numpy as np
import random
import math
from tensorflow import keras
from numpy.linalg import norm
from tensorflow.python.keras import initializers# for data load
import os
import imageio
import matplotlib.pyplot as plt
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from keras import layers
import time
import os

# This guide can only be run with the TensorFlow backend.
os.environ["KERAS_BACKEND"] = "tensorflow"

#install med readers to read image data as np arrays
#pip install medreaders
import medreaders
from medreaders import ACDC
ACDC.set_encoder(ACDC.one_hot_encode)
ACDC.set_decoder(ACDC.one_hot_decode)
ACDC.set_images_format('Keras')
ACDC.load("./ACDC/ACDC/database/training", "all", "both") #set the path to the data 
ACDC.resize(160, 160)
ACDC.normalize() #normalize data
images = ACDC.get_images()
masks = ACDC.get_masks()

for i in range(len(masks)): #reshape the masks to be 160 160 4 instead of 160 160 1 4 
    j = masks[i].shape[0]
    masks[i] = np.reshape(masks[i], (j,160, 160,4))
        
# Augmentations for contrastive and supervised training
contrastive_augmentation = {
    "min_area": 0.25,
}
classification_augmentation = {
    "min_area": 0.75,

}

image_size = 160
image_channels = 1
width = 128
temperature = 0.1

#augments data 
def get_augmenter(min_area):
    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential(
        [
           # keras.Input(shape=(image_size, image_size, image_channels)),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            layers.RandomRotation((0.1,0.1),
    fill_mode="reflect",
    interpolation="bilinear",
    seed=None,
    fill_value=0.0,)

        ]
    )

#the unet model 
def Unet():
#Build the model
    inputs = Input((160, 160, 1))
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(4, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model
#encoder part of model
def Encoder():
    inputs = Input((160, 160, 1))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    model = Model(inputs, c5)
    return model
#projection head used to pre train encoder 
def projection_head():
    return keras.Sequential(
    [
        keras.Input(shape=(10,10,256)),
        layers.Dense(2560, activation="relu"),
        layers.Dense(2560),
        keras.layers.Flatten()
    ],
)

#Contrastive loss 

def sims(x,y,i): #gets loss for each pair 
    similar = tf.keras.losses.cosine_similarity(x[i],y[i])
    similar  = tf.math.exp(similar/temperature)
    disimilar = -1*similar
    for j in range(10):
        d =tf.keras.losses.cosine_similarity(x[i],y[j])
        de = tf.math.exp(d/(temperature))
        disimilar = disimilar+de
    
    loss = -tf.math.log(similar/disimilar)
    return loss

def simcl(x,y): #gets loss for entire batch
    globalL = 0
    for i in range(batchSize):
        globalL =globalL+ sims(x,y,i)
    globalL = (1/(batchSize)) *globalL
    return globalL

#split the data into batches  of 10
batch10 = np.zeros([188,10,160,160,1])
batch10Labels = np.zeros([188,10,160,160,4])

#put first 1188 slices into an arrry
s = []
ms = []
for i in range(198):
    im = images[i]
    m = masks[i]
    for j in range(im.shape[0]):
        slic = im[j]
        maskSlic = m[j]
        s.append(slic)
        ms.append(maskSlic)
        
temp = list(zip(s,ms))
res1, res2 = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
s, ms = list(res1), list(res2)

c1 =0 
for i in range(188):
    for j in range(10):
        batch10[i,j] = s[c1]
        batch10Labels[i,j] = ms[c1]
        c1 = c1+1
        

training,ft, trainingL, ftL = train_test_split(batch10, batch10Labels, test_size=0.3, random_state=1)



#pretrain the encoder with contrastive loss

model = Encoder()
proj = projection_head()
Unet_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
epochs = 50
batchSize = 10
contrastiveLoss = []
for epoch in range(epochs):
    print("\nStart epoch", epoch)

    for i in range(len(training)):
        loss =[]
        projection1 = get_augmenter(**contrastive_augmentation)(training[i])
        projection2 = get_augmenter(**contrastive_augmentation)(training[i])
        with tf.GradientTape() as tape:  
            enc1 = model(projection1)
            enc2 = model(projection2)
            reg_flat1 = proj(enc1)
            reg_flat2 = proj(enc2)
            l = simcl(reg_flat1, reg_flat2)
            loss.append(l)
        grads = tape.gradient(l, model.trainable_weights+proj.trainable_weights)
        Unet_optimizer.apply_gradients(zip(grads, model.trainable_weights+proj.trainable_weights))
    contrastiveLoss.append(loss)
    
#model.save_weights('./enddec/70encdec50Epoch.h5')

#apply weights of pre trained encoder to entire u net model
modelFT = Unet()
weights = modelFT.get_weights()
x = model.get_weights()
for i in range(20):
    weights[i] = x[i]
modelFT.set_weights(weights)

#fine tune the model with supervised learning 

Unet_optimizer2 = keras.optimizers.Adam(learning_rate=0.0001)
epochs = 20

for epoch in range(epochs):
    print("\nStart epoch", epoch)

    for i in range(ft.shape[0]):
        aug =get_augmenter(**classification_augmentation)(ft[i])
        with tf.GradientTape() as tape:  
            enc1 =modelFT(aug)
            l =  keras.losses.categorical_focal_crossentropy(ftL[i],enc1,gamma=3)
        if(i==10):
            m.update_state(ftL[i][5],enc1[5])
            print(m.result().numpy())
        grads = tape.gradient(l, modelFT.trainable_weights)
        Unet_optimizer2.apply_gradients(zip(grads,modelFT.trainable_weights))
        
for epoch in range(epochs):
    print("\nStart epoch", epoch)

    for i in range(ft.shape[0]):
        aug =get_augmenter(**classification_augmentation)(ft[i])
        with tf.GradientTape() as tape:  
            enc1 =modelFT(aug)
            l =  keras.losses.categorical_focal_crossentropy(ftL[i],enc1,gamma=2)
        if(i==10):
            m.update_state(ftL[i][5],enc1[5])
            print(m.result().numpy())
        grads = tape.gradient(l, modelFT.trainable_weights)
        Unet_optimizer2.apply_gradients(zip(grads,modelFT.trainable_weights))
    
epochs = 10 
for epoch in range(epochs):
    print("\nStart epoch", epoch)

    for i in range(ft.shape[0]):
        aug =get_augmenter(**classification_augmentation)(ft[i])
        with tf.GradientTape() as tape:  
            enc1 =modelFT(aug)
            l =  keras.losses.categorical_focal_crossentropy(ftL[i],enc1,gamma=3)
        if(i==10):
            m.update_state(ftL[i][5],enc1[5])
            print(m.result().numpy())
        grads = tape.gradient(l, modelFT.trainable_weights)
        Unet_optimizer2.apply_gradients(zip(grads,modelFT.trainable_weights))


modelFT.save_weights('./7030ft.h5')