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
import medreaders
from medreaders import ACDC

#reading in the test dataset
ACDC.set_encoder(ACDC.one_hot_encode)
ACDC.set_decoder(ACDC.one_hot_decode)
ACDC.set_images_format('Keras')
ACDC.load("./ACDC/ACDC/database/testing", "all", "both")        
ACDC.resize(160, 160)
ACDC.normalize()
#ACDC.save(images = "PatientImages", masks = "PatientMasks", both = "PatientImagesWithMasks")
images = ACDC.get_images()
masks = ACDC.get_masks()

for i in range(len(masks)):
    j = masks[i].shape[0]
    masks[i] = np.reshape(masks[i], (j,160, 160,4))
        
import tensorflow.keras.backend as K                                        

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2*And) / (K.sum(y_truef) + K.sum(y_predf)))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

from keras.metrics import MeanIoU

kernel_initializer =  'he_uniform' #Try others if you want
def Unet():
#Build the model
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
        
    return model

#splitting into batches 
test = np.zeros([60,10,160,160,1])
testL = np.zeros([60,10,160,160,4])

s = []
ms = []
for i in range(98):
    im = images[i]
    m = masks[i]
    for j in range(im.shape[0]):
        slic = im[j]
        maskSlic = m[j]
        s.append(slic)
        ms.append(maskSlic)
        
temp = list(zip(s,ms))
random.shuffle(temp)
res1, res2 = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
s, ms = list(res1), list(res2)


for i in range(60):
    for j in range(10):
        test[i,j] = s[c1]
        testL[i,j] = ms[c1]
        c1 = c1+1

baseline = Unet()
m7030 = Unet()
m8515 = Unet()
mts = Unet()

#load the weights for the models
baseline.load_weights('./baseline.h5')
m7030.load_weights('./7030ft.h5')
m8515.load_weights('./8515ft.h5')
mts.load_weights('./tsft.h5')

#GETIING SCORES FOR BASELINE
m = tf.keras.metrics.MeanIoU(num_classes=4,sparse_y_pred=False,sparse_y_true= False)
iouB = []
diceB = []
for j in range(test.shape[0]):
    print("j ",j)
    enc = baseline(test[j])
    iouS = 0
    diceS = 0
    for i in range(10):
        m.update_state(testL[j][i],enc[i])
        x = m.result().numpy()
        iouS = iouS+ x
        yt = tf.argmax(testL[j][i],axis=-1)
        yp = tf.argmax(enc[i],axis=-1)
        y = dice_coef(yt, yp).numpy()
        diceS = diceS + y
        
    iouS = iouS/10
    diceS = diceS/10
    iouB.append(iouS)
    diceB.append(diceS)

print(sum(iouB)/60)
print(np.nansum(diceB)/60)

#getting scores for 70/30 ssl
m = tf.keras.metrics.MeanIoU(num_classes=4,sparse_y_pred=False,sparse_y_true= False)
iou7 = []
dice7 = []
for j in range(test.shape[0]):
    print("j ",j)
    enc =  m7030(test[j])
    iouS = 0
    diceS = 0
    for i in range(10):
        m.update_state(testL[j][i],enc[i])
        x = m.result().numpy()
        iouS = iouS+ x
        yt = tf.argmax(testL[j][i],axis=-1)
        yp = tf.argmax(enc[i],axis=-1)
        y = dice_coef(yt, yp).numpy()
        diceS = diceS + y
        
    iouS = iouS/10
    diceS = diceS/10
    iou7.append(iouS)
    dice7.append(diceS)


print(sum(iou7)/60)
print(np.nansum(dice7)/60)

#getting scores for 85 15
m = tf.keras.metrics.MeanIoU(num_classes=4,sparse_y_pred=False,sparse_y_true= False)
iou8 = []
dice8 = []
for j in range(test.shape[0]):
    print("j ",j)
    enc =  m8515(test[j])
    iouS = 0
    diceS = 0
    for i in range(10):
        m.update_state(testL[j][i],enc[i])
        x = m.result().numpy()
        iouS = iouS+ x
        yt = tf.argmax(testL[j][i],axis=-1)
        yp = tf.argmax(enc[i],axis=-1)
        y = dice_coef(yt, yp).numpy()
        diceS = diceS + y
        
    iouS = iouS/10
    diceS = diceS/10
    iou8.append(iouS)
    dice8.append(diceS)
print(sum(iou8)/60)
print(np.nansum(dice8)/60)

#getting scores for temp sced
m = tf.keras.metrics.MeanIoU(num_classes=4,sparse_y_pred=False,sparse_y_true= False)
iouts = []
dicets = []
for j in range(test.shape[0]):
    print("j ",j)
    enc =  mts(test[j])
    iouS = 0
    diceS = 0
    for i in range(10):
        m.update_state(testL[j][i],enc[i])
        x = m.result().numpy()
        iouS = iouS+ x
        yt = tf.argmax(testL[j][i],axis=-1)
        yp = tf.argmax(enc[i],axis=-1)
        y = dice_coef(yt, yp).numpy()
        diceS = diceS + y
        
    iouS = iouS/10
    diceS = diceS/10
    iouts.append(iouS)
    dicets.append(diceS)

print(sum(iouts)/60)
print(np.nansum(dicets)/60)

a = [0,4,30,33]
f, axarr = plt.subplots(4,6, figsize=(12,12), sharex=True, sharey=True)

for i in range(4):
    axarr[i,0].imshow(test[a[i]][5],cmap='gray')
    
for i in range(4):
    axarr[i,1].imshow(tf.argmax(testL[a[i],5],axis=-1))
    
for i in range(4):
    enc = baseline(test[a[i]])
    axarr[i,2].imshow(tf.argmax(enc[5],axis=-1))
    
for i in range(4):
    enc = m7030(test[a[i]])
    axarr[i,3].imshow(tf.argmax(enc[5],axis=-1))
    
for i in range(4):
    enc = m8515(test[a[i]])
    axarr[i,4].imshow(tf.argmax(enc[5],axis=-1))
    
for i in range(4):
    enc = mts(test[a[i]])
    axarr[i,5].imshow(tf.argmax(enc[5],axis=-1))
    
# using padding
f.tight_layout()
plt.show()


