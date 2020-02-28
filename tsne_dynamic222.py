#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:57:55 2019

@author: ts-siddharth.kumar
"""

import tensorflow as tf
import keras
import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import ast
#from tensorflow.layers import flatten


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.layers import merge, Input, Dense,Reshape
from keras.models import Model
import time

from keras.models import Sequential
from keras.layers import Dense
import numpy
from tensorflow.keras.callbacks import TensorBoard

from keras.models import model_from_json
from keras.models import load_model


from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from sklearn.decomposition import PCA
import matplotlib.cm as cm


def get_embeddings(model_path,data):
#    autoencoder = load_model('/Users/ts-siddharth.kumar/Downloads/model.hdf5')
    autoencoder = load_model(model_path)    
    
    intermediate_layer_model = Model(inputs=autoencoder.input, 
                                     outputs=Flatten()(autoencoder.get_layer("encoder").output))
    
    embeddings = intermediate_layer_model.predict(data)
    return embeddings
    

class Data_Set_Creator():
    
    def __init__(self):
#        PATH = os.getcwd()

        self.data_path ='/Users/ts-siddharth.kumar/Desktop/man/'
#        self.data_path = '/Users/ts-siddharth.kumar/Desktop/udacity_sim/train_data/IMG/'
        #data_path = PATH + '/Downloads/memorization3/'
        data_dir_list = os.listdir(self.data_path)
        data_dir_list.sort(key=lambda s:int(s.split('.')[0].split(' ')[1][1:-1]))
        path = []
        for i in data_dir_list:
            if 'jpg' in i:
                path.append(i)
        self.dir_list = data_dir_list
        self.img_rows = 60
        self.img_cols = 60
        self.num_channel = 3
        self.num_epoch = 10

        self.i = 0      
        self.img_data_list=[]
        self.img_data = []


    
    def set_up_data(self):
        print('Setting up Data')
        for img in self.dir_list:
            img=cv2.imread(self.data_path + img)
            input_img_resize=cv2.resize(img,(self.img_rows,self.img_cols))
            self.img_data_list.append(input_img_resize)
                
        self.img_data = np.array(self.img_data_list)
        self.img_data = self.img_data.astype('float32')
        self.img_data /= 255
#        x = shuffle(self.img_data, random_state=2)
        print('Done Setting up Data')
        return self.img_data

while True:
    obj = Data_Set_Creator()
    x = obj.set_up_data()
    
    x_data = []
    for i in range(0,len(x)):
        if (i*20)+20 < len(x) :
            x_data.append(x[i*20:(i*20)+20])
    x_data = np.array(x_data)
    x_data = x_data.reshape((7,20,60,60,3))
    
    model_path = '/Users/ts-siddharth.kumar/Downloads/model.hdf5'
    #model_path = '/Users/ts-siddharth.kumar/Desktop/model_udacity.hdf5'
    #tsne = TSNE(n_components=2, random_state=0)
    pca = PCA(n_components=2)
    embeddings = get_embeddings(model_path,x)
    X_train_pca = pca.fit_transform(embeddings)
    
    #for i in range(0,len(x)):
    #    file = open('example.txt','a') 
    #    embeddings = get_embeddings(model_path,np.expand_dims(x[i],axis=0))
    #    embed_2d = pca.transform(embeddings.reshape(1,-1))
    ##    print(str(embed_2d[0][0])+","+str(embed_2d[0][1])+","+str(embed_2d[0][2])+"\n")
    #    print(str(embed_2d[0][0])+","+str(embed_2d[0][1])+"\n")
    #    cv2.imshow('qwe',cv2.resize(x[i],(300,300)))
    #    cv2.waitKey(1)
    #    file.write(str(embed_2d[0][0])+","+str(embed_2d[0][1])+"\n") 
    #    file.close() 
    #    print('im here')
    
    
    ## 10 trajectories
    model_temp = load_model(model_path)
    reconstructions = model_temp.predict(x)
    colors = cm.rainbow(np.linspace(0, 1, 7))
    a = [i for i in range(0,7)]
    for i,c in zip(a,colors):
        embeddings = get_embeddings(model_path,x[i*20:(i*20)+20])
        embed_2d = pca.transform(embeddings)
        for j in range(0,len(embed_2d)):
            file = open('example.txt','a') 
            file.write(str(embed_2d[j][0])+"&"+str(embed_2d[j][1])+"&"+str(list(c))+"\n") 
            file.close()
            aa = np.hstack((x[i*20+j],reconstructions[i*20+j]))
            cv2.imshow('Original',cv2.resize(x[i*20+j],(300,300)))
            cv2.imshow('Reconstruction',cv2.resize(reconstructions[i*20+j],(300,300)))
            cv2.waitKey(150)
        cv2.waitKey(0)
        print('im here')
    #    cv2.waitKey(0)
    f = open('example.txt', 'r+')
    f.truncate(0)
    
cv2.destroyAllWindows()