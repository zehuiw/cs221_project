import random
import sys
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
selected_feature_num = 43 #31
selected_feature_index=[4, 32, 36, 11, 28, 26, 15, 27, 38, 18, 17, 33, 31, 9, 23, 20, 13, 24, 34, 21, 16,  2 ,25, 42 ,12,
  3, 37, 39  ,7 ,41 ,30 ,14, 22 , 1 ,29 , 6 ,35  ,0 , 5 ,40 ,10 ,19, 8]
#read dataset
data_path='PPI_all_score_size_distance.txt'
x_all = []
y_all = []
with open(data_path) as f:
    lines=f.readlines()
    for line in lines:
        t_info=line.strip('\n').split(',')
        if t_info[0]=='ID':
            continue
        #x_all.append([float(t) for t in t_info][1:-1])
        x_tmp = []
        for i in selected_feature_index:
        	x_tmp.append(float(t_info[i]))
        x_all.append(x_tmp)
        y_tmp = [0, 0, 0]
        if float(t_info[-1]) < 56:
        	y_tmp[0] = 1
        elif float(t_info[-1]) < 68:
        	y_tmp[1] = 1
        else:
        	y_tmp[2] = 1
        y_all.append(y_tmp)

print(x_all)
print(y_all)
x_all = np.matrix(x_all)
y_all = np.matrix(y_all)
model = Sequential()
model.add(Dense(40, activation="relu", input_dim = x_all.shape[1]))
model.add(Dense(y_all.shape[1], activation = "softmax"))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
model.fit(x_all, y_all, epochs=100000, batch_size=200, verbose=1)







