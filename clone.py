# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 19:11:39 2021

@author: vamiq
"""
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np

lines = []
with open('../Data_behavioral_cloning/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
lines = lines[1:]
images = []
measurements = []
for line in lines:
    current_path = line[0]
    #filename = source_path.split('/')[-1]
    #current_path = '../Data_behavioral_cloning/data/IMG/' + filename
    image = plt.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Flatten())
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=2)

model.save('model.h5')