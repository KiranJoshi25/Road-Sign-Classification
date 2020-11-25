import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
img = image.load_img("C:/Users/Kiran/Desktop/Base data/Training/Signal_ahead/1.png")
plt.imshow(img)
cv2.imread("C:/Users/Kiran/Desktop/Base data/Training/Signal_ahead/1.png").shape
train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)
train_dataset = train.flow_from_directory('C:/Users/Kiran/Desktop/Base data/Training',
                                          target_size = (300,300),
                                          batch_size = 3,
                                          class_mode ='binary')
Validation_dataset = train.flow_from_directory('C:/Users/Kiran/Desktop/Base data/Validation',
                                          target_size = (300,300),
                                           batch_size = 3,
                                           class_mode ='binary')
train_dataset.class_indices
train_dataset.classes
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    
                                    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    
                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(512,activation='relu'),

                                    tf.keras.layers.Dense(1,activation='sigmoid')
                                    
                                    ])
from tensorflow.keras import optimizers
model.compile(
              loss='binary_crossentropy', 
              
              optimizer = optimizers.RMSprop(lr=0.001),
              
              metrics = ['accuracy'])
history = model.fit(train_dataset,steps_per_epoch=6,epochs=30,
                      validation_data = Validation_dataset)
Validation_dataset.class_indices
dir_path = 'C:/Users/Kiran/Desktop/Base data/testing'
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+i,target_size=(300,300))
    plt.imshow(img)
    plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    print(val)
    if (val == 0):
        print("Signal_ahead")
    else:
        print("Stop")

