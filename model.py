############### Importing Modules ###############
import os 
import cv2
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D ,BatchNormalization
from keras.layers.core import Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import numpy as np
import pandas as pd

####### Loading images and labels #######
# Initializing global variables #
training_images = []
validation_images = []
training_labels = []
validation_labels = []

# CSV path of driving data #
data_csv  =  pd.read_csv("../data_mine/driving_log.csv")

# Load single image from the path #
def load_and_preprocess_image(img_path):
    # Load Image from the path #
    img = cv2.imread(img_path)
    # Crop the image #
    img = img[60:140,:]
    # Apply Gaussian Blur with kernel size 3 #
    img = cv2.GaussianBlur(img, (3,3), 0)
    # Resizing to (100,100) helps in reducing parameters size #
    img = cv2.resize(img,(100, 100), interpolation = cv2.INTER_AREA)
    # Convert from BGR to YUV scale #
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    return img
 
# Distort the image to create augmented data #
def distort(img, angle):
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255) #
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening #
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    # randomly shift horizon #
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)

# Load the complete data set #
def load_data():
    # Using global variables #
    global training_images
    global validation_images
    global training_labels
    global validation_labels
    
    # Local variables initialization #
    labels = []
    center_images = []
    
    # Read the 'steering' angle complete column from csv #
    steering = data_csv['steering']
    
    for index in range(0,len(steering)):
        angle = float(steering[index])
        img   = load_and_preprocess_image(data_csv['center'][index].strip())
        
        # Original Image #
        labels.append(angle)
        center_images.append(img)

        # Flipping the image vertically #
        labels.append(-1.0 * angle)
        center_images.append(cv2.flip( img, 1 ))
        
        # Distort the original image #
        d_img,d_angle = distort(img, angle)
        labels.append(d_angle)
        center_images.append(d_img)
        
        # Distort the vertically flipped image #
        d_img,d_angle = distort(cv2.flip( img, 1 ), -1.0 * angle)
        labels.append(d_angle)
        center_images.append(d_img)
    
    # Convert to numpy arrays #
    labels = np.array(labels)
    center_images = np.array(center_images)
    
    # Split the dataset into training and validation #
    training_images,validation_images,training_labels,validation_labels =             train_test_split(center_images,labels,test_size=0.1)

    print("\nLoaded the images and labels....") 
    
# Create and train the network #
def train():
    # Initializing image generators #
    train_generator = ImageDataGenerator()
    validation_generator = ImageDataGenerator()
    
    # Initialize #
    model = Sequential()

    # Normalize input shape is 100,100,3)#
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(100,100,3)))

    # Convolution layer with ELU activation #
    model.add(Conv2D(6, kernel_size=(5,5), strides=(1, 1), padding='valid'))
    model.add(ELU())
    
    # Max Pooling #
    model.add(MaxPooling2D((2, 2)))
    # Convolution layer with ELU activation #
    model.add(Conv2D(16, kernel_size=(5,5), strides=(1, 1), padding='valid'))
    model.add(ELU())
    
    # Dropout layer to reduce over fitting #
    #model.add(Dropout(0.5))
    
    # Batch Normalization #
    model.add(BatchNormalization())
    # Convolution layer with ELU activation #
    model.add(Conv2D(20, kernel_size=(5,5), strides=(1, 1), padding='valid'))
    model.add(ELU())
    
    # Batch Normalization #
    model.add(BatchNormalization())
    # Convolution layer with ELU activation #
    model.add(Conv2D(24, kernel_size=(3,3), strides=(1, 1), padding='valid'))
    model.add(ELU())
    
    # Batch Normalization #
    model.add(BatchNormalization())
    # Convolution layer with ELU activation #
    model.add(Conv2D(32, kernel_size=(3,3), strides=(1, 1), padding='valid'))
    model.add(ELU())
    
    # Batch Normalization #
    model.add(BatchNormalization())
    # Convolution layer with ELU activation #
    model.add(Conv2D(48, kernel_size=(3,3), strides=(1, 1), padding='valid'))
    model.add(ELU())
    
    # Flatten the network #
    model.add(Flatten())
    
    # Fully Connected layer with ELU activation #
    model.add(Dense(250))
    model.add(ELU())
    # Fully Connected layer with ELU activation #
    model.add(Dense(120))
    model.add(ELU())
    # Fully Connected layer with ELU activation #
    model.add(Dense(84))
    model.add(ELU())
    # Final layer having the predicted steering angle #
    model.add(Dense(1))
    
    # Adam optimizer with mean squared error loss #
    model.compile(optimizer=Adam(lr=1e-5), loss='mse',metrics=['accuracy'])
    # Generating the model #
    model.fit_generator(train_generator.flow(training_images, training_labels, batch_size=32), \
                        steps_per_epoch=len(training_images)/32 , 
                        validation_data=validation_generator.flow(validation_images, validation_labels, batch_size=32), \
                        validation_steps = len(validation_images)/32 , \
                        epochs=3, verbose=1)
    print("Trained the network....")
    
    # Save the model #
    model.save('model.h5')
    print("Saved the model....")
    
if __name__ == "__main__":
    # Load the data set #
    load_data()
    # Train and save the network #
    train()
    