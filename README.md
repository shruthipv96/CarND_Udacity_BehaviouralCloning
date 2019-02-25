# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[myNet]:         ./document_images/myNet.png     "myNet"
[center1]:       ./document_images/center1.jpg   "center1"
[center2]:       ./document_images/center2.jpg   "center2"
[center3]:       ./document_images/center3.jpg   "center3"
[original]:      ./document_images/original.png  "original"
[blur]:          ./document_images/blur.png      "blur"
[resize]:        ./document_images/resize.png    "resize"
[yuv]:           ./document_images/yuv.png       "yuv"
[flip]:          ./document_images/flip.png      "flip"
[crop]:          ./document_images/crop.png      "crop"
[distort]:       ./document_images/distort.png   "distort"
[distort_flip]:  ./document_images/distort_flip.png     "distort_flip"
[distort_step]:  ./document_images/distort_step.png     "distort_step"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The complete solution is available in :
**/home/workspace/CarND-Behavioral-Cloning-P3/Solution/**

The project includes the following files:
* **model.py** containing the script to create and train the model. 
* **drive.py** for driving the car in autonomous mode.
* **model.h5** containing a trained convolution neural network.
* **writeup_report.md** summarizing the results.
* **video.mp4** showing the output video of the car driven in 9 mph.
* extra_videos folder containing additional outputs.
* extra_model folder containing additional models.

#### 2. Submission includes functional code
Using the Udacity provided simulator and **drive.py** script and **model.h5** model file, the car can be driven autonomously around the track by executing 
```sh
# Make sure to be in /home/workspace/CarND-Behavioral-Cloning-P3/Solution/ folder 
# or use to go to the location
# cd /home/workspace/CarND-Behavioral-Cloning-P3/Solution/ 
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The **model.py** file contains the code for loading the dataset, augmenting the dataset, training and saving the neural network. The code is neatly organized and the steps are properly commented for explanation.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is an upgradation to the model I used in **Traffic Sign Classifier project**(which is an upgradation of LeNet architecture) and it also proved to be working.

* My model consists of normalization layer to help the network converge faster. (model.py : line-128)
* My model consists of a convolution neural network with 5x5 filter sizes and depths of 6,16,20 (model.py : lines 131-147) 
* My model consists of a convolution neural network with 3x3 filter sizes and depths of 24,32,48 (model.py : lines 150-165) 
* The model contains ELU layers to introduce nonlinearity (model.py : lines 132,138,147,153,159,165,175,178)
* The model contains BatchNormalization to further enhance the network converge faster. (model.py : lines 144,150,156,162)
* The model contains Fully Connected layers to drive a single value. (model.py : lines 171,174,177,180)

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The following shows the output of the model trained.

* 942/941 [==============================] - 72s - loss: 0.0478 - acc: 0.4024 - val_loss: 0.0121 - val_acc: 0.4217
* Epoch 2/3
* 942/941 [==============================] - 70s - loss: 0.0078 - acc: 0.4163 - val_loss: 0.0076 - val_acc: 0.4191
* Epoch 3/3
* 942/941 [==============================] - 70s - loss: 0.0049 - acc: 0.4162 - val_loss: 0.0060 - val_acc: 0.4268

You can see that the training loss is less than the validation loss by 0.001 in the last epoch (Looks like overfitting. I beleive it is because of less validation data). But the car was able to drive and stay on the track. 

Still, I did an attempt to add Dropout layer (model.py : line 141) and I observed that the car was able to drive. But at times the car almost touched the lane line. Video of this is available in *extra_videos/video_dropout.mp4*. So I commented the Dropout layer.


#### 3. Model parameter tuning

The model used an Adam optimizer with a learning rate of 1e-5 (model.py : line 183).

#### 4. Appropriate training data

The neural network is like a clay which we can shape. So I started teaching the network by driving the car myself.

The following points give details on how I taught the network.

* I drove the car almost along the center of the lane for around 2 laps. Then I made the network drive the car. It was not able to maintain the center lane. 
* So,I drove the car in the opposite direction for around 1 lap. Then I made the network drive the car. It was able to maintain the center lane most of the time but was too much of wobbling.
* So,I drove the car along the center of the lane and stopped collecting data at turns. Then I made the network drive the car. It was able to maintain the center lane with less wobbling.
* So,I trained the network to come back to center of the lane when the car is nearing the lane lines . Then I made the network drive the car. It was able to maintain the center lane.
* In all the above cases, the car was able to drive autonomously upto the curve next to bridge successfully. But it failed to clear the curve.
* So,I drove the car along the curve after the bridge for 2 times. Then I made the network drive the car. It was able to clear the curve.
* Once it crossed the curve after the bridge, the car was not able to clear the last curve properly. So,I drove the car along the last curve for 1 time. Then I made the network drive the car. It was able to clear the curve.

For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first approach was to use the same network used in Traffic Sign Classifier project by adding a final layer to output 1 value.It did not seem to work fine. This is obvious because the input size of this network was (32,32) and here reducing the image from (160,320) to (32,32) will have a large amount of data loss.

Hence, I was looking into some other neural network for solving the problem. I found nvidia's neural network for self driving car. I read the paper and used the same preprocessing steps and modified my neural network similar to nVidia neural network. One difference is nVidia neural network has an input shape of (66,200) whereas my modified network has an input shape of (100,100)

I added a _normalization layer_, added _batch normaliaztion_, added _convolutional layers of (5x5) and (3x3) filters_.I increased the depth of the network which helps to understand more features. More details about the model is explained above.

I collected the training data and made the network to drive the car. It failed. So I collected more data in the way I explained before and trained the model.

The final step was to run the simulator to see how well the car was driving around track one and I found the vehicle is able to drive autonomously around the track without leaving the road. The [video](./video.mp4) is in Solution folder named video.mp4.

#### 2. Final Model Architecture

The final model architecture is available in the code under `train()` function. (model.py : lines 119-194) My neural network consisted of the following layers and layer sizes:

| Layer (type)          |     Output Shape                |        Param     |
|:---------------------:|:-------------------------------:|:----------------:|
|lambda_1 (Lambda)      |      (None, 100, 100, 3)        |    0     |        
|conv2d_1 (Conv2D)      |      (None, 96, 96, 6)          |    456   |    
|elu_1 (ELU)            |      (None, 96, 96, 6)          |    0     |    
|max_pooling2d_1        |(MaxPooling2 (None, 48, 48, 6)   |    0     |        
|conv2d_2 (Conv2D)      |     (None, 44, 44, 16)          |    2416  |    
|elu_2 (ELU)            |      (None, 44, 44, 16)         |    0     |    
|batch_normalization_1  | (Batch (None, 44, 44, 16)       |    64    |    
|conv2d_3 (Conv2D)      |      (None, 40, 40, 20)         |    8020  |    
|elu_3 (ELU)            |      (None, 40, 40, 20)         |    0     |       
|batch_normalization_2  | (Batch (None, 40, 40, 20)       |    80    |    
|conv2d_4 (Conv2D)      |      (None, 38, 38, 24)         |    4344  |    
|elu_4 (ELU)            |      (None, 38, 38, 24)         |    0     |    
|batch_normalization_3  | (Batch (None, 38, 38, 24)       |    96    |    
|conv2d_5 (Conv2D)      |      (None, 36, 36, 32)         |    6944  |    
|elu_5 (ELU)            |      (None, 36, 36, 32)         |    0     |    
|batch_normalization_4  | (Batch (None, 36, 36, 32)       |    128   |    
|conv2d_6 (Conv2D)      |      (None, 34, 34, 48)         |    13872 |    
|elu_6 (ELU)            |      (None, 34, 34, 48)         |    0     |    
|flatten_1 (Flatten)    |      (None, 55488)              |    0     |    
|dense_1 (Dense)        |      (None, 250)                |    13872250 |
|elu_7 (ELU)            |      (None, 250)                |    0      |    
|dense_2 (Dense)        |      (None, 120)                |    30120  |   
|elu_8 (ELU)            |      (None, 120)                |    0      |   
|dense_3 (Dense)        |      (None, 84)                 |    10164  |   
|elu_9 (ELU)            |      (None, 84)                 |    0      |   
|dense_4 (Dense)        |      (None, 1)                  |    85     |   

Total params: 13,949,039

Trainable params: 13,948,855

Non-trainable params: 184
________________________________________________________________

Here is a visualization of the architecture.(I have avoided few layers like ELU, batch normalization so that the complete architecture can be captured)

![alt text][myNet]

#### 3. Creation of the Training Set & Training Process

A dataset is already provided by Udacity. First I tried that data to create the model. The model was lagging to drive the car. 

So I collected data by driving myself as explained above. The data is available in the [data_mine]() folder.It has IMG folder having images from center,left and right cameras and a csv file having the information about the images.

First,I loaded the driving csv file and extracted the steering angle from the csv.I am loading only center image corresponding to the steering angle.This code is available under `load_data()` function. (model.py : lines 73-116)
Few of the images from the dataset looks like below:

![alt text][center1]![alt text][center2]![alt text][center3]

The generation of dataset has two parts:

1) **Preprocessing** : As mentioned earlier, I am using the same preprocessing steps as that of nvidia self driving car model. This code is available under `load_and_preprocess_image()` fucntion. (model.py : lines 30-40)

The following are the steps:
* Crop the image to remove extra area.
![alt text][crop]

* Apply Gaussian Blur of kernel size 3.
![alt text][blur]

* Resize the image to (100,100).
![alt text][resize]

* Convert the image to YUV scale from BGR scale.
![alt text][yuv]

2) **Augmentation** : Only the data recorded is not enough for training the model. It has mostly counter clockwise steering angle images. This biases the model. Hence, I did some augmentation on the preprocessed images.

* **Original image** - The output of the preprocessed image is used as such.
![alt text][original]

* **Flipped image** -  The preprocessed image is flipped vertically and the angle is multiplied by -1.
```python
    labels.append(-1.0 * angle)
    center_images.append(cv2.flip( img, 1 )) # Flip Vertically
```
![alt text][flip]

* **Distortion image** - The image is distorted by adding random brightness(model.py : lines 47-53),random darkness (model.py : lines 54-61),random horizontal shifting(model.py : lines 62-69). The code for this is available under `distort()` function. (model.py : lines 45-70) and the angle is retained the same. The following gives an example of distortion at each step.
![alt text][distort_step]

The following is an example of distortion applied on a preprocessed image.

![alt text][distort]

* **Distortion flipped image** - The vertically flipped preprocessed image is distorted using the _same_ `distort()` function and the angle is multiplied by -1.
![alt text][distort_flip]

After preprocessing and augmenting the data, I split them into `training(90%)` and `validation(10%)` sets using
```python    
training_images,validation_images,training_labels,validation_labels =             train_test_split(center_images,labels,test_size=0.1)
```
Once the data is loaded, the datasets look like:

Complete Dataset:

Images:  (66948, 100, 100, 3)

Labels:  (66948,)

Training Dataset:

Images:  (60253, 100, 100, 3)

Labels:  (60253,)

Validation Dataset:

Images:  (6695, 100, 100, 3)

Labels:  (6695,)

I am using `Adam` optimiser and `mean square error` loss for training the network.
```python
model.compile(optimizer=Adam(lr=1e-5), loss='mse',metrics=['accuracy'])
```
I am feeding the training and validation dataset to the network using [ImageDataGenerator](https://keras.io/preprocessing/image/) class available in Keras. This generator takes care of creating batches for training.
```python
model.fit_generator(train_generator.flow(training_images, training_labels, batch_size=32), \
           steps_per_epoch=len(training_images)/32 , \
           validation_data=validation_generator.flow(validation_images, validation_labels,batch_size=32),\
           validation_steps = len(validation_images)/32 ,epochs=3, verbose=1)
```

By doing this, I obtained a model. I have changed the `drive.py`(lines 65-73) to do the preprocessing before prediction and it was able to drive the car autonomously on track.