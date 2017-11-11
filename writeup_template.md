#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model_viz]: ./examples/model_viz.PNG "Model Visualization"
[center]: ./examples/center.jpg "Center driving"
[recovery1]: ./examples/recovery1.jpg "Recovery Image"
[recovery2]: ./examples/recovery2.jpg "Recovery Image"
[recovery3]: ./examples/recovery3.jpg "Recovery Image"
[left_cam]: ./examples/left_cam.jpg "Left cam"
[track2]: ./examples/track2.jpg "Track2 Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I decided to use the model architecture by nVidia described in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
Additionally I added a cropping layer right before the normalization layer, otherwise the CNN remains unchanged.
The CNN consists of five convolutional layers with RELU activations followed by three fully connected layers without activations.
The first three convolutional layers use 5x5 kernels with stride 2 and dephts 24, 36 and 48 respectivly;
the 4th and 5th convolutional layers use 3x3 kernels with stride 1 and depths 64 and 64.

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting: the data sets include images from both track 1 and 2, driven in forward and reverse directions.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. As a loss function I used the mean squared error (MSE).
The batch size was chosen as the maximum which could comfortably fit into memory on the training machine, 32 samples in this case.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in reverse direction and data from both track 1 and 2.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a proven architecture as a basis and then fine tune it to match the specific use case.
I thought this model might be appropriate because the described use case in the paper (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
was very similar to the project goal: predicting steering angles from camera images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I used training data from both track 1 and 2 driven in forward and reverse directions. I also
added an early stopping callback to keras, which terminates training as soon as the validation loss does not decrease for two
consecutive epochs. Adding a dropout layer to the model architecture after the first 3 convolutional layers produced worse
results, so this idea was discarded.

The final step was to run the simulator to see how well the car was driving around track one. 
In my first attempt the vehicle made it all around track 1 until the last sharp right curve where it briefly crossed the
left lane line; to fix this I collected a few more training images for this curve,  focusing on keeping more to the right.
After retraining the model the vehicle finally made it all the way around the track.

####2. Final Model Architecture

For the  final model architecture I decided to use the model architecture by nVidia
described in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
Additionally I added a cropping layer right before the normalization layer, otherwise the CNN remains unchanged.
The CNN consists of five convolutional layers with RELU activations followed by three fully connected layers without activations.
The first three convolutional layers use 5x5 kernels with stride 2 and dephts 24, 36 and 48 respectivly;
the 4th and 5th convolutional layers use 3x3 kernels with stride 1 and depths 64 and 64.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][model_viz]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from a bad position.
These images show what a recovery looks like starting from the right side of the road :

![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]

To help the model to generalize more and prevent a left turning bias, I recorder two more laps driving in the opposite direction.

Then I repeated this process on track two in order to get more data points.

![alt text][track2]

To augment the data sat, I also flipped images and angles thinking that this would help to generalize more For example, here is an image that has then been flipped:

Additionally to the center images I also used the left and right camera images, combined with a 0.1 steering angle offset compared to the center image.

![alt text][left_cam]

For the final model I used only the unaltered center images.
After the collection process, I had 26984 data points. During training the images where cropped about in the middle to only contain relevant information; specifically the image of the hood at the bottom
of the center image and the background in the upper half of the image were cropped out.
Furthermore the images were normalized to lie between -1 and 1 with mean 0.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
From the training log we can see that the validation loss reached a low level at around epoch 20 and then didn't change
much from there on; the lowest overall validation loss however was recorded in epoch 98. The model was trained for 100 epochs.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
