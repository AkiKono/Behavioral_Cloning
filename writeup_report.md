# **Behavioral Cloning**
## Udacity Self-Driving Car Engineer Nano Degree Term 1 Project 3
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Build a deep convolution neural network in Keras that predicts steering angles from image data from cameras placed on the car
* Preprocess and augment images for training the model
* Train and validate the model with a training and validation dataset
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/original.png "Model Visualization"
[image2]: ./examples/preprocessed.png "Grayscaling"
[image3]: ./examples/loss.png "Recovery Image"
[image4]: ./examples/ELU.png "Recovery Image"
[image5]: ./examples/-1.jpg "Recovery Image"
[image6]: ./examples/0.jpg "Normal Image"
[image7]: ./examples/1.jpg "Flipped Image"
[image8]: ./examples/yaw_.png "Model Visualization"
[image9]: ./examples/posi_.png "Grayscaling"
[image10]: ./examples/yaw.png "Recovery Image"
[image11]: ./examples/position.png "Recovery Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

* **model.ipynb** containing the script to create and train the model and visualize preprocessed imagess
* **drive.py** for driving the car in autonomous mode, the speed is set to 20mph. PI speed controller is unchanged.
* **model.h5** containing a trained convolution neural network
* **writeup_report.md** summarizing the results
* **run1.mp4** video record for successful run on track 1.

#### 2. Submission includes functional code
The car can be driven autonomously in the simulator Udacity provided with the command below
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.ipynb file contains the code for loading images and training and saving the deep convolution neural network as model.h5 file. All codes were commented to explain what the code does. The first cell in the ipython notebook read in CSV file that contains image directries and steering angles data. The second cell in the ipython notebook shuffle the data and then preprocess and augment the images. The third code cell defines model architecture. The forth cell train and save the model. The fifth cell is for visualizing processed and augmented images.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The input to the model is 64x64x1 preprocessed images as shown in the image below.

**original**  
![alt text][image1]   

**preprocessed**  
![alt text][image2]  

The input values was normalized in a Keras lambda layer. The model consists of 3 convolutional layers with 5X5 filter sizes with stride of 2x2 with same padding followed by 1 convolutional layer with 3x3 filter size with stride of 1x1 with same padding. The depth of the layers were increased from 1 to 6 to 12 to 24 to 36. The output was then flattened and dropout with 20% drop rate was applied. Three fully connected layers were used to condense output to 120 to 84 to 1. Mean square error between the final output and the correct steering angle was used as a cost function. After each convolutional layers and fully connected layers, ELU activation function was applied to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

* **Training data and Validation data separated:**  
The model was trained and validated with different datasets to check if the model is overfitted or not.

* **Number of Epochs:**     
The training was stopped when the validation accuracy starts to decrease. This is the sign of overfitting. The number of Epochs used for the final model results was 3. The early stoping prevented over fitting.

* **Dropout:**  
The model contains a dropout layer with dropping rate of 20%, in order to reduce overfitting.

* **Batch Size:**   
For mini-batch SGD, weights and biases are updated using the averaged gradients from one mini-batch. Mini-batch sizes are commonly between 32-512. The effect of batch sizes on neural network using ADAM optimizer is explored in the article ["ON LARGE-BATCH TRAINING FOR DEEP LEARNING: GENERALIZATION GAP AND SHARP MINIMA"](https://arxiv.org/pdf/1609.04836.pdf). The author concluded that large batch method causes overfitting and tends to attracted to closest local minima near initial point. So the small batch size was used.

#### 3. Model parameter tuning

* **Optimizer:**  
Adaptive Moment Estimation or ADAM optimizer was used to optimize gradient descent process. An optimum learning rate and an optimum added momentum are calculated for each parameter and updated in every batch run. This adaptive optimal step-size is calculated from the ratio between averaged past gradients and square root of squared averaged past gradients, such that the optimum step-size is independent of the magnitude of gradients. Figure below is the plot of  loss with respect to Epochs.
 ![alt text][image3]  
References:  
["An overview of gradient descent optimization algorithms"](http://sebastianruder.com/optimizing-gradient-descent/index.html#fn:18)  
["ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"](https://arxiv.org/pdf/1412.6980.pdf)   
["Improving the way neural networks learn"](
http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function)

#### 4. Appropriate training data

Using the simulator provided, center lane driving for whole two laps and recovery from sides of the road were recorded and used to train the model. However, instead of using my recorded driving data, the performance was better if the training samples provided by Udacity was used. So the given data was used for the training. In the given data, there were three images from three cameras attached to the car all facing straight forward but placed left front, center front, and right front of the car. Steering angle of the car at the time of driving and taking these three images from cameras were also given. In the actual autonomous driving mode, the model uses the images from center camera only. So, for left and right camera images, steering angles were modified and pointed right and left respectively. In doing so, the model was trained and learned to recover to the center position from off the center position. However, the driving behavior of the car was not smooth. It drove off the center and suddenly made acute curve and overshoot and repeated it again and again.

To create more various recovery data, images were augmented. For details about how I created the training data by augmentation, see the next section "3. Creation of the Training Set & Training Process".

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First, I used a convolutional neural network model similar to the model use in the article ["End to End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
This model might be appropriate because a convolutional neural network works well with image processing problems. Images are locally depended data that the position of the data matters. Filters in convolutional layers extract patterns, such as edges or lines, in the image data locally. For this project, the model needs to detect lane lines and correlate the angles of these lane lines with the angles of the steering.

My model has less convolutional layers than the model in the NVIDIA article because this project is to drive the car in the simulated environment. There is no other cars on the road, no traffic signs need to be recognized, and no pedestrians to detect. So the model has less convolutional layers, and less data processing efforts while training.

Max pooling was considered, but not used in the final model. The article explains the reason ["Striving for Simplicity: The All Convolutional Net"](https://arxiv.org/abs/1412.6806). Although max pooling greatly reduce the number of neurons, (For example, 75% reduction with 2x2 max pooling with stride of 2x2), it also loses a lot of information which might be important.

The model was tested by running it through the simulator. The model drove the car off the track when there was a shadow on the road which the model misinterpreted as lane lines. To reduce the effect of the shadow, the CLAHE was applied to the image before it was fed into the model.

Finally, the vehicle was able to run the whole lap without going off the track.

#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| Preprocessed 64x64x1 Grayscale image        	|
| Convolution 5x5     	| 2x2 stride, same padding 	|
| ELU					|												|
| Convolution 5x5	    | 2x2 stride, same padding 	|
| ELU					|												|
| Convolution 5x5	    | 2x2 stride, same padding 	|
| ELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding	|
| ELU					|												|
| Flattened     		|           									|
| Dropout       		| Drop Probability: 0.2							|
| Fully connected		| Output: 120    					|
| Fully connected		| Output: 84    					|
| Fully connected		| Output: 1    					|
| MSE				|           									||

All activation function was changed from Rectified Linear Units (ReLU) to Exponential Linear Units (ELU). The plot of ELU function can be found in the figure below.

![alt text][image4]  

The idea of having negative activation is to make the mean values after activations closer to zero for all activations throughout the network so to prevent internal covariate shift, the same effect as batch normalization. What is notable here is that ELU outperformed ReLU trained with batch normalization, according to the author of ["FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)."](https://arxiv.org/pdf/1511.07289.pdf)  
Also, not having zero activation arguments, dying ReLU problem is not a concern for ELU.

#### 3. Creation of the Training Set & Training Process

The sample training image data provided by Udacity were shown below.

![alt text][image5]  
![alt text][image6]  
![alt text][image7]

Considering what are the conditions to keep the car driving on the center continuously. One is to keep the car yaw angle parallel to the lane lines. The other is to keep the position of the car in the center of the lane lines. What the model need to be trained in addition to the center driving images and steering angles data, is recovery from off center position and recovery from off center yaw angle.

To create off center yaw angle images, the top of the original image data was sheared while bottom is kept the same.
To create off center position images, the bottom of the original image data was sheared while top is kept the same.
The steering angles were also modified to fit in the various situations.  

**preprocessed**  
![alt text][image2]

**off center yaw angle**  
![alt text][image8]

**off center position**  
![alt text][image9]

The speed of the car while running autonomous mode was also important. The model only predicts steering angles from the images while the drive.py PI controller controls the speed and tries to keep the speed at the specified ideal speed. The ideal speed was set to 20 mph for my model.

The total number of original data points were 8037.  
Each data point contains three images from left, center, and right cameras.
Eight different yaw angle images were created per images.  

![alt text][image10]  

Eight different position images were also used, two of them were the original left and right camera images, and six of them were created from those left and right images. This process was repeated for flipped images.  

![alt text][image11]

The total number of dataset was then 8037x42=337512.

80% of these augmented images were used for training and 20% of them were used for validation.

The model was able to run the full track without going off the road. The same model was used to try driving the car on track 2; however, it went off road in the first curve of the track. The model needs to be improved and generalized further.

One thing to note here is that for the actual driving, model need to predict not only the steering angles of the wheel, but also the throttles and breaks. The neural network model can be designed to predict the ideal throttles and breaks to control the speed of the car, provided that the current speed and acceleration are given, or calculated from the sequences of images from the cameras using localization.
