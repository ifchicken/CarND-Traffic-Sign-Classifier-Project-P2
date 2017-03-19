# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/train_sample.png "10_sample"
[image2]: ./report/chart_train.png "train_data_set"
[image3]: ./report/chart_valid.png "valid_data_set"
[image4]: ./report/chart_test.png "test_data_set"
[image5]: ./report/normalize.png "normalize data"
[image6]: ./report/architecture.JPG "architect"

[image7]:  ./report/more_test.png  "moretest"
[image8]:  ./more_test/00010.ppm  "moretest2"
[image9]:  ./more_test/00013.ppm  "moretest3"
[image10]: ./more_test/00020.ppm  "moretest4"
[image11]: ./more_test/00022.ppm  "moretest5"
[image12]: ./more_test/00027.ppm  "moretest6"
[image13]: ./more_test/00032.ppm  "moretest7"
[image14]: ./more_test/00037.ppm  "moretest8"
[image15]: ./more_test/00070.ppm  "moretest9"
[image16]: ./more_test/00188.ppm  "moretest10"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ifchicken/CarND-Traffic-Sign-Classifier-Project-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 3rd and 4th code cell of the IPython notebook.  

Here is an exploratory visualization of 10 data in the data set. And there are 3 bar charts showing how the data distributed between 43 classes in the train, validation and test data. Some classes have fewer data in the data set, this could make the model difficult to predict after training.

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 5th and 6th code cell of the IPython notebook.

As a first step, I decided to normalize the images becasue it's faster to get the optimized training model with nornalized data

Here is an example of a traffic sign image before and after normalize.

![alt text][image5]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the data set provided by udacity and this data set already has train, valid and test data.

Here is the code if we need to set up train, valid and test data:

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 7th cell of the ipython notebook. 

Diagram:
![alt text][image6]

My final model consisted of the following layers:

| Layer         		| Layer name	|     Description	        					| 
|:---------------------:|:--------------|:---------------------------------------------:| 
| Input         		|           	| 32x32x3 RGB image   							| 
| Convolution 5x5     	| conv1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| conv1     	| Activation									|
| Max pooling	      	| conv1     	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | conv2     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					| conv2     	| Activation									|
| Max pooling	      	| conv2     	| 2x2 stride,  outputs 5x5x16   				|
| Flatten       		| fc0       	| reshape outputs 1x400         				|
| Dropout       		| fc0       	| Probability is 50%            				|
| Fully connected		| fc1       	| outputs 1x120									|
| RELU					| fc1       	| Activation									|
| Dropout       		| fc1       	| Probability is 50%            				|
| Fully connected		| fc2       	| outputs 1x84									|
| RELU					| fc2       	| Activation									|
| Dropout       		| fc2       	| Probability is 50%            				|
| Output        		| fc3       	| outputs 1x43                  				|
| Softmax				|           	| softmax output								|
|						|           	| 												|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 8th, 9th and 10th cell of the ipython notebook. 

To train the model, I use EPOCHS = 30. At first, I use EPOCHS = 10, but I found out it needs more turn to train the data. Therefore, I increased the number to 30.

I used AdamOptimizer with rate = 0.0008. I used rate = 0.001 at beginning, but I found out the accuracy of validation data cannot improve at certain level, I guessed lower the rate a little bit could fix this issue. Therefore, I decrease rate a little and get a better accuracy of Validation data


#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.982
* validation set accuracy of 0.952
* test set accuracy of 0.934

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used LeNet model as my starting point beacause it's also a training model for the figure recognition form 0 to 9

* What were some problems with the initial architecture?
The accuracy I got accuracy for the original LeNet model is about 0.874. In order to achieve higher accuracy, I adjust the model

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
First, I tried to normalize data becasue the traing process with normalized data could be faster compare to un-processed data.
After the normalized, the accuracy of the LeNet model could reach to 0.9.

Then I looked at the accuracy of both training data and validation data, I found out the accuracy of training data is about 9% higher than validation data. The result shows overfittrd a little. Thereofre, to improve overfitted, I tried to add regulation term (dropout) in the model. the result improved (about 0.925) after I added dropout at the last 3 level(fc0, fc1 and fc2)

* Which parameters were tuned? How were they adjusted and why?
After added dropout layer in the architecture, I found out the result of accuracy cannot be further improved with EPOCHS=10 and rate=0.001. I tried to lower the rate a little bit(rate=0.008) and increase EPOCHS to 30. The accuracy results of trianing and validation both exceed 0.93  

At last, I used this model to predict test set. The accuracy is about 0.93.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

![alt text][image7] 

The last image might be difficult to classify because the amout of class 0 in the training set is fewer than other class

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 12th ~ 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Roundabout mandatory 	| Roundabout mandatory 							|
| Wild animals crossing	| Wild animals crossing							|
| No passing	   		| No passing					 				|
| Children crossing		| Children crossing      						|
| Priority road    		| Priority road   								| 
| Beware of ice/snow   	| Dangerous curve to the right 					|
| Turn left ahead		| Turn left ahead								|
| Keep right	   		| Keep right					 				|
| Speed limit (20km/h)	| Speed limit (20km/h)    						|

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This result is close to the accuracy of test set. It seems like most of time the model can predict class correctly

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the 1st image, the model is relatively sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop              							| 
| .00     				| No entry   									|
| .00					| Bicycles crossing         					|
| .00   				| No vehicles					 				|
| .00  				    | Yield             							|


The top five soft max probabilities for the 2nd image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Roundabout mandatory 							| 
| .00     				| Go straight or left   						|
| .00					| Keep right                					|
| .00   				| Ahead only					 				|
| .00  				    | Keep left             						|


The top five soft max probabilities for the 3rd image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .70         			| Wild animals crossing 						| 
| .06     				| Double curve          						|
| .05					| General caution              					|
| .03   				| Bicycles crossing				 				|
| .03  				    | Beware of ice/snow           					|


The top five soft max probabilities for the 4th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No passing            						| 
| .00     				| Vehicles over 3.5 metric tons prohibited     	|
| .00					| No passing for vehicles over 3.5 metric tons 	|
| .00   				| No vehicles   				 				|
| .00  				    | End of no passing           					|


The top five soft max probabilities for the 5th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Children crossing        						| 
| .00     				| Beware of ice/snow                        	|
| .00					| Slippery road                             	|
| .00   				| Road narrows on the right   	 				|
| .00  				    | Bicycles crossing           					|


The top five soft max probabilities for the 6th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority road         						| 
| .00     				| Road work                                 	|
| .00					| Right-of-way at the next intersection        	|
| .00   				| Speed limit (60km/h)      	 				|
| .00  				    | Beware of ice/snow           					|


For the 7th image, the model predicted it as Dangerous curve to the right. In fact, it's Beware of ice/snow.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .58         			| Dangerous curve to the right 					| 
| .20     				| Beware of ice/snow                           	|
| .10					| Slippery road                             	|
| .06   				| Children crossing          	 				|
| .03  				    | Right-of-way at the next intersection			|


The top five soft max probabilities for the 8th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Turn left ahead         						| 
| .00     				| Ahead only                                 	|
| .00					| Keep right                                	|
| .00   				| Roundabout mandatory      	 				|
| .00  				    | Go straight or right         					|


The top five soft max probabilities for the 9th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Keep right            						| 
| .00     				| Turn left ahead                              	|
| .00					| Roundabout mandatory                         	|
| .00   				| Go straight or right      	 				|
| .00  				    | Go straight or left         					|


The top five soft max probabilities for the 10th image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .86         			| Speed limit (20km/h)     						| 
| .12     				| Speed limit (30km/h)                         	|
| .02					| Speed limit (70km/h)                         	|
| .00   				| Speed limit (120km/h)      	 				|
| .00  				    | End of speed limit (80km/h)  					|


