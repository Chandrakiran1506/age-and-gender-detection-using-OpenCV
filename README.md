 AIM:
To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture using openCV in python on the Audience dataset.

SCOPE:
           
To predict the age, we are going to use a convolutional neural network (CNN) architecture. This CNN uses 3 convolutional layers and 2 fully connected layers with one final output layer.
We run the application via visual studio code
This problem can be considered as a classification problem instead of regression. The reason being estimating the exact age using regression is a challenging task. Even human beings cannot predict age just by looking at the face. So, we will try to predict the age in an age group like in 20 – 30 or 30-40 and so on.  It is tough to predict the age of a person from a single image as perceived age depends upon many factors.

Modules used in this project
OpenCV: As the name suggests, OpenCV is an open-source Computer Vision library. OpenCV is capable of processing real-time images and videos with analytical capabilities. It supports deep learning frameworks like TensorFlow, Pytorch, and Caffe.


MODELS, DATASETS AND FUNCTIONS USED

We are using caffe-models for age and gender detection and defining variables for age and gender values

CAFFE Model: CAFFE (Convolutional Architecture for Fast Feature Embedding) is a deep learning framework, originally developed at University of California, Berkeley. It is open source, under a BSD license. It is written in C++, with a Python interface. Caffe supports types of deep learning concepts related in the fields of image classification and image segmentation. It supports CNN and fully connected neural network designs. Caffe supports kernel libraries such as NVIDIA, CNN, and Intel MKL. In this project caffe model helps us define the internal states of the parameters of the layers.

Protocol Buffer Files: Protocol Buffers (Protobuf) is a free and open source cross-platform library. They are used for data serialization. These are TensorFlow files which are used to describe the network configuration. The protobuf files are written in xml which has .pbtxt extension. Whereas the files with .pb extension contain data in binary format which is hard to read. Google developed Protocol Buffers for internal use and provided a code generator for multiple languages under an open-source license. These Protocol Buffers were designed with an aim for simplicity and better performance. Also were aimed to be faster than XML. However, these are used at Google to store and interchange various kinds of data. Also used for many inter-machine communications.




Steps for practicing gender and age detection

 1.The contents of this zip are:

gender_net.caffemodel: It is the pre-trained model weights for gender detection.  
deploy_gender.prototxt: It is the model architecture for the gender detection model (a plain text file with a JSON-like structure containing all the neural network layer’s definitions). 
age_net.caffemodel: It is the pre-trained model weights for age detection. 
deploy_age.prototxt: is the model architecture for the age detection model         (a plain text file with a JSON-like structure containing all the neural network layer’s definitions).  
res10_300x300_ssd_iter_140000_fp16.caffemodel: The pre-trained model weights for face detection. 
deploy.prototxt.txt: This is the model architecture for the face detection model.
• opencv_face_detector.pbtxt 
• opencv_face_detector_uint8.pb

 DATASETS USED:

The database is divided in the CNN release layer (possible layer) on CNN contains 8 values for 8-year courses ("0-2", "4-6", "8--13", "15 - 20", "25– 32 "," 38-43 "," 48-55 "and" 60- "). 

Training Data: A training dataset is a set of examples used to train the model i.e. equations and parameters. Most of the methods used to train the samples tend to skip if the database is not mounted and used in a variety of ways. 

Validation Data: The validation data is also called the 'development dataset' or 'dev set' and is used to fit the hyper parameters of the classifier. You are required to have validation data as well as training and assessment data because it helps to avoid excesses. The ultimate goal is to select the network that performs best on the raw data which is why we use an independent validation database in the training dataset. 

Testing Data: Test data does not depend on training manual or validation data. If the model is suitable for both the training data and the experimental data it can be said that an excessive bias has occurred. Test data is data used only to evaluate the performance of a classifier or model. An evaluation dataset was used to look at performance characteristics such as accuracy, loss, sensitivity, etc.

 FUNCTIONS USED

OpenCV’s new deep neural network (dnn) module contains two functions that can be used for pre-processing images and preparing them for classification via pre-trained deep learning models.

cv2.dnn.blobFromImage
cv2.dnn.blobFromImages

       These two functions perform:

•	Mean subtraction
•	Scaling
•	And optionally channel swapping

Mean subtraction is used to help combat illumination changes in the input images in our dataset. We can therefore view mean subtraction as a technique used to aid our Convolutional Neural Networks.
Before we even begin training our deep neural network, we first compute the average pixel intensity across all images in the training set for each of the Red, Green, and Blue channels.


This implies that we end up with three variables:
 uR,uG , and uB  
Typically, the resulting values are a 3-tuple consisting of the mean of the Red, Green, and Blue channels, respectively.
For example, the mean values for the ImageNet training set are R=103.93, G=116.77, and B=123.68 (you may have already encountered these values before if you have used a network that was pre-trained on ImageNet).
However, in some cases the mean Red, Green, and Blue values may be computed channel-wise rather than pixel-wise, resulting in an MxN matrix. In this case the MxN matrix for each channel is then subtracted from the input image during training/testing.
Both methods are perfectly valid forms of mean subtraction; however, we tend to see the pixel-wise version used more often, especially for larger datasets.
When we are ready to pass an image through our network (whether for training or testing), we subtract the mean u  , from each input channel of the input image:
R=R-uR
 G=G-uG
 B=B-uB
 
 
 
We may also have a scaling factor,  which adds in a normalization:
R= ( R-uR )/scaling factor
G= ( G-uG )/scaling factor
B= ( B-uB )/scaling factor
 
 The value of scaling factor may be the standard deviation across the training set (thereby turning the preprocessing step into a standard score/z-score). However,  scaling factor may also be manually set (versus calculated) to scale the input image space into a particular range — it really depends on the architecture, how the network was trained, and the techniques the implementing author is familiar with.
It’s important to note that not all deep learning architectures perform mean subtraction and scaling! Before you pre-process your images, be sure to read the relevant publication/documentation for the deep neural network you are using.
As you’ll find on your deep learning journey, some architectures perform mean subtraction only (thereby setting scaling factor = 1 ). Other architectures perform both mean subtraction and scaling. Even other architectures choose to perform no mean subtraction or scaling. Always check the relevant publication you are implementing/using to verify the techniques the author is using.


We load the network by using cv2.dnn.readNet to load the pre-trained models of age, gender and face. Use the webcam of your system by using cv2.VideoCapture(0)

          blobFromImage function:

blobFromImage creates 4-dimensional blob from image. Optionally resizes and crops image from centre, subtract mean values, scales values by scalefactor, swap Blue and Red channels.
Informally, a blob is just a (potentially collection) of image(s) with the same spatial dimensions (i.e., width and height), same depth (number of channels), that have all be pre-processed in the same manner.

•	The cv2.dnn.blobFromImage and cv2.dnn.blobFromImages functions are near identical.

Let us start with examining the cv2.dnn.blobFromImage function signature below:
•	blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)

I have provided a discussion of each parameter below:

Image :

           This is the input image we want to pre-process before passing it through our deep neural network for classification.

Scalefactor:

               After we perform mean subtraction, we can optionally scale our images by some factor. This value defaults to `1.0` (i.e., no scaling) but we can supply another value as well. It is also important to note that scalefactor
should be 1/scale factor  as we’re actually multiplying the input channels (after mean subtraction) by scalefactor.
          
Size:
Here we supply the spatial size that the Convolutional Neural Network expects. For most current state-of-the-art neural networks this is either 224×224, 227×227, or 299×299.

Mean:
         These are our mean subtraction values. They can be a 3-tuple of the RGB means or they can be a single value in which case the supplied value is subtracted from every channel of the image. If you’re performing mean subtraction, ensure you supply the 3-tuple in `(R, G, B)` order, especially when utilizing the default behaviour of swapRB=True.
 
 swapRB :

OpenCV assumes images are in BGR channel order; however, the `mean` value assumes we are using RGB order. To resolve this discrepancy, we can swap the R and B channels in image by setting this value to `True`. By default, OpenCV performs this channel swapping for us.
 
•	The cv2.dnn.blobFromImage function returns a blob which is our input image after mean subtraction, normalizing, and channel swapping.
•	The cv2.dnn.blobFromImages function is exactly the same:

blob = cv2.dnn.blobFromImages(images, scalefactor=1.0, size, mean, swapRB=True)

The only exception is that we can pass in multiple images, enabling us to batch process a set of images.
If you are processing multiple images/frames, be sure to use the    cv2.dnn.blobFromImages function as there is less function call overhead and you’ll be able to batch process the images/frames faster.

getFaceBox function:
          We will create a function called getFaceBox for pre-processing the image,
create a blob of the image(so that it can concentrate on more important features) and detect the faces using pre-trained models such as opencv_face_detector_uint8.pb and if we detect a face (which we can only be detected if it has more probability than conf_threshold) we draw a box around it using cv2.rectangle() function and yes it will detect multiple faces.
