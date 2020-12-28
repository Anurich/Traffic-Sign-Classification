<h2> Traffic Sign Classification. </h2>
In this work I make a small model for classification of different signs in road for example stop sign, walk sign, no entry sign etc.<br/>
This work is not based on any paper instead inspired by udacity nano degree program for self driving car. The blog post for this is available in Medium artical so you can check that out. 
<h2> Project Overview. </h2>
In this work I used LeNet architecture by <b>Yann LeCun </b> proposed in 1989, This architecutre consist of convolutional layer followed by <b> sigmoid </b> activation but in my case I used <b> tanh </b> and Subsampling layer which is nothing but average pooling followed by Linear layers.
This is very compact and easy to implement architecture, which perform really well for this project.
<img src="lenet.png"/>
<h2> Installion. </h2>
  <ul>
  <li> Python==3.6.6</li>
  <li> Pytorch==1.6.0</li>
  </ul>
<h2> Preprocessing. </h2>
If you go the <b> dataset.py </b> it contain two function for visualizing  distribution of classes and distribution of images, You can play with different normalization vs standardization techniques, In this project I use standardization approach, where i transform my image to have mean 0 and standard deviation of 1. Their is no standard way to choose which technqiues we should use, Mostly depend on the distribution of data. 
Second thing, in given dataset we are provided with bounding boxes so instead of sending the whole image to Model, I cropped the image using those bounding box convert to gray scale image and apply <b> low pass filter </b> (GaussianBlur), to remove noise and smooth images. 
Than I resize my image to 32 by 32 with channel of 1. Next thing which is really imortant if you visualize the classes from the function given in dataset.py you will realize that data is highly imabalanced, So it can easily become bais for certain classes which is not good.
So I used few techniques to deal with this situation thanks to pytorch. 
<ul>
  <li> One Number i seprate train, test and validation data using GroupShuffleSplit and SubsetRandomSampler <li>
  <li> Second I computer the weight for frequency of Labels, whose formula is given as <b>(1./count_no_label)</b> and used WeightedRandomSampler with replacement False so that we don't have repeated values, this help to balanced a data. </li>
</ul>
Above approaches may not necessary work for every case of imabalanced dataset.





