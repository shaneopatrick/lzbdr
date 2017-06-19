# LazyBirder
---
## Bird Classification with CNNs
<img src="http://i.imgur.com/BdO4gev.jpg" height="400"/>

###### Red-Tailed Hawk (📷 : [@CubanRalph](https://www.reddit.com/user/CubanRalph))
---

### Project Description

As a capstone project for the DSI I chose to create a bird classifier, given my fascination with image recongnition and an amateur interest in birds. Like most people I interact with, my free time is limited. And as a serial hobbiest I am wary of becoming a full fledged "birder". With not enough time to really invest in becoming a bird expert, I aim to create a solution which will let me quickly identify a bird from a picture.

While Googling I found the [CUB200 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), a robust resource with approx 40-60 images for 200 classes of birds. [continued...]

### Approach

Given the dataset consists of 200 classes my initial ambitous goal is to create a final solution which can accurately predict any of the 200 classes given an unseen image. I quickly realized with this number of classes a high accuracy rate would be difficult.

I decided to start small, and build from there.

Starting with only two classes of birds, I was able to get a simple CNN up and running on my local machine using Keras and Theano. 100% test accuracy was acheived in no time using two very distinct species:

<img src="img/Red_Headed_Woodpecker.jpg" height="300"/> <img src="img/Painted_Bunting.jpg" height="300"/>
> _trained on 90 images, tested on 30 images_


Next to make it more difficult, I choose two similar species... 87.99% test accuracy:

<img src="img/Black_Throated_Sparrow.jpg" width="300"/> <img src="img/Harris_Sparrow.jpg" width="300"/>
> _trained on 90 images, tested on 30 images_


I then decided to establish a benchmark for 10 classes, and try a number of different methods to optimize the CNN and improve performance.
> _trained on 598 images, tested on 299 images_

* Test accuracy: 0.4247 — baseline
* Test accuracy: 0.5752 — w/ flipped images in training
* Test accuracy: 0.2348 — w/ masked images in training
* Test accuracy: 0.4782 — w/ masked images in training & test

[best 10 class model thus far]
* Test accuracy:  0.6522 — w/ flipped images & ImageDataGenerator
* Top 5 accuracy: 0.9531
* Top 3 accuracy: 0.8829

In order to see where I stood overall with the entire dataset all 200 classes, the next milestone was to run the full dataset with 200 classes through my initial model.
>_trained on 8841 images, tested on 2947 images_

[best thus far...]
* Test accuracy:  0.1286
* Top 5 accuracy: 0.5567
* Top 3 accuracy: 0.3329


Tinkered with ResNet50 however I believe there is some serious overfitting happening as my dataset largely comes from ImageNet. The training accuracy is very high while the testing accuracy very low.
> _Warning: Images in this dataset overlap with images in ImageNet. Exercise caution when using networks pretrained with ImageNet (or any network pretrained with images from Flickr) as the test set of CUB may overlap with the training set of the original network. — CUB200 disclaimer_

```
Epoch 1/3
598/598 [==============================] - 200s - loss: 0.9545 - acc: 0.7191 - val_loss: 2.9153 - val_acc: 0.0870
Epoch 2/3
598/598 [==============================] - 200s - loss: 0.1653 - acc: 0.9498 - val_loss: 3.1140 - val_acc: 0.0870
Epoch 3/3
598/598 [==============================] - 200s - loss: 0.1111 - acc: 0.9649 - val_loss: 2.6080 - val_acc: 0.1037
```




#### Still need to try...
* Incorporate meta data, see how much improvement can be made
* Advanced architectures
* More image data!!


### Data & preprocessing

The [CUB200 Birds Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) is made available via the Caltech Vision team. I was fortunate in that this dataset is clean, organized, and comes with some goodies.

There was a little bit of preprocessing to be done and I immediately dove in building things from scratch. The dataset provided __bounding box__ data for every image to take advantage of. I used this to crop the images:

Original image                     |  Cropped image
:-------------------------:|:-------------------------:
<img src="img/Red_Headed_Woodpecker_0071_183132.jpg" width="500" height="393"/>  | <img src="img/wp_cropped.jpg" height="179"/>


In order to ensure every cropped image was fed into the CNN with the same dimension the cropped image needed to be resized. However inorder to maintain the original aspect ratio margins were added to the top & bottom, or left & right, the scaled to size:

Cropped image              |  Squared & resized image
:-------------------------:|:-------------------------:
<img src="img/wp_cropped.jpg" height="179"/>  | <img src="img/wp_sqr_resize.jpg" height="224"/>

Two data augmentation techniques I tried was flipping the images (axis=1) which double the number of datapoints the model trains on.

__Flipped image__

<img src="img/wp_flip.jpg" height="224"/>

The data set also includes a corresponding __segmentation__ image for every image. I experimented with masking the data  with the help of the segmentation images.

Provided segmentation              |  Masked input
:-------------------------:|:-------------------------:
<img src="img/wp_segment.png" height="393"/>  | <img src="img/wp_mask_sqr.jpg" height="224"/>









### Best solution
...


### Results

....
