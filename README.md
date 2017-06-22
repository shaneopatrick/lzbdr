<img src="https://s3.amazonaws.com/lazybirder/big-alt.svg" height="150"/>

---
## Bird Classification with CNNs
<img src="http://i.imgur.com/BdO4gev.jpg" height="400"/>

###### Red-Tailed Hawk (📷 : [@CubanRalph](https://www.reddit.com/user/CubanRalph))
---

### Project Description

As a capstone project for the Galvnize Data Science Immersive I chose to create a bird classifier, given my fascination with image recongnition and an amateur interest in birds. Like most people I interact with, my free time is limited. And as a serial hobbiest I am wary of becoming a full fledged "birder". With not enough time to really invest in becoming a bird expert, I aimed to create a solution which will let me quickly identify a bird from a picture.

While Googling I found the [CUB200 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), a robust resource with approx 40-60 images for 200 classes of birds. There are a number of papers published using this dataset and I decided this would be a good data set to start with.

When I started this project I did not truly understand how difficult it would be. I quickly understood my task at hand has been appropriately dubbed __Fine-grained Object Classification__, which aims to identify categories that are both visually and semantically very similar within a general category, i.e. species of birds. "Unfortunately, it is an extremely difficult task because objects from similar subordinate categories may have marginal visual difference that is even difficult for humans to recognize. In addition, objects within the same subordinate category may present large appearance variations due to changes of scales, viewpoints, complex backgrounds and occlusions." ([source](https://arxiv.org/pdf/1606.08572.pdf))


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

Two data augmentation techniques I tried was mirroring the images (axis=1) which doubles the number of datapoints the model trains on.

__Mirrored image__

<img src="img/wp_flip.jpg" height="224"/>

The data set also includes a corresponding __segmentation__ image for every image. I experimented with masking the data  with the help of the segmentation images. This improved my results however I decided to not incorporate segmentation because in the real world my input images would not include segmentation.

Provided segmentation              |  Masked input
:-------------------------:|:-------------------------:
<img src="img/wp_segment.png" height="393"/>  | <img src="img/wp_mask_sqr.jpg" height="224"/>


After experimenting with the CUB200 dataset and acheiving mediocre results I decided to explore using more data / images.

I was lucky to find the [NABirds](http://dl.allaboutbirds.org/nabirds) dataset, which contains over 48,000 images for 404 classes of species. The NABirds dataset is also very well organized and comes with provided bounding box data.

My final model was trained using the bounding box cropping and squaring techniques described above.


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

* Test accuracy: 0.42 — baseline
* Test accuracy: 0.56 — w/ flipped images in training
* Test accuracy: 0.24 — w/ masked images in training
* Test accuracy: 0.48 — w/ masked images in training & test

[best 10 class with CNN architecture built from scratch]
* Test accuracy:  0.65 — w/ flipped images & ImageDataGenerator
* Top 5 accuracy: 0.95
* Top 3 accuracy: 0.88

In order to see where I stood overall with the entire dataset all 200 classes, the next milestone was to run the full dataset with 200 classes through my initial model.
>_trained on 8841 images, tested on 2947 images_

[baseline with scratch built CNN]
* Test accuracy:  0.13
* Top 5 accuracy: 0.56
* Top 3 accuracy: 0.33



### VGG16

At this point I pivoted to using a pretrained network, the VGG16. This drastically improved my results. Pretrained networks provide a substantial lift because the network weights have already been optimized using __ImageNet__.

The goal of the ImageNet classification challenge is to train a model that can correctly classify an input image into 1,000 separate object categories. Models are trained on approximately 1.2 million training images with another 50,000 images for validation and 100,000 images for testing.

What this means is that the pretrained VGG16 network can already detect differences in colors, shapes, and edges instead of starting from ground 0 and trying to "learn" the optimal weights.

The VGG16 trained on the CUB200 dataset yeiled the following results.

>_trained on 8841 images, tested on 2947 images_

* Test accuracy:  0.68
* Top 5 accuracy: 0.78
* Top 3 accuracy: 0.72

At this point it was time to add more data. The NABirds dataset has 60-140 images for 404 classes of birds. However due to computational limitation on AWS ec2 instances I was never able to train a model on the full 404 classes. I decided to stay with my original task of classifying 200 classes although the class labels (bird species) themselves were changed. I selected the 200 classes based on the number of images available for the class, using the classes with the most data. Utilizing the NABirds dataset my training data went from 8841 images to 23569 images. This yeiled the best results to date.


### Architecture

The [VGG16](https://arxiv.org/pdf/1409.1556.pdf) was introduced by a team at Oxford in 2014, known for its simplicity. The network consists of 13 convolutional layers with 3x3 kernels at each convolution. Size reduction is handled by max pooling layers. Two fully-connected layers are at the end of the network and are followed by a softmax classifier.


<img src="https://www.cs.toronto.edu/~frossard/post/vgg16/vgg16.png" width="600"/>

[(source)](https://www.cs.toronto.edu/~frossard/post/vgg16/)

The modifications I made to the architecture were:
* Custom input, 224x224x3 preprocessed images
* 2 fully-connected layers with 512 neurons
* Softmax classifier with 200 neurons for my number of classes.
* ReLu activation functions
* Stochastic Gradient Descent optimizer with a reducing learning rate


### Results

My final network was trained on 23569 images, validated on 4000 images, and tested on 3856 images. The amount of time needed for 20 epochs of training was approximately 60 hours. The following chart shows the training accuracy and loss for both the training and _validation_ data sets.

<img src="img/top200_vgg16-final_model_accuracy_loss.png" width="600"/>

After the model finished traing it was tested on the hold-out set of 3856 images.

>#### Test accuracy:  0.83
#### Top 5 accuracy: 0.97
#### Top 3 accuracy: 0.94

Looking at the Top 3 accuracy, out of 3856 test images the network failed to return the correct species in the top 3 predictions for only 231 images. Each time the network is given an image it will predict 200 probabilities, one for each class. You can sort those probabilities and return the top _n_ and compare it to the true label. Here's an visual of the Top 3:

<img src="img/top3_vis.png" width="400"/>


I was very pleased with this accuracy rate and decided to incorporate the Top 3 predictions in my web application.


### Web app

#### [www.lazybirder.club](http://www.lazybirder.club/)
<br>
Lazybirder is built with Flask, a microframework for running applications with in Python. The site uses a Bootstrap 4 template from www.wrapbootstrap.com.

My CNN was built using Keras and Theano and trained on AWS using a NVIDIA GRID GPU.

The image preprocessing is performed in real-time using OpenCV and Numpy. I am also hosting the application on AWS with a MongoDB.


### Next steps

* Incorporate more bird species.
* Employ "bird detection" before passing images into the network.
* Automate cropping of images with a sophisticated bounding box technique.
* Add more features to web app:
    * Allow users to confirm the system's predictions of species after submitting a photo. The user provided images could then used to train the model further to improve accuracy.
    * Incorporate geo-tagging integrated with [eBird](http://ebird.org/content/ebird/about/), allowing users to see on a map where the species has also been sighted nearby.
    * Add a social feed component to see other photographs of birds uploaded by users.
    * Mobile app.


### References

Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001.
[CUB200 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

The Birds of North America (P. Rodewald, Ed.). Ithaca: Cornell Laboratory of Ornithology; Retrieved from The Birds of North America: https://birdsna.org; AUG 2015.
[NABirds dataset](https://birdsna.org)

Zhao, B., Wu, X., Feng, J., Peng, Q. (2017). Diversified Visual Attention Networks for Fine-Grained Object Classification. arXiv.org.
[arXiv:1606.08572](https://arxiv.org/abs/1606.08572v2)

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large­scale image recognition. arXiv.org.
[arXiv:1409.1556](https://arxiv.org/abs/1409.1556)

CS231n Convolutional Neural Networks for Visual Recognition. Stanford online course.
[http://cs231n.github.io/](http://cs231n.github.io/)


---
