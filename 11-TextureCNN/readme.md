# Convolutional Neural Networks for Texture Image Classification
In this lab we will be exploring the use of Convolutional Neural Networks (CNN) for image classification. We will start from the very beginning, that is, we will design and train a CNN from a random weight initialization.

**Note, while executing the code does not take that long as in other Labs, you will have to discover an appropriate network by trial and error, this might be time consuming, plan ahead and don't get discouraged by the negative results.**

## 1. Resources

### 1.1. python users
Following the last lab, in this lab feel free to use any combination of layers that comes up with a decent performance. Then, you can build over your network (adding layers, non-lineatities, etc) and improve the performance. 

If you wish, you can use the traditional [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). 

If you decided to go through AlexNet, take a time to figure out the AlexNet architecture and their groundbreaking approach (at the time), or at least get a nice grasp about their contribution. 

You can easily get the entire network in the Pytorch repository: [alexnet-pytorch](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py). Note that this implementation uses a non-linearity that is different from the original paper. 

### 1.2. matlab users

The [VGG Convolutional Neural Networks Practical](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html) will be a great kick start for this Lab. We will stick to our traditions and use the [vl-feat](http://www.vlfeat.org/matlab/matlab.html) along with its extension for CNNs [MatConvNet](http://www.vlfeat.org/matconvnet/functions/). 


#### 1.2.1. Creating a CNN
Read through [Part 1](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html#part1) of the practical.
Learn about the different types of layers and what they do, specially:

- Convolution filters
- Nonlinear gating
- Pooling
- Dropout layers

Specifically, try to understand the dimensions of their inputs and outputs. How can you connect them? How does the data flows across the network?

#### 1.2.2. Training a CNN

Go through [Part 4](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html#part-4-learning-a-character-cnn) of the practical.

What is *training with Jitter*?
What are its advantages?

#### 1.2.3. Classifying images with a CNN

Go through [Part 5](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html#part-5-using-pretrained-models) of the practical.

- Visualize the model and see the different layers.
- Test it on the *peppers* image
- Test it on some of the *ImageNET* images from the PHOW Lab (Lab 08)
- Test it on a random image

## 2. Data

We will fall back to the texture dataset (because those pesky local patterns have been asking for it!!). Unlike the set you already know from Lab5, we randomly sampled *128x128* patches from each image in the set in order to create a 'new' texture dataset with 20000 images (15000 train - 2500 val - 2500 test), this sampling is a mere technical shortcut, as larger images will require a lot more time to process.

The file [TexturesDB](http://157.253.63.7/texturesCNN.zip) contains a zip file with the new dataset with their corresponding labels. Note that all the test images have '0' as label (obviously).

### 2.1. Optional

Additionally, alike the set you already know from Lab5, you are encouraged to use different image size to train/test your network. Remember the image size that got the best performance from Lab5? Well, it is time to bring it back from oblivion. For instance: 64x64, 128x128, 256x256, 227x227 for AlexNet. If you wish, you can go even further: 224x224 for [VGG](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py), 299x299 for [ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py). Take a look at the pytorch [model-zoo](https://github.com/pytorch/vision/tree/master/torchvision/models). 

**Do not waste too much time trying different sizes. Focus on the network.**

**It is important to bear in mind that the full dataset (without random sampling) is about 1000 images. If you use AlexNet or another more complex/deeper architectures, you will certainly overtfit. Moreover, you'll need to upscale the 128x128 patches to the desired size (I know it is kind of weird)** 

### 2.2. Data Augmentation

#### 2.2.1. pytorch users

During our session, I will talk you about the different data augmentation that Pytorch provides.

#### 2.2.2. matlab users
The function *getBatchWithJitter* is hardcoded for 32x32 images, its adaptation for images with different sizes is not exactly trivial. If you want, you can use my modification of that function called [getBatchWithJitter128.m](http://157.253.63.7/getBatchWithJitter128.m). But beware, while it worked ok for me, it just a quick and dirty hack over the initial function, I can't guarantee it works as expected in every possible scenario.

**(I will explain this to those users in detail)**

## 3. Recommended Resources
This [paper](https://arxiv.org/abs/1407.1610) will give you further insight on what to try while training a CNN. It is certainly not a technical tutorial, but I strongly recommend to read it before you start designing CNN architectures.

## 4. Your turn

Design a Neural Network to classify this new texture dataset. Just like in Lab 10 you are on your own. 

The one requirement is to use **only a CNN**, that is you are not allowed to apply any pre/postprocessing and other vision or learning strategies are forbidden. Additionally you must **stick to the provided data** as it would be rather easy to cheat with the already known texture dataset.

## 5. Report
The report for this lab should be brief (no more than 4 pages, -you can still use additional pages for figures and references-). It must contain the following information:

- A description of your network, and the ideas you tried to implement on that design.
- What challenges did you face while designing the architecture? How much you had to change your original design until it worked?
- Does the use of jitter helps?
- Ablation tests, we will try to explain why the network works by removing some layers from it, how does each removed layer affect the performance? What does it tell about your architecture?
- The results of your network in train and validation sets.

Do not forget to upload a MATLAB .m || PYTHON .py file containing the description of the network (similar to ``initializeCharacterCNN.m``). By running this script, it must displays the network, the number of parameters and the output size of each layer (only those layers that change the input size).

## 6. Due Date:
**April 25 2017 11:59am** As usual just upload you report to git

## 7. BONUS, The Texture Recognition Challenge 
We will be holding our small 'texture classification challenge', like most real-world challenges you are free to use **any strategy (cheating is not a valid strategy!)** to produce the better classification over the test set of our modified texture database. Unlike real world challenges, you cannot develop a joint solution with another group, any such submission will be disregarded. 

Your Submissions will have a standard format (just like in Lab 10): *.txt* extension. 

The challenge server is available [here](http://157.253.199.141:3000). As before, it will evaluate the per class accuracy, precision and recall, and this time it will rank your submission according to their average F1 measure. A leaderboard is also available on the same server. 

As this extra credit requires a lot of effort, there will be a special bonus. The best two submissions will get a +2.0 that can be added to any one of their Labs grades. 

Finally, to add some extra motivation, your lab instructor will also be part of this challenge (he will not cheat), can you beat him?

![](https://media.giphy.com/media/26BRzQS5HXcEWM7du/giphy.gif)


