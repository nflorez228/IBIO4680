# Lab 05 - Textons 

In this lab you will use a strategy to represent images using Textons. Then you will train, and evaluate a classifier based on the texton representation. 

Try to develop quality code so that you can reuse it in the following labs.

**The code for this lab takes a VERY LONG time to execute, plan ahead a use wisely the course servers, you are not the only one there!!!**

## Database

The database for this lab is provided by the [ponce group](http://www-cvr.ai.uiuc.edu/ponce_grp/data/). You can find it in the link as **Texture Database**.

Create a folder called *data* and locate your dataset there. 
    
## Image Representation

The *lib* folder contains functions that can be used to represent images as textons.

Pay special attention to the following functions (you will see them again in the example), try to figure out what they do, what their inputs and outputs are.

    -   fbCreate
    -   fbRun
    -   computeTextons
    -   assignTextons

The following script will give you some ideas on how to create a texton dictionary from 2 sample images, then use it  to compare another 2 images:

**Matlab**

```Matlab
addpath('lib/matlab')

%%Create a filter bank with deafult params
[fb] = fbCreate;

%%Load sample images from disk
imBase1=double(rgb2gray(imread('img/person1.bmp')))/255;
imBase2=double(rgb2gray(imread('img/goat1.bmp')))/255;

%Set number of clusters
k = 16*8;

%Apply filterbank to sample image
filterResponses=fbRun(fb,horzcat(imBase1,imBase2))

%Computer textons from filter
[map,textons] = computeTextons(filterResponses,k);

%Load more images
imTest1=double(rgb2gray(imread('img/person2.bmp')))/255;
imTest2=double(rgb2gray(imread('img/goat2.bmp')))/255;

%Calculate texton representation with current texton dictionary
tmapBase1 = assignTextons(fbRun(fb,imBase1),textons');
tmapBase2 = assignTextons(fbRun(fb,imBase2),textons');
tmapTest1 = assignTextons(fbRun(fb,imTest1),textons');
tmapTest2 = assignTextons(fbRun(fb,imTest2),textons');

%Check the euclidean distances between the histograms and convince yourself that the images of the goats are closer because they have similar texture pattern
%Can you tell why we need to create a histogram before measuring the distance?
D = norm(histc(tmapBase1(:),1:k)/numel(tmapBase1) - histc(tmapTest1(:),1:k)/numel(tmapTest1))
D = norm(histc(tmapBase1(:),1:k)/numel(tmapBase1) - histc(tmapTest2(:),1:k)/numel(tmapTest2))

D = norm(histc(tmapBase2(:),1:k)/numel(tmapBase2) - histc(tmapTest1(:),1:k)/numel(tmapTest1))
D = norm(histc(tmapBase2(:),1:k)/numel(tmapBase2)  - histc(tmapTest2(:),1:k)/numel(tmapTest2))
```

**Python**
```python
import sys
sys.path.append('lib/python')

#Create a filter bank with deafult params
from fbCreate import fbCreate
fb = fbCreate()

#Load sample images from disk
from skimage import color
from skimage import io

imBase1=color.rgb2gray(io.imread('img/person1.bmp'))
imBase2=color.rgb2gray(io.imread('img/goat1.bmp'))

#Set number of clusters
k = 16*8

#Apply filterbank to sample image
from fbRun import fbRun
import numpy as np
filterResponses = fbRun(fb,np.hstack((imBase1,imBase2)))

#Computer textons from filter
from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)

#Load more images
imTest1=color.rgb2gray(io.imread('img/person2.bmp'))
imTest2=color.rgb2gray(io.imread('img/goat2.bmp'))

#Calculate texton representation with current texton dictionary
from assignTextons import assignTextons
tmapBase1 = assignTextons(fbRun(fb,imBase1),textons.transpose())
tmapBase2 = assignTextons(fbRun(fb,imBase2),textons.transpose())
tmapTest1 = assignTextons(fbRun(fb,imTest1),textons.transpose())
tmapTest2 = assignTextons(fbRun(fb,imTest2),textons.transpose())

#Check the euclidean distances between the histograms and convince yourself that the images of the goats are closer because they have similar texture pattern

# --> Can you tell why we need to create a histogram before measuring the distance? <---

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

D = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
	 histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
D = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
	 histc(tmapTest2.flatten(), np.arange(k))/tmapTest2.size)

D = np.linalg.norm(histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size - \
	 histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
D = np.linalg.norm(histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size - \
	 histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size)

```
    
## Classification

After the images are represented using a texton dictionary, train and evaluate a classifier using the provided database. Notice that the images in the mirror have been already divided into train and test sets, use this split. This was done by randomly assigning 10 images from each category to the test. Try two different classifiers:

-   **Nearest neighbour:** Use intersection of histograms or Chi-Square metrics.

	Matlab: 
	- [KNN Clasifiers](https://www.mathworks.com/help/stats/classification-using-nearest-neighbors.html#btap7k2)
	- [distance metrics] (https://www.mathworks.com/help/stats/classification-using-nearest-neighbors.html).

	Python
	- [scikit - NN](http://scikit-learn.org/stable/modules/neighbors.html).
	- [scikit - KNN](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

-   **Random forest:** 

	Matlab:
	- Use the matlab [tree bagger](http://www.mathworks.com/help/stats/treebagger.html) function. See an example at [kawahara.ca](http://kawahara.ca/matlab-treebagger-example/)

	Python:
	- [scikit - RF](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

Train both classifiers **only** with images from the *train* directory and then test them with images **ONLY** from the *test* directory. Provide the confusion matrix for training and test datasets. 

## Your Turn

The report for this laboratory must include:

-   Small (one or two paragraphs) description of the database.
-   Overall description of the method and filters used for representing the images
    -   How can we classify an image using textons? (don't be overly detailed on this, just one or two paragraphs)
    -   What does the texton representation of an image tell us?
    -   How did you create the dictionary?
    -   How many textons are you using?, Why?
    -   Can you tell if some filters are more discriminative than others?, why would this happen?
-   Description of the classifiers, hyperparameters and distance metrics
    -   What hyperparameters can you find in the classifiers? How can you choose their values?
    -   Did you apply any adjustments or preprocessing to the data? why?
-   Results
    - Provide the confusion matrix for the training and test sets, what is it telling you?. 
    - Do you have another metric to measure the performance of your method? why do you need it?
-   Discussion of the results
    -   Which classifier works best?, any idea why?
    -   How much time does it takes to create the texton dictionary? why is it so slow?
    -   How much time does it takes to train and apply both kinds of classifiers?
    -   What categories cause the most confusion? could you give some insight on why this happens?
    -   What are the limitations of the method? (CPU and RAM constraints are well known limitations, go beyond this!!)
    -   How could your method be improved?

###Some advice on creating your texton dictionary
Creating a large texton dictionary takes A LOT of time and uses A LOT of memory. Creating a dictionary with the full set of training images **is not possible, on the course server**. Unless you can find a larger machine on your own, you will have to subsample the training database. Your subsample strategy is critical for the success in this laboratory, be sure to report and discuss it. Do not forget that textures are **local patterns** that repeat over the whole images.

The report should have max 5 pages, if it is necessary use any additional pages **for references and images only**.
Use the standard CVPR sections: abstract, introduction (be concise), materials and methods, results, conclusions and references. There is no need for an 'State of the art' section. 

Upload the code you used into a directory named 'code', by extraordinary results (either good or bad), I will check it.

## Deadline 
**February 27 11:59 pm,**, as usual just upload your report to your github repo.

----

## Tip for python debugging from terminal
As I told you, you can debug in Python as you usually does it in Matlab (those pretty red points that you click on). For Python is a little different as you do not have an interface (unless you have Spyder, but, let's assume you do not). For this purpose you can use `ipython` (you might have noticed that I love it) and a module called `ipdb`.

Just `import ipdb` and call `ipdb.set_trace()` wherever you want the code to make a pause and interact with. That's it. enjoy it. **It really changed my life**. 

----

## DISCLAIMER
It is the very first time that `Python scripts` are provided. Therefore, they are in *beta* mode. Please, do not lose your mind if you find any bug or error. If so, kindly report this bug to me, I'll be happy to fix it. 

