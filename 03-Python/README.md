# Python tutorial
This is a basic tutorial on Python. We aim to cover general topics such as syntax and data structures, packaging, debugging, data loading and visualization.

## Warming up
- [ipython](https://ipython.org/)
- [Jupyter](http://jupyter.org/)

## Additional Info about LaTeX
- [Sharelatex](https://www.sharelatex.com)
- [Overleaf](https://www.overleaf.com/)

## FYI - Requierements and dependencies installation
It is recommended to use Python [Anaconda](https://www.continuum.io/downloads) and its powerful package manager, [conda](https://github.com/conda/conda) to install additional packages.

--------------

# Homework

Please select one dataset of your preference. If you do not have one, you might use [BDSD](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.htm). Your dataset _SHOULD_ have labels. 

Write a **python** script that does the following things:
1. Download (and untar if the case) your dataset. 
2. Choose _randomly_ an specific number (Let's say **N**, N>6) of those images. Resize them to 256x256, and save them in a new folder. 
3. Plot your original **N** images along the corresponding labels. 
4. Finally, save the variables as a _dict_, images and labels. For this purpose, you may use _numpy_, _pickle_, or any module of your preference.

Let's explore the last item. For instance, if you choose a classification problem, you might do it by showing the original images and somewhat saying the label of each one. Ej:

![fake](imgs/fake.png)

Where the label is depicted at the center of each image.

On the other hand, if you choose a detection/segmentation problem, you might do it by showing the original images along the corresponding groundtruths. Ej:

![bsds](imgs/bsds.png)

Where the first row are the original images, second and third row are the boundaries and segmentation grountruth labels respectively. 

---

Your script **must** be ONE SINGLE excecutable script that does all the aforementioned items just by typing: `./run.py`

Notes:
- Once the dataset has been downloaded, the script must skip step 1. 
- If you make use of a module that is rather uncommon (I do not know, nobody knows, internet is weird), just make sure that your script internally install it.
- Print the processing time at the end of the script. _time_ is the module you need fot it.

Bonus: 
- I could contemplate the posibility of a bonus if someone does remarkable job by reducing to a minimum the processing time. Less is more. 

--------------

## Acknowledgements

Special thanks to @ftorres11 and @andfoy for developing part of this introductory repository during July 2017 for the Biomedical Computer Vision (BCV). 


