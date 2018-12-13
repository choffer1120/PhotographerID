# Photographer Identification

PhotographerID is an MIT class project in which we developed a deep learning model to learn the individual photographic styles of popular instagram photograhers. To learn more about the details of the project visit [https://choffer1120.github.io/PhotographerID/](https://choffer1120.github.io/PhotographerID/)

We built two separate models, one for 10 photographers and the other for 24 photographers. The process of building those models is almost entirely the same, except for the number of classes that are specified. The following explanation is for 10 Users, but it works also for 24 Users.

# Requirements

You will require Keras 2.0 or above with Tensorflow backend to run the code. Keras can be installed very easily using package managers like pip or conda. You would also need the keras-vis package to run the filter visualization code. Check instructions to install the keras-vis package [here](https://github.com/raghakot/keras-vis).

# PreProcessing

```PreProcess_10User.py``` takes a varible ```train_path``` which is the path to the directory of all images and prepares the training and validation sets from them. The training and validation sets are stored as h5 files. ```train_h5_name``` and ```valid_h5_name``` set the filename of the h5 files. 

# Training and Validation

```resnetTraining_10User.py``` can be executed directly or through a bash script like ```runTraining.sh``` to have the training process going. In ```resnetTraining_10User.py```, ```train_path``` and ```val_path``` take the paths to the training and validation sets generated in preprocessing. After each epoch of training, the model is validated on the validation data and the results are printed. By default, training runs for 80 epochs. The model with the best validation accuracy and the model after 80 epochs of training are saved.

# Confusion Matrices

```best_epoch_10User.py``` can be used to plot the confusion matrix on the validation set. It can be executed by running ```runBestModel.sh```.

# Filter Analysis

We performed two types of filter analyses:
* Activation Maximization
* Attention Maps

```activation_maximization.py``` performs activation maximization on some random filters of conv layer ```layer_name``` of the model and saves the sticted image of the filters.
```attention.ipynb``` can be executed to find the find the attention maps on images of a given class. The layer used for attention is the last layer and the image class has to be specified.

# Others
The ```.out``` files store the outputs of running the training. They can be run by issuing the command ```cat filename.out```.
FilterAnalysis also holds the images of filter analysis that we performed for our project.
