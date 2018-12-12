from keras.preprocessing import image
from keras.applications import vgg16

import os, math
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
#following added by JMS 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py

val_path = "validate_separate_24.h5"


with h5py.File(val_path, 'r') as hf:
    X_val, y_val = hf['images'][:], hf['labels'][:]



classes = ["airpixels",
    "fursty",
    "shortstache",
    "cschoonover",
    "thiswildidea",
    "loki",
    "helloemilie",
    "danielkordan",
    "cestmaria",
    "jessfindlay",
    "aaronbhall",
          "adrienneraquel",
          "alexstrohl",
          "ChristinaMittermeier",
          "dudelum",
          "edkashi",
          "elliepritts",
          "emmett_sparling",
          "garethpon",
          "haakeaulana",
          "hannes_becker",
          "hirozzzz",
          "jasoncharleshill",
          "kelianne"
          ]


from keras.models import load_model
import functools

top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'

best_model = load_model('best_epoch_res_24User.h5', custom_objects={'top3_acc': top3_acc})
pred_probs = best_model.predict(X_val, True)
predictions = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(y_val, axis=1)


import numpy as np
import pandas as pd
predfile = pd.DataFrame(
    {'true': true_labels,
     'predictions': predictions
    }).to_csv('prediction.csv')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig = plt.gcf()
    fig.set_size_inches(16, 16)
   
    fig.savefig('cm_best_res_24User.png')

    


cm = confusion_matrix(true_labels, predictions)
    
cm_plot_labels = classes
plot_confusion_matrix(cm, cm_plot_labels, normalize = True, title = 'Confusion Matrix 24 Users')