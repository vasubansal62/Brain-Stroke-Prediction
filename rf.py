import aiofiles
import pickle
from sklearn.metrics import accuracy_score, classification_report, roc_curve, precision_recall_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib.offsetbox import AnchoredText
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# import system libs 
import os
import itertools

# import data handling tools 
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sklearn
import seaborn as sns
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import warnings
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from utils import *
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import chain
from datetime import datetime
import statistics
from tqdm import tqdm
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC, BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201, VGG16, ResNet50
from keras import backend as K
from tensorflow.keras import Sequential
import keras
import matplotlib
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import roc_curve, auc
from fastapi import FastAPI, File, UploadFile, Request
from starlette.responses import FileResponse
from pydantic import BaseModel
app = FastAPI()


warnings.filterwarnings('ignore')

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()
labels = df['stroke'].value_counts(sort=True).index
sizes = df['stroke'].value_counts(sort=True)

le = LabelEncoder()
en_df = df.apply(le.fit_transform)
en_df.head()

df = df.drop('id', axis=1)
len_data = len(df)
len_w = len(df[df["gender"] == "Male"])
len_m = len_data - len_w

men_stroke = len(df.loc[(df["stroke"] == 1) & (df['gender'] == "Male")])
men_no_stroke = len_m - men_stroke

women_stroke = len(df.loc[(df["stroke"] == 1) & (df['gender'] == "Female")])
women_no_stroke = len_w - women_stroke

labels = ['Men with stroke', 'Men healthy',
          'Women with stroke', 'Women healthy']
values = [men_stroke, men_no_stroke, women_stroke, women_no_stroke]

features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type',
            'smoking_status']
correlation_table = []
for cols in features:
    y = en_df["stroke"]
    x = en_df[cols]
    corr = np.corrcoef(x, y)[1][0]
    dict = {
        'Features': cols,
        'Correlation coefficient': corr,
        'Feat_type': 'numerical'
    }
    correlation_table.append(dict)
dF1 = pd.DataFrame(correlation_table)


X = en_df[features]
y = en_df['stroke']
en_df_imputed = en_df
imputer = KNNImputer(n_neighbors=4, weights="uniform")
imputer.fit_transform(en_df_imputed)

X, y = en_df_imputed[features], en_df_imputed["stroke"]
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23)
sm = SMOTE()
X_res, y_res = sm.fit_resample(x_train, y_train)


data = {'gender': 0,
        'age': 88, 'hypertension': 0, 'heart_disease': 1, 'ever_married': 1, 'work_type': 2, 'Residence_type': 1, 'smoking_status': 1}
new = pd.DataFrame.from_dict([data])

model = KNeighborsClassifier()
model.fit(x_train, y_train)
print("Accuracy of first model(risk factor) :",accuracy_score(y_test, model.predict(x_test)))   