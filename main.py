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

app = FastAPI()
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="D:\\dv\\templates")
print(templates)

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

modell = KNeighborsClassifier()
modell.fit(x_train, y_train)



@app.get("/", response_class=HTMLResponse)
async def read_items(request : Request):
    return templates.TemplateResponse('first.html', {"request": request})


from typing import Dict, Any
from pydantic import BaseModel
@app.post("/predict")
async def getdata(request: Request):
    try:
        print("here")
        body = await request.form()
        body = {k:int(v) for k,v in body.items()}
        print(body)
        pdf = pd.DataFrame.from_dict([body])
        # print(pdf)
        # print(accuracy_score(y_test, model.predict(x_test)))
        # print(model.predict(pdf))
        return templates.TemplateResponse('response1.html', {"request": request , "result" : modell.predict(pdf)})
        
    except Exception as e:
        print(e)


def define_paths(dir):
    filepaths = []
    labels = []
    folds = os.listdir(dir)
    for fold in folds:
        foldpath = os.path.join(dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)
    return filepaths, labels

def define_df(files, classes):
    Fseries = pd.Series(files, name= 'filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis= 1)

def create_df(tr_dir, val_dir, ts_dir):
    # train dataframe 
    files, classes = define_paths(tr_dir)
    train_df = define_df(files, classes)

    # validation dataframe
    files, classes = define_paths(val_dir)
    valid_df = define_df(files, classes)

    # test dataframe
    files, classes = define_paths(ts_dir)
    test_df = define_df(files, classes)
    return train_df, valid_df, test_df

def create_gens(train_df, valid_df, test_df, batch_size):
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    ts_length = len(test_df)
    test_batch_size = test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size
    def scalar(img):
        return img
    tr_gen = ImageDataGenerator(preprocessing_function= scalar, horizontal_flip= True)
    ts_gen = ImageDataGenerator(preprocessing_function= scalar)
    train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)
    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)
    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= False, batch_size= test_batch_size)
    print(test_gen)
    return train_gen, valid_gen, test_gen


def show_images(gen):
    g_dict = gen.class_indices        
    classes = list(g_dict.keys())    
    images, labels = next(gen)       
    plt.figure(figsize= (20, 20))
    length = len(labels)             
    sample = min(length, 25)         
    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255       
        plt.imshow(image)
        index = np.argmax(labels[i])  
        class_name = classes[index]   
        plt.title(class_name, color= 'blue', fontsize= 12)
        plt.axis('off')
    plt.show()

def plot_training(hist):
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)    
    val_lowest = val_loss[index_loss]    
    index_acc = np.argmax(val_acc)      
    acc_highest = val_acc[index_acc]    

    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(tr_acc))]	      
    loss_label = f'best epoch= {str(index_loss + 1)}' 
    acc_label = f'best epoch= {str(index_acc + 1)}'    
    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout
    plt.show()

def plot_confusion_matrix(cm, classes, normalize= False, title= 'Confusion Matrix', cmap= plt.cm.Blues):
	plt.figure(figsize= (10, 10))
	plt.imshow(cm, interpolation= 'nearest', cmap= cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation= 45)
	plt.yticks(tick_marks, classes)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis= 1)[:, np.newaxis]
		print('Normalized Confusion Matrix')
	else:
		print('Confusion Matrix, Without Normalization')
	print(cm)
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')
	plt.tight_layout()
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')

class GAN:

    def build_generator(dim, depth, channels=1, inputDim=100,
        outputDim=512):
        model = Sequential()
        
        inputShape = (dim, dim, depth)
        chanDim = -1
        model.add(Dense(input_dim=inputDim, units=outputDim))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(dim * dim * depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Reshape(inputShape))
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2),padding="same"))
        model.add(Activation("tanh"))
       # model.add(layers.Dense((), activation="tanh"))

        return model

    def build_discriminator(width, height, depth, alpha=0.2):
        model = Sequential()
    #    print(height,width,depth)

        inputShape = (height, width, depth)
        model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2),
        input_shape=inputShape))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))

        model.add(LeakyReLU(alpha=alpha))
        model.add(Flatten())
        print(model.output_shape)
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        return model


train_dir = 'Brain_Stroke_CT-SCAN_image/Train'
valid_dir = 'Brain_Stroke_CT-SCAN_image/Test'
test_dir = 'Brain_Stroke_CT-SCAN_image/Validation'
v1 = 'Brain_Stroke_CT-SCAN_image/v3'
v2 = 'Brain_Stroke_CT-SCAN_image/v2'
v3 = 'Brain_Stroke_CT-SCAN_image/v1'
v4 = 'Brain_Stroke_CT-SCAN_image/v4'

train_df, valid_df, test_df = create_df(train_dir, valid_dir, test_dir)
train_df, valid_df, test1_df = create_df(train_dir, valid_dir, v1)
train_df, valid_df, test2_df = create_df(train_dir, valid_dir, v2)
train_df, valid_df, test3_df = create_df(train_dir, valid_dir, v3)
train_df, valid_df, test4_df = create_df(train_dir, valid_dir, v4)


# Get Generators
batch_size = 40
train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)
train_gen, valid_gen, test1_gen = create_gens(train_df, valid_df, test1_df, batch_size)
train_gen, valid_gen, test2_gen = create_gens(train_df, valid_df, test2_df, batch_size)
train_gen, valid_gen, test3_gen = create_gens(train_df, valid_df, test3_df, batch_size)
train_gen, valid_gen, test4_gen = create_gens(train_df, valid_df, test4_df, batch_size)
show_images(train_gen)

img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys()))



# model = Sequential([
#     Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape= img_shape),
#     Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
#     MaxPooling2D((2, 2)),
    
#     Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
#     Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
#     MaxPooling2D((2, 2)),
    
#     Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
#     Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
#     Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
#     MaxPooling2D((2, 2)),
    
#     Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
#     Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
#     Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
#     MaxPooling2D((2, 2)),
    
#     Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
#     Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
#     Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
#     MaxPooling2D((2, 2)),
    
#     Flatten(),
#     Dense(256,activation = "relu"),
#     Dense(64,activation = "relu"),
#     Dense(class_count, activation = "softmax")
# ])

# model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

# model.summary()

# history = model.fit(x= train_gen, epochs= 40, verbose= 1, validation_data= valid_gen, 
#                     validation_steps= None, shuffle= False, initial_epoch= 0)
# pickle.dump(model,open("model.pkl",'wb'))

# ts_length = len(test_df)
# test_batch_size = test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
# test_steps = ts_length // test_batch_size
# train_score = model.evaluate(train_gen, steps= test_steps, verbose= 1)
# valid_score = model.evaluate(valid_gen, steps= test_steps, verbose= 1)
# test_score = model.evaluate(test_gen, steps= test_steps, verbose= 1)

# print("Train Loss: ", train_score[0])
# print("Train Accuracy: ", train_score[1])
# print('-' * 20)
# print("Validation Loss: ", valid_score[0])
# print("Validation Accuracy: ", valid_score[1])
# print('-' * 20)
# print("Test Loss: ", test_score[0])
# print("Test Accuracy: ", test_score[1])

print("CNN MODEl")
print("-------------------------------------")

from keras.models import load_model
model = load_model("model.h5")

preds = model.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)

target_names = ['Normal', 'Stroke']

# Classification report
report = classification_report(test_gen.classes, y_pred, target_names= target_names,output_dict=True)
for label in report.keys():
    if label != 'accuracy':
        print(f"Label: {label}")
        print(f"Precision: {report[label]['precision']:.2f}")
        print(f"Recall: {report[label]['recall']:.2f}")
        print(f"F1-score: {report[label]['f1-score']:.2f}")
preds = model.evaluate_generator(test_gen)
print(preds)
# # gen = GAN.build_generator(128, 64, channels=1)
# # #dis = GAN.build_discriminator(32, 32, 1)
# # dis = GAN.build_discriminator(512, 512, 1)
# # dis.trainable = False




# # optimizer = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
# # dis.compile(loss="binary_crossentropy", optimizer=optimizer)


# # ganInput = Input(shape=(100,))
# # ganOutput = dis(gen(ganInput))
# # gan = Model(ganInput, ganOutput)
# # ganOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
# # gan.compile(loss="binary_crossentropy", optimizer=optimizer)


print("GAN MODEl")
print("-------------------------------------")


exception_model = load_model("excm.h5")

preds = exception_model.predict_generator(test1_gen)
y_pred = np.argmax(preds, axis=1)

target_names = ['Normal', 'Stroke']

#Classification report
report = classification_report(test1_gen.classes, y_pred, target_names= target_names,output_dict=True)
for label in report.keys():
    if label != 'accuracy':
        print(f"Label: {label}")
        print(f"Precision: {report[label]['precision']:.2f}")
        print(f"Recall: {report[label]['recall']:.2f}")
        print(f"F1-score: {report[label]['f1-score']:.2f}")
preds = model.evaluate_generator(test1_gen)
print(preds)
# #     base_model = Xception(weights='imagenet',
# #                           include_top=False, input_shape=(299, 299, 3))
# #     x = base_model.output
# #     x = GlobalAveragePooling2D()(x)
# #     x = Dropout(0.15)(x)
# #     y_pred = Dense(6, activation='sigmoid')(x)

# #      Model(inputs=base_model.input, outputs=y_pred)


# # LR = 0.00005
# # model = create_model()
# # model.compile(optimizer=Adam(learning_rate=LR),
# #               loss='binary_crossentropy',  # <- requires balance/ Binary for unbalanced
# #               metrics=[tf.keras.metrics.AUC()])  # run both
# # BATCH_SIZE = 1  # had to revert back to 16 to have a comparaison point with the large model I ran locally
# # # Early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1,
# # #                                               mode='auto', baseline=None, restore_best_weights=False)
# # # #train_length = len(train_df)
# # # total_steps = sample_files.shape[0] // BATCH_SIZE


# # # history = model.fit_generator(
# # #     train_gen,
# # #     validation_data=val_gen,
# # #     validation_steps=total_steps * 0.15,
# # #     callbacks=[checkpoint, Early_stop],
# # #     epochs=100
# # # )
# # # model.save('tlmodel.h5')

print("Transfer Learning MODEl")
print("-------------------------------------")


transfer_model = load_model("tlmodel.h5")

preds = transfer_model.predict_generator(test2_gen)
y_pred = np.argmax(preds, axis=1)

target_names = ['Normal', 'Stroke']

#Classification report
report = classification_report(test2_gen.classes, y_pred, target_names= target_names,output_dict=True)
for label in report.keys():
    if label != 'accuracy':
        print(f"Label: {label}")
        print(f"Precision: {report[label]['precision']:.2f}")
        print(f"Recall: {report[label]['recall']:.2f}")
        print(f"F1-score: {report[label]['f1-score']:.2f}")
preds = model.evaluate_generator(test2_gen)
print(preds)
# # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # # freeze the pre-trained layers
# # for layer in base_model.layers:
# #     layer.trainable = False

# # # add custom layers on top of the pre-trained model
# # model = Sequential()
# # model.add(base_model)
# # model.add(Flatten())
# # model.add(Dense(256, activation='relu'))
# # model.add(Dense(1, activation='sigmoid'))

# # # compile the model
# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # # print summary of the model
# # model.summary()

print("VGG MODEl")
print("-------------------------------------")

vggmodel = load_model('vgg.h5')

preds = vggmodel.predict_generator(test3_gen)
y_pred = np.argmax(preds, axis=1)

target_names = ['Normal', 'Stroke']

#Classification report
report = classification_report(test3_gen.classes, y_pred, target_names= target_names,output_dict=True)
for label in report.keys():
    if label != 'accuracy':
        print(f"Label: {label}")
        print(f"Precision: {report[label]['precision']:.2f}")
        print(f"Recall: {report[label]['recall']:.2f}")
        print(f"F1-score: {report[label]['f1-score']:.2f}")
preds = model.evaluate_generator(test3_gen)
print(preds)
# # ensemble_model = Sequential()
# # ensemble_model.add(concatenate([tlmodel.output, vgg.output, model.output]))
# # ensemble_model.add(Dense(256, activation='relu'))
# # ensemble_model.add(Dropout(0.5))
# # ensemble_model.add(Dense(1, activation='sigmoid'))

# # # Compile the model
# # ensemble_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # # Print the summary of the model
# # ensemble_model.summary()

print("Ensemble MODEl")
print("-------------------------------------")

ensemble_model = load_model("ensemble.h5")

preds = ensemble_model.predict_generator(test4_gen)
y_pred = np.argmax(preds, axis=1)

target_names = ['Normal', 'Stroke']

#Classification report
report = classification_report(test4_gen.classes, y_pred, target_names= target_names,output_dict=True)
for label in report.keys():
    if label != 'accuracy':
        print(f"Label: {label}")
        print(f"Precision: {report[label]['precision']:.2f}")
        print(f"Recall: {report[label]['recall']:.2f}")
        print(f"F1-score: {report[label]['f1-score']:.2f}")
preds = model.evaluate_generator(test4_gen)
print(preds)





@app.post("/predct")
async def create_upload_file(request : Request,file: UploadFile = File(...)):
    try:
        # extension = os.path.splitext(file.filename)[1]_, path = tempfile.mkstemp(prefix='parser_', suffix=extension)
        # print("hi")
        async with aiofiles.open("static/noidea/newfile.png", 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)  # async write
            
        files, classes = define_paths("static/")
        test_df = define_df(files, classes)
        def scalar(img):
            return img
        ts_gen = ImageDataGenerator(preprocessing_function= scalar)
        test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= False, batch_size= 1)
        preds = model.predict_generator(test_gen)
        print(preds)
        return templates.TemplateResponse("response.html", {"request" : request,"result" : preds})
        # result = model.predict()
        # if(result > 0.5):
        #     return str("Risk")
        # else:
        #     return str("No risk")
        # print(img)
        # if (img):
        #     return str("No risk")
        # else:
        #     return str("Risk")
    except Exception as e:
        print(e)

# @app.post("/predct")
# async def create_upload_file(file: UploadFile = File(...)):
#     try:
#         # extension = os.path.splitext(file.filename)[1]_, path = tempfile.mkstemp(prefix='parser_', suffix=extension)
#         # print("hi")
#         async with aiofiles.open("form_responses/newfile.png", 'wb') as out_file:
#             content = await file.read()  # async read
#             await out_file.write(content)  # async write
        
#         img = tf.keras.preprocessing.image_dataset_from_directory(
#             "form_responses",
#             batch_size=1,
#             target_size= (224, 224), 
#             class_mode= 'categorical',
#             color_mode= 'rgb', 
#             shuffle= False, 
#         )
#         # test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
#         #                                 color_mode= 'rgb', shuffle= False, batch_size= test_batch_size)
        
#         result = model.predict_generator(img)
#         if (result > 0.5):
#             return str("Risk")
#         else:
#             return str("No risk")
#         # print(img)
#         # if (img):
#         #     return str("No risk")
#         # else:
#         #     return str("Risk")
#     except Exception as e:
#         print(e)