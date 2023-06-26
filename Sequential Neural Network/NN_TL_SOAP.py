import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import statistics as stats
import random as rn
import os
import tensorflow.compat.v1 as tf
import keras
import warnings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, GaussianNoise, GaussianDropout
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.activations import relu
from keras import initializers as init
from keras.constraints import max_norm
from keras.utils.vis_utils import plot_model
from tensorflow.compat.v1.keras import backend as K
from math import sqrt
from IPython.display import display
from matplotlib.lines import Line2D
from collections import defaultdict
from IPython.display import Image
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

K.clear_session()
tf.reset_default_graph()
warnings.filterwarnings("ignore")

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(23)
rn.seed(23)
tf.random.set_random_seed(23)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

pd.set_option("display.max_columns", None)

df = pd.read_csv("ML_TL_SOAP.csv", sep = ',')
df = df.dropna(axis=1, how='all')

x = df.iloc[:, 14:114]
y = df.iloc[:, 8:14]

# separate categorical and continuous data 
categorical=pd.DataFrame()
continuous=pd.DataFrame()

for column in x.columns:
    if(str(x[column].dtypes) == "int64"):
        categorical[column]=x[column]
    elif(str(x[column].dtypes) == "float64"):
        continuous[column]=x[column]
    else:
        pass
                
# one hot encode categorical data
onehotencoder = OneHotEncoder()
categorical = onehotencoder.fit_transform(categorical).toarray()
   
# standardize continuous data
input_scaler = StandardScaler()
continuous = input_scaler.fit_transform(continuous)

# re-combine categorical and continuous x values
x = np.concatenate((continuous, categorical), axis=1)
y = y.to_numpy()

in_dim = x.shape[1]
out_dim = y.shape[1]

x = pd.DataFrame(x)
y = pd.DataFrame(y)

des_labels = df[['Type', 'Site']]

enc = OrdinalEncoder(dtype=np.int)
encode_labels = enc.fit_transform(des_labels)
labels = pd.DataFrame(encode_labels, columns=["Type", "Site"])
labels = labels.applymap(str)
labels = labels[["Type", "Site"]].apply(lambda x: ''.join(x), axis=1)

# encode the new string col to 0-14 (15 total classes - 3 sctypes x 5 defsites)
combined_labels = np.array(labels).reshape(-1, 1)
combined_labels = enc.fit_transform(combined_labels)

des_labels = des_labels.to_numpy()

def construct_model(k_reg, dropout_rate, learning_rate, noise, adam_decay, activation = 'relu'):
    
    # Create and add layers to model
    model = Sequential()
    #### Input layer
    model.add(Dense(out_dim*308,
                    input_dim=x.shape[1], 
                    activation=activation,
                    kernel_regularizer=l2(k_reg)
                    )
             )
    model.add(GaussianNoise(noise))
    model.add(Dense(out_dim*154, activation=activation))
    model.add(Dense(out_dim*154, activation=activation))
    model.add(Dense(out_dim))

    # configure optimizer & compile model
    opt = Adam(lr=learning_rate, decay=adam_decay)
    model.compile(loss="mse", optimizer=opt)
    
    return model

def train_model(k_reg, dropout_rate, learning_rate, noise, adam_decay, verbose, epochs, batch_size, random_state):

    model = construct_model(k_reg, dropout_rate, learning_rate, noise, adam_decay)

    skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    
    RMSE_32_train = []
    RMSE_21_train = []
    RMSE_10_train = []
    RMSE_01_train = []
    RMSE_12_train = []
    RMSE_23_train = []

    RMSE_32_test = []
    RMSE_21_test = []
    RMSE_10_test = []
    RMSE_01_test = []
    RMSE_12_test = []
    RMSE_23_test = []

    RMSE_list = []

    for train_index, test_index in skf.split(des_labels, combined_labels):
        
        # split training and testing data
        x_train, x_test = x.iloc[train_index,:], x.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
    
        # train model
        prediction = model.fit(x_train, y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose)
    
        # Finding mean RMSE of testing data
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        
        for i in range(0, 6):
            RMSE_train = sqrt(mean_squared_error(y_train.iloc[:,i], pred_train[:,i]))
            RMSE_test = sqrt(mean_squared_error(y_test.iloc[:,i], pred_test[:,i]))
            if i == 0:
                RMSE_32_train.append(RMSE_train)
                RMSE_32_test.append(RMSE_test)
            elif i == 1:
                RMSE_21_train.append(RMSE_train)
                RMSE_21_test.append(RMSE_test)
            elif i == 2:
                RMSE_10_train.append(RMSE_train)
                RMSE_10_test.append(RMSE_test)
            elif i == 3:
                RMSE_01_train.append(RMSE_train)
                RMSE_01_test.append(RMSE_test)
            elif i == 4:
                RMSE_12_train.append(RMSE_train)
                RMSE_12_test.append(RMSE_test)
            else:
                RMSE_23_train.append(RMSE_train)
                RMSE_23_test.append(RMSE_test)
    
    mean_train_32 = stats.mean(RMSE_32_train)
    mean_test_32 = stats.mean(RMSE_32_test)
    std_train_32 = stats.stdev(RMSE_32_train)
    std_test_32 = stats.stdev(RMSE_32_test)

    mean_train_21 = stats.mean(RMSE_21_train)
    mean_test_21 = stats.mean(RMSE_21_test)
    std_train_21 = stats.stdev(RMSE_21_train)
    std_test_21 = stats.stdev(RMSE_21_test)

    mean_train_10 = stats.mean(RMSE_10_train)
    mean_test_10 = stats.mean(RMSE_10_test)
    std_train_10 = stats.stdev(RMSE_10_train)
    std_test_10 = stats.stdev(RMSE_10_test)

    mean_train_01 = stats.mean(RMSE_01_train)
    mean_test_01 = stats.mean(RMSE_01_test)
    std_train_01 = stats.stdev(RMSE_01_train)
    std_test_01 = stats.stdev(RMSE_01_test)

    mean_train_12 = stats.mean(RMSE_12_train)
    mean_test_12 = stats.mean(RMSE_12_test)
    std_train_12 = stats.stdev(RMSE_12_train)
    std_test_12 = stats.stdev(RMSE_12_test)

    mean_train_23 = stats.mean(RMSE_23_train)
    mean_test_23 = stats.mean(RMSE_23_test)
    std_train_23 = stats.stdev(RMSE_23_train)
    std_test_23 = stats.stdev(RMSE_23_test)

    RMSE_32 = [mean_train_32, std_train_32, mean_test_32, std_test_32]
    RMSE_21 = [mean_train_21, std_train_21, mean_test_21, std_test_21]
    RMSE_10 = [mean_train_10, std_train_10, mean_test_10, std_test_10]
    RMSE_01 = [mean_train_01, std_train_01, mean_test_01, std_test_01]
    RMSE_12 = [mean_train_12, std_train_12, mean_test_12, std_test_12]
    RMSE_23 = [mean_train_23, std_train_23, mean_test_23, std_test_23]

    #RMSE_list.append(RMSE_32)
    #RMSE_list.append(RMSE_21)
    #RMSE_list.append(RMSE_10)
    #RMSE_list.append(RMSE_01)
    #RMSE_list.append(RMSE_12)
    #RMSE_list.append(RMSE_23)

    RMSE_list = RMSE_32 + RMSE_21 + RMSE_10 + RMSE_01 + RMSE_12 + RMSE_23

    # clear session & reset model graphs
    K.clear_session()
    tf.reset_default_graph()
    return RMSE_list

## parameters
k_folds = 5
repetition = 10  #20

epochs = 500           #200
batch_size = 10       #50
learning_rate = 0.001059083168892229   #5e-4   
adam_decay = 0.0004768871687821953     #5e-4
beta1= 0.001    #0.005
beta2=0.999     #0.995
amsgrad=False
dropout = 0.25   #0.25
noise = 0  #1e-2
kernal_regularizer = 0   #0.025
hidden_neurons = 40  #40

random_state = [0, 200, 400, 600, 800]

verbose = 0
plot_graph=True

strings = []
for i in random_state:

    RMSE_test = train_model(kernal_regularizer, dropout,learning_rate, noise, adam_decay, verbose, epochs, batch_size, i)
    
    print(RMSE_test)

    results = ','.join(str(i) for i in RMSE_test)

    strings.append(results + '   ')

with open("TL_SOAP_RMSE.txt", "w") as fp:
    fp.writelines(strings)

