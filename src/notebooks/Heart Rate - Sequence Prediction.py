#!/usr/bin/env python
# coding: utf-8

# # Heart Rate - Sequence Prediction
# 
# This notebook is responsible to fit different RNN models like LSTM and GRU to try to model the heart rate sequences. These models would then be used to extract the features from the given heart rate time series so far

# ## Importing Libraries

# In[2]:


# Importing scientific libarires required for analysis and handling data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

# Importing libraries related to handling of files and directory
import os
import glob
import pickle
import random

# Importing utility functions from the code base
from utils.directory_utils import *
from utils.general_utils import *
from utils.sleep_utils import *
from data_preprocessor.get_user_data import *
from clustering_utils import *
from kmeans_dm import *

# Importing Machine Learning utilities
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import boxcox
from scipy.spatial import distance
from tslearn.metrics import dtw, cdist_dtw
from sklearn.metrics import silhouette_score
from scipy.stats import entropy

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator


# In[3]:


# First we load the data for each user seperately from their own numpy array and then stack them to get the final array
numpy_array_directory = f'../data/data_numpy_arrays/'

heart_rate_ts_data = []
calories_ts_data = []
activity_label_ts_data = []
activity_percentages = []
sleep_effeciency_ratio = []
sleep_stages_summary = []

for user_name in get_subdirectory_nms(numpy_array_directory):
#     if user_name in  ['Meghna\\']:
#         continue
    user_directory = construct_path(numpy_array_directory, user_name)

    user_heart_rate_ts_data = np.load(construct_path(user_directory, f'heart_rate_ts_data.npy'))
    user_calories_ts_data = np.load(construct_path(user_directory, f'calories_ts_data.npy'))
    user_activity_label_ts_data = np.load(construct_path(user_directory, f'activity_label_ts_data.npy'))
    user_activity_percentages = np.load(construct_path(user_directory, f'activity_percentages.npy'))
    user_sleep_effeciency_ratio = np.load(construct_path(user_directory, f'sleep_efficiency_ratio.npy'))
    user_sleep_stages_summary = pd.read_csv(construct_path(user_directory, f'sleep_stages_summary.csv'))

    heart_rate_ts_data.append(user_heart_rate_ts_data)
    calories_ts_data.append(user_calories_ts_data)
    activity_label_ts_data.append(user_activity_label_ts_data)
    activity_percentages.append(user_activity_percentages)
    sleep_effeciency_ratio.append(user_sleep_effeciency_ratio)
    sleep_stages_summary.append(user_sleep_stages_summary)

heart_rate_ts_data = np.vstack(heart_rate_ts_data)[:, :]
calories_ts_data = np.vstack(calories_ts_data)[:, :]
activity_label_ts_data = np.vstack(activity_label_ts_data)[:, :]
activity_percentages = np.vstack(activity_percentages)
sleep_effeciency_ratio = np.hstack(sleep_effeciency_ratio)
sleep_stages_summary = pd.concat(sleep_stages_summary)


# In[ ]:


def generate_one_long_series(orig_data, input_shape = 15, out_shape=2):
    for day_time_series in orig_data:
        generator = TimeseriesGenerator(day_time_series, day_time_series, length=input_shape, )


# In[13]:


gen = TimeseriesGenerator(heart_rate_ts_data[0], heart_rate_ts_data[0], length=15, stride=15, batch_size=1)

for i in range(len(gen)):
	x, y = gen[i]
	print('%s => %s' % (x, y))

