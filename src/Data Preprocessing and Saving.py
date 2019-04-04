#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing and Saving
# 
# This notebook is used to load the data from the pickle files saved with health vitals data that has been collected using the Fitbit API. Then it applies some preprocessing and saves all the numpy arrays and dataframes into respective file formats (.h5, .csv)

# ## Importing Required Libraries

# In[46]:


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

# Importing utility functions from the code base
from utils.directory_utils import *
from utils.general_utils import *
from utils.sleep_utils import *
from data_preprocessor.get_user_data import *

# Importing Machine Learning utilities
from statsmodels.tsa.seasonal import seasonal_decompose


# ## Data
# 
# This section loads all different types of data from the pickle files that we have saved and then loads the relevant data into numpy array for further analysis

# ### Heart Rate, Sleep, Calories and Activity Time Series Data

# ##### Define the paths to the directories that contain the pickle files with the corresponding data

# In[73]:


user_name = 'Saksham'
heart_rate_data_directory_path = f'../data/data_files/{user_name}/heart/'
heart_rate_ts_data = []
calories_data_directory_path = f'../data/data_files/{user_name}/calories/'
calories_ts_data = []
activity_label_ts_data = []
sleep_data_directory_path = f'../data/data_files/{user_name}/sleep/'
sleep_efficiency_ratio = []
sleep_stages_summary = []
user_numpy_array_output_directory = f'../data/data_numpy_arrays/{user_name}/'
print(check_output_directory(user_numpy_array_output_directory))

common_file_nms = get_file_nms(heart_rate_data_directory_path, '*.pickle')
common_file_nms = common_file_nms.intersection(get_file_nms(calories_data_directory_path, '*.pickle'))
common_file_nms = common_file_nms.intersection(get_file_nms(sleep_data_directory_path, '*.pickle'))
len(common_file_nms)


# ##### Load the data from the pickle files into a list

# In[74]:


counter = 0
for file_name in sorted(list(common_file_nms)):
    heart_rate_file_path = construct_path(heart_rate_data_directory_path, file_name)
    calories_file_path = construct_path(calories_data_directory_path, file_name)
    sleep_file_path = construct_path(sleep_data_directory_path, file_name)
    heart_rate_data_raw = pickle.load(open(heart_rate_file_path, 'rb'))
    heart_rate_data = get_attribute_val_array_by_minute(heart_rate_data_raw, 'value')
    
    calories_data_raw = pickle.load(open(calories_file_path, 'rb'))
    calories_value_ts = get_attribute_val_array_by_minute(calories_data_raw, 'value')
    activity_value_ts = get_attribute_val_array_by_minute(calories_data_raw, 'level')
    sleep_data_raw = pickle.load(open(sleep_file_path, 'rb'))
    if heart_rate_data.shape != (0,) and not np.isnan(heart_rate_data).all():
        if calories_value_ts.shape != (0,) and not np.isnan(calories_value_ts).all():
            try:
                if get_sleep_stages_summary(sleep_data_raw):
                    sleep_stages = get_sleep_stages_summary(sleep_data_raw)
                    sleep_ratio = get_time_asleep_to_time_in_bed_ratio(sleep_data_raw, 'fake_user_id')
                    counter += 1

                    heart_rate_ts_data.append(heart_rate_data)
                    calories_ts_data.append(calories_value_ts)
                    activity_label_ts_data.append(activity_value_ts)
                    sleep_efficiency_ratio.append(sleep_ratio)
                    sleep_stages_summary.append(sleep_stages)
            except:
                print(file_name)


# ##### Convert the lists into the correct format: array and dataframe

# In[75]:


heart_rate_ts_data = np.array(heart_rate_ts_data)
calories_ts_data, activity_label_ts_data = np.array(calories_ts_data), np.array(activity_label_ts_data)
sleep_efficiency_ratio, sleep_stages_summary = np.array(sleep_efficiency_ratio), pd.DataFrame(list(sleep_stages_summary))


# In[76]:


# Check for the shape of all the arrays and dataframes
heart_rate_ts_data.shape, calories_ts_data.shape, activity_label_ts_data.shape, sleep_efficiency_ratio.shape, sleep_stages_summary.shape


# In[77]:


# Saving the sleep ratio and sleep stages
np.save(construct_path(user_numpy_array_output_directory, f'sleep_efficiency_ratio.npy'), sleep_efficiency_ratio)
sleep_stages_summary.to_csv(construct_path(user_numpy_array_output_directory, f'sleep_stages_summary.csv'), index=False)


# In[78]:


# Make sure activity value does not have a nan field (not sure how we would fill this)
np.isnan(activity_label_ts_data).any()


# In[79]:


# Saving the activity_value_ts array
np.save(construct_path(user_numpy_array_output_directory, f'activity_label_ts_data.npy'), activity_label_ts_data)


# ## Activity Percentages
# 
# In this section of the notebook we aggregate the activity labels of a person from minute level to percentage level

# In[82]:


activity_percentages = activity_percentage_finder(activity_label_ts_data[:, list(range(120)) + list(range(600, 24*60))], hours=16)
activity_percentages.shape


# In[83]:


# Saving the activity percentages array
np.save(construct_path(user_numpy_array_output_directory, f'activity_percentages.npy'), activity_percentages)


# ## Normalizing the Time Series Data

# In[84]:


# Remove nans from heart and calories data
heart_rate_ts_data = remove_nans_from_array(heart_rate_ts_data)
calories_ts_data = remove_nans_from_array(calories_ts_data)


# In[85]:


# Check that no nans in any of the data
np.isnan(heart_rate_ts_data).any(), np.isnan(calories_ts_data).any()


# In[86]:


# Saving the Calories and Heart Rate data
np.save(construct_path(user_numpy_array_output_directory, f'heart_rate_ts_data.npy'), heart_rate_ts_data)
np.save(construct_path(user_numpy_array_output_directory, f'calories_ts_data.npy'), calories_ts_data)


# In[87]:


# Saving the Calories and Heart Rate data
np.save(construct_path(user_numpy_array_output_directory, f'norm_heart_rate_ts_data.npy'), 
        normalize_time_series_array(heart_rate_ts_data, 24, 60))
np.save(construct_path(user_numpy_array_output_directory, f'norm_calories_ts_data.npy'), 
        normalize_time_series_array(calories_ts_data, 24, 60))

