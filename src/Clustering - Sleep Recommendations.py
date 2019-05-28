#!/usr/bin/env python
# coding: utf-8

# # Clustering - Sleep Recommendations
# 
# This notebook is used to produce results related to clustering of data from the fitbit vitals data loaded from the corresponding pickle files and using sleep efficiency labels to then further find cluster impurities, distrinution and good sleep reciepes

# ## Importing Required Libraries

# In[1]:


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


# ## Data
# 
# This section loads all different types of data from the pickle files that we have saved and then loads the relevant data into numpy array for further analysis

# ### Heart Rate, Sleep, Calories and Activity Time Series Data

# #### User Data Loader

# In[4]:


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


# In[3]:


activity_percentages = activity_percentages * 1440 / 100


# #### Check for the shape of all the arrays and dataframes

# In[5]:


# Check for the shape of all the arrays and dataframes
heart_rate_ts_data.shape, calories_ts_data.shape, activity_label_ts_data.shape, sleep_effeciency_ratio.shape, sleep_stages_summary.shape


# In[6]:


# Make sure activity value does not have a nan field (not sure how we would fill this)
print(np.isnan(activity_label_ts_data).any())
# Check that no nans in any of the data
np.isnan(heart_rate_ts_data).any(), np.isnan(calories_ts_data).any()


# # Transformations
# 
# This section uses different ways to transform the original time series data

# ### Heart Rate and Calories Trends
# 
# This section will essentially find the trends from the original data

# In[7]:


trend_window_length = 10


# #### Heart Trends

# In[8]:


heart_trends = []
counter = 0
for day in heart_rate_ts_data:
    counter += 1
    result = seasonal_decompose(day, model='additive', freq=trend_window_length, extrapolate_trend='freq')
    heart_trends.append(result.trend)
heart_trends = np.array(heart_trends)
heart_trends = remove_nans_from_array(heart_trends)
# Make sure the shape is same and there are no nan values
heart_trends.shape, np.isnan(heart_trends).any()


# In[9]:


# plotting heart trends to asses the fit to the overall data
plt.figure(figsize=(10, 5))
plt.plot(heart_rate_ts_data[0, :], lw=2, label='Original Heart Rate')
plt.plot(heart_trends[0, :], color='r', lw=2, label='Decomposed Heart Trends')
plt.xlabel('Minute')
plt.ylabel('BPM')
plt.legend()


# #### Calories Trends

# In[10]:


calories_trends = []
for day in calories_ts_data:
    result = seasonal_decompose(day, model='additive', freq=trend_window_length, extrapolate_trend='freq')
    calories_trends.append(result.trend)
calories_trends = np.array(calories_trends)
calories_trends = remove_nans_from_array(calories_trends)
# Make sure the shape is same and there are no nan values
calories_trends.shape, np.isnan(calories_trends).any()


# In[11]:


# plotting caloires trends to asses the fit to the overall data
plt.figure(figsize=(10, 5))
plt.plot(calories_ts_data[0, :], lw=2, label='Original Calories Burned')
plt.plot(calories_trends[0, :], color='r', lw=2, label='Decomposed Calories Burned Trends')
plt.xlabel('Minute')
plt.ylabel('Calories Burned')
plt.legend()


# # Chipping the Data
# 
# This section chips away some heart data

# In[12]:


heart_trends = heart_trends[:, 480:1200]
calories_trends = calories_trends[:, 480:1200]
heart_trends.shape, calories_trends.shape


# ### Dimensionality Reduction
# 
# This section will reduce the dimensions of the arrays so that we can easily apply different clustering techniques on them

# In[13]:


mean_window_length = 10


# In[14]:


# Reduce the dimension of the arrays
reduced_heart_trends = reduce_time_series_dimension(heart_trends, mean_window_length, hours=12)
reduced_calories_trends = reduce_time_series_dimension(calories_trends, mean_window_length, hours=12)
# Check for the shape of the arrays
reduced_heart_trends.shape, reduced_calories_trends.shape


# ## Sleep Labels
# 
# In this section of the notebook we try to find the optimal boundary for constructing the sleep labels using different techniques

# In[34]:


# Constructing a histogram plot for the sleep efficiency ratio.
# Sleep Efficiency Ratio is found as total_time_asleep / total_time_in_bed
sns.distplot(sleep_effeciency_ratio, kde=False)
plt.xlabel('Sleep Efficiency')
plt.ylabel('Frequency')
plt.title('Distribution of sleep efficiency of all subjects')


# In[14]:


# Constructing a histogram plot for the different sleep stages.
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
sns.distplot(sleep_stages_summary['wake'], ax = ax[0, 0])
ax[0, 0].set_xlabel('Minutes Awake')
ax[0, 0].set_ylabel('Frequency')
ax[0, 0].set_title('Minutes Awake Histogram')

sns.distplot(sleep_stages_summary['light'], ax = ax[0, 1])
ax[0, 1].set_xlabel('Minutes in Light Sleep')
ax[0, 1].set_ylabel('Frequency')
ax[0, 1].set_title('Minutes in Light Sleep Histogram')

sns.distplot(sleep_stages_summary['rem'], ax = ax[1, 0])
ax[1, 0].set_xlabel('Minutes in Rem Sleep')
ax[1, 0].set_ylabel('Frequency')
ax[1, 0].set_title('Minutes in REM Sleep Histogram')

sns.distplot(sleep_stages_summary['deep'], ax = ax[1, 1])
ax[1, 1].set_xlabel('Minutes in Deep Sleep')
ax[1, 1].set_ylabel('Frequency')
ax[1, 1].set_title('Minutes in Deep Sleep Histogram')


# ### Gap Definition For Sleep Efficiency

# Create a gap of certain length: Which will be a __parameter__
# 
# Example: 0.05 - 0.875 and above, 0.825 and below

# In[15]:


final_sleep_labels = sleep_effeciency_ratio > 0.89
sns.barplot(['Poor Sleep', 'Good Sleep'], [np.sum(~final_sleep_labels), np.sum(final_sleep_labels)], hue=[True, True])
plt.legend([])
plt.title('Number of Records v/s Sleep Class')
plt.ylabel('Number of Records')


# ### HeatMap for Euclidean and DTW Distances

# In[16]:


good_sleep_heart_trends = reduced_heart_trends[final_sleep_labels]
poor_sleep_heart_trends = reduced_heart_trends[~final_sleep_labels]
print(good_sleep_heart_trends.shape, poor_sleep_heart_trends.shape)
ordered_heart_trends = np.vstack((good_sleep_heart_trends, poor_sleep_heart_trends))
print(ordered_heart_trends.shape)


# In[17]:


good_sleep_calories_trends = reduced_calories_trends[final_sleep_labels]
poor_sleep_calories_trends = reduced_calories_trends[~final_sleep_labels]
print(good_sleep_calories_trends.shape, poor_sleep_calories_trends.shape)
ordered_calories_trends = np.vstack((good_sleep_calories_trends, poor_sleep_calories_trends))
print(ordered_calories_trends.shape)


# In[19]:


get_ipython().run_cell_magic('time', '', 'dtw_dist_heart = cdist_dtw(ordered_heart_trends)\ndtw_dist_calories = cdist_dtw(ordered_calories_trends)')


# In[20]:


get_ipython().run_cell_magic('time', '', 'euc_dist_heart = distance.cdist(ordered_heart_trends, ordered_heart_trends)\neuc_dist_calories = distance.cdist(ordered_calories_trends, ordered_calories_trends)')


# In[21]:


m_dist_heart = distance.cdist(ordered_heart_trends, ordered_heart_trends, 'mahalanobis')
m_dist_calories = distance.cdist(ordered_calories_trends, ordered_calories_trends, 'mahalanobis')
l1_dist_heart = distance.cdist(ordered_heart_trends, ordered_heart_trends, 'minkowski', p=1)
l1_dist_calories = distance.cdist(ordered_calories_trends, ordered_calories_trends, 'minkowski', p=1)


# In[26]:


cor_dist_heart = distance.cdist(ordered_heart_trends, ordered_heart_trends, 'correlation')
cor_dist_calories = distance.cdist(ordered_calories_trends, ordered_calories_trends, 'correlation')


# In[79]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(dtw_dist_heart, xticklabels=137, yticklabels=137, ax=ax[0])
ax[0].set_title('DTW Distance Cross Matrix for Heart Trends')
sns.heatmap(dtw_dist_calories, xticklabels=137, yticklabels=137, ax=ax[1])
ax[1].set_title('DTW Distance Cross Matrix for Calories Trends')


# In[78]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(euc_dist_heart, xticklabels=137, yticklabels=137, ax=ax[0])
ax[0].set_title('L-2 Norm Distance Cross Matrix for Heart Trends')
sns.heatmap(euc_dist_calories, xticklabels=137, yticklabels=137, ax=ax[1])
ax[1].set_title('L-2 Norm Distance Cross Matrix for Calories Trends')


# In[74]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(m_dist_heart, xticklabels=137, yticklabels=137, ax=ax[0])
ax[0].set_title('All Sleep Mahalanobis Distance Cross Matrix for Heart Trends')
sns.heatmap(m_dist_calories, xticklabels=137, yticklabels=137, ax=ax[1])
ax[1].set_title('All Sleep Mahalanobis Distance Cross Matrix for Calories Trends')


# In[77]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(l1_dist_heart, xticklabels=137, yticklabels=137, ax=ax[0])
ax[0].set_title('L1 Norm Distance Cross Matrix for Heart Trends')
sns.heatmap(l1_dist_calories, xticklabels=137, yticklabels=137, ax=ax[1])
ax[1].set_title('L1 Norm Distance Cross Matrix for Calories Trends')


# In[76]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(cor_dist_heart, xticklabels=137, yticklabels=137, ax=ax[0])
ax[0].set_title('Correlation Cross Matrix for Heart Trends')
sns.heatmap(cor_dist_calories, xticklabels=137, yticklabels=137, ax=ax[1])
ax[1].set_title('Correlation Cross Matrix for Calories Trends')


# ## Activity Percentages
# 
# In this section of the notebook we aggregate the activity labels of a person from minute level to percentage level

# In[ ]:


# Constructing a histogram plot for the different activity level percentages.
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
sns.distplot(activity_percentages[:, 0], ax = ax[0, 0])
ax[0, 0].set_xlabel('% Sedentary Activity')
ax[0, 0].set_ylabel('Frequency')
ax[0, 0].set_title('% Sedentary Activity Histogram')

sns.distplot(activity_percentages[:, 1], ax = ax[0, 1])
ax[0, 1].set_xlabel('% Light Activity')
ax[0, 1].set_ylabel('Frequency')
ax[0, 1].set_title('% Light Activity Histogram')

sns.distplot(activity_percentages[:, 2], ax = ax[1, 0])
ax[1, 0].set_xlabel('% Moderate Activity')
ax[1, 0].set_ylabel('Frequency')
ax[1, 0].set_title('% Moderate Activity Histogram')

sns.distplot(activity_percentages[:, 3], ax = ax[1, 1])
ax[1, 1].set_xlabel('% Vigorous Activity')
ax[1, 1].set_ylabel('Frequency')
ax[1, 1].set_title('% Vigorous Activity Histogram')


# In[ ]:


# Constructing a histogram plot for the different activity level percentages visualizing with respect to the good sleep label
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
sns.distplot(activity_percentages[~final_sleep_labels, 0], ax = ax[0, 0], color='red', label='Poor Sleep')
sns.distplot(activity_percentages[final_sleep_labels, 0], ax = ax[0, 0], color='green', label='Good Sleep')
ax[0, 0].set_xlabel('% Sedentary Activity')
ax[0, 0].set_ylabel('Frequency')
ax[0, 0].set_title('% Sedentary Activity Histogram')
ax[0, 0].legend()

sns.distplot(activity_percentages[~final_sleep_labels, 1], ax = ax[0, 1], color='red', label='Poor Sleep')
sns.distplot(activity_percentages[final_sleep_labels, 1], ax = ax[0, 1], color='green', label='Good Sleep')
ax[0, 1].set_xlabel('% Light Activity')
ax[0, 1].set_ylabel('Frequency')
ax[0, 1].set_title('% Light Activity Histogram')
ax[0, 1].legend()

sns.distplot(activity_percentages[~final_sleep_labels, 2], ax = ax[1, 0], color='red', label='Poor Sleep')
sns.distplot(activity_percentages[final_sleep_labels, 2], ax = ax[1, 0], color='green', label='Good Sleep')
ax[1, 0].set_xlabel('% Moderate Activity')
ax[1, 0].set_ylabel('Frequency')
ax[1, 0].set_title('% Moderate Activity Histogram')
ax[1, 0].legend()

sns.distplot(activity_percentages[~final_sleep_labels, 3], ax = ax[1, 1], color='red', label='Poor Sleep')
sns.distplot(activity_percentages[final_sleep_labels, 3], ax = ax[1, 1], color='green', label='Good Sleep')
ax[1, 1].set_xlabel('% Vigorous Activity')
ax[1, 1].set_ylabel('Frequency')
ax[1, 1].set_title('% High Activity Histogram')
ax[1, 1].legend()


# ## Clustering
# 
# In this section of the notebook we apply different clustering techniques on the data that we have got and see what are the different recipes

# In[18]:


num_master_clusters = 4
num_activity_clusters = 12


# ### K-Means - Euclidean
# 
# Here we apply K-Means on the data with euclidean (L-2 Norm) as the distance metric

# #### Getting the Best Model

# In[118]:


kmeans_mod = get_purest_clustering_model(lambda num_clusters: KMeans(num_clusters), reduced_heart_trends, final_sleep_labels)


# #### Fitting the Model

# In[124]:


# Set the seed so that get the same clustering everytime
# random.seed(2)
# np.random.seed(1000)
# Performing the Clustering
kmeans_mod = KMeans(n_clusters=12)
kmeans_mod.fit(reduced_heart_trends)
cluster_assignments = kmeans_mod.predict(reduced_heart_trends)
sil_score = silhouette_score(reduced_heart_trends, cluster_assignments)
print(kmeans_mod.n_clusters, sil_score)
np.unique(cluster_assignments, return_counts=True)


# In[126]:


get_all_clusters_sleep_purity(cluster_assignments, final_sleep_labels)


# In[127]:


# Update the number of activity clusters based on the minimum amount of records assigned to a cluster
num_activity_clusters = min(num_activity_clusters, *(np.unique(cluster_assignments, return_counts=True)[1]))
print('Updated Number of activity clusters:', num_activity_clusters)


# In[98]:


# Visualizing the number of points in each cluster
sns.distplot(cluster_assignments, kde=False)


# #### Visualization of Clusters

# In[105]:


# Simple Cluster Visualization
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
plt.figure(figsize=(7, 5))
# sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], hue=cluster_assignments, style=cluster_assignments)
sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], hue=cluster_assignments)#, size=cluster_assignments)
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')
plt.title('Visualization of Clusters')
# plt.legend([f'Cluster: {i+1}' for i in range(4)])
plt.legend([])


# In[103]:


# Cluster Visualization based on Sleep Efficiency
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
plt.figure(figsize=(7, 5))
sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], size=final_sleep_labels, hue=cluster_assignments)
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')
plt.title('Clusters Visualized')
plt.legend([])


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# Simple Cluster Visualization
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], hue=cluster_assignments, style=cluster_assignments, ax=ax[0])
ax[0].set_xlabel('PCA Dim 1')
ax[0].set_ylabel('PCA Dim 2')
ax[0].set_title('Clusters Visualized')
ax[0].legend([f'Cluster: {i+1}' for i in range(4)])

# Cluster Visualization based on Sleep Efficiency
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], hue=final_sleep_labels, style=cluster_assignments, ax=ax[1])
ax[1].set_xlabel('PCA Dim 1')
ax[1].set_ylabel('PCA Dim 2')
ax[1].set_title('Clusters Visualized')
ax[1].legend([])


# #### Cluster Purity
# 
# Finding cluster purity based on the sleep labels

# In[ ]:


# Clustering Purity is defined by ratio of dominant class of sleep label instance in the cluster 
# to total number of instances in the cluster
for master_cluster_num in range(len(kmeans_mod.cluster_centers_)):
    cluster_sleep_labels = final_sleep_labels[cluster_assignments == master_cluster_num]
    pos_sleep_label_purity = sum(cluster_sleep_labels) / cluster_sleep_labels.shape[0]
    print(f'Cluster Number: {master_cluster_num}, Purity:', max(pos_sleep_label_purity, 1 - pos_sleep_label_purity))


# #### Sub-Clustering on Activity Data

# In[128]:


sub_clusters = activity_percentage_clusterer(KMeans(n_clusters=num_activity_clusters), cluster_assignments, activity_percentages)


# In[107]:


# Sanity Check for the number of points in each cluster
print(np.unique(cluster_assignments, return_counts=True))
for sub_cluster in sub_clusters:
    print(sub_cluster.shape)


# ##### Cluster Purity in each subcluster

# In[108]:


# Clustering Purity is defined by ratio of dominant class of sleep label instance in the cluster
# to total number of instances in the cluster
for index, sub_cluster in enumerate(sub_clusters):
    print('Master Cluster:', index+1)
    cluster_sleep_labels = final_sleep_labels[(cluster_assignments == index)]
    for sub_cluster_assignment in range(num_activity_clusters):
        sub_cluster_sleep_labels = cluster_sleep_labels[(sub_cluster==sub_cluster_assignment)]
        try:
            pos_sleep_label_purity = sum(sub_cluster_sleep_labels) / sub_cluster_sleep_labels.shape[0]
            print(f'Sub Cluster Number: {sub_cluster_assignment}, Purity:', max(pos_sleep_label_purity, 0))#, 1 - pos_sleep_label_purity))
        except:
            print(f'Sub Cluster Number: {sub_cluster_assignment}, No Points assigned')


# In[129]:


sleep_recipes = get_good_sleep_recipes(cluster_assignments, sub_clusters, activity_percentages, final_sleep_labels)
sleep_recipes * 720 / 100


# In[132]:


for i, sleep_recipe in enumerate(sleep_recipes):
    plt.figure(i)
    plt.bar(['Sedentary', 'Light', 'Moderate', 'Vigorous'], (sleep_recipe * 720 / 100))
    plt.ylabel('Minutes')
    plt.title('Activity Recipes for Sleep')


# ### K-Means - DTW
# 
# Here we apply K-Means on the data with Dynamic Time Wrapping (DTW) as the distance metric

# In[ ]:


num_activity_clusters = 2


# #### Fitting the Model

# In[ ]:


clusterer = get_best_clustering_model(lambda num_clusters: TimeSeriesKMeans(num_clusters, metric='dtw', max_iter=50), 
                                       reduced_heart_trends, cluster_range=range(2, 3))


# In[ ]:


clusterer


# In[ ]:


clusterer.labels_


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Setting the seed\nclusterer.fit(reduced_heart_trends)\ncluster_assignments = clusterer.labels_\nsil_score = silhouette_score(reduced_heart_trends, cluster_assignments)\nprint(clusterer.n_clusters, sil_score)\nnp.unique(cluster_assignments, return_counts=True)')


# In[ ]:


print(np.unique(cluster_assignments, return_counts=True))


# In[ ]:


# Update the number of activity clusters based on the minimum amount of records assigned to a cluster
num_activity_clusters = min(num_activity_clusters, *(np.unique(cluster_assignments, return_counts=True)[1]))
print('Updated Number of activity clusters:', num_activity_clusters)


# In[ ]:


# Visualizing the number of points in each cluster
sns.distplot(cluster_assignments, kde=False)


# #### Visualization of Clusters

# In[ ]:


# Simple Cluster Visualization
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
plt.figure(figsize=(7, 5))
sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], hue=cluster_assignments, style=cluster_assignments)
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')
plt.title('Clusters Visualized')
plt.legend([f'Cluster: {i+1}' for i in range(4)])


# In[ ]:


# Cluster Visualization based on Sleep Efficiency
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
plt.figure(figsize=(7, 5))
sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], hue=final_sleep_labels, style=cluster_assignments)
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')
plt.title('Clusters Visualized')
plt.legend([])


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# Simple Cluster Visualization
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], hue=cluster_assignments, style=cluster_assignments, ax=ax[0])
ax[0].set_xlabel('PCA Dim 1')
ax[0].set_ylabel('PCA Dim 2')
ax[0].set_title('Clusters Visualized')
ax[0].legend([f'Cluster: {i+1}' for i in range(4)])

# Cluster Visualization based on Sleep Efficiency
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], hue=final_sleep_labels, style=cluster_assignments, ax=ax[1])
ax[1].set_xlabel('PCA Dim 1')
ax[1].set_ylabel('PCA Dim 2')
ax[1].set_title('Clusters Visualized')
ax[1].legend([])


# #### Cluster Purity
# 
# Finding cluster purity based on the sleep labels

# In[ ]:


# Clustering Purity is defined by ratio of dominant class of sleep label instance in the cluster 
# to total number of instances in the cluster
for master_cluster_num in np.unique(cluster_assignments):
    cluster_sleep_labels = final_sleep_labels[cluster_assignments == master_cluster_num]
    pos_sleep_label_purity = sum(cluster_sleep_labels) / cluster_sleep_labels.shape[0]
    print(f'Cluster Number: {master_cluster_num}, Purity:', max(pos_sleep_label_purity, 1 - pos_sleep_label_purity))


# #### Sub-Clustering on Activity Data

# In[ ]:


sub_clusters = activity_percentage_clusterer(TimeSeriesKMeans(num_activity_clusters, metric='dtw', max_iter=50), cluster_assignments, activity_percentages)


# In[ ]:


# Sanity Check for the number of points in each cluster
print(np.unique(cluster_assignments, return_counts=True))
for sub_cluster in sub_clusters:
    print(sub_cluster.shape)


# ##### Cluster Purity in each subcluster

# In[ ]:


# Clustering Purity is defined by ratio of dominant class of sleep label instance in the cluster
# to total number of instances in the cluster
for index, sub_cluster in enumerate(sub_clusters):
    print('Master Cluster:', index+1)
    cluster_sleep_labels = final_sleep_labels[(cluster_assignments == index)]
    for sub_cluster_assignment in range(num_activity_clusters):
        sub_cluster_sleep_labels = cluster_sleep_labels[(sub_cluster==sub_cluster_assignment)]
        try:
            pos_sleep_label_purity = sum(sub_cluster_sleep_labels) / sub_cluster_sleep_labels.shape[0]
            print(f'Sub Cluster Number: {sub_cluster_assignment}, Purity:', max(pos_sleep_label_purity, 1 - pos_sleep_label_purity))
        except:
            print(f'Sub Cluster Number: {sub_cluster_assignment}, No Points assigned')


# In[ ]:


sleep_recipes = get_good_sleep_recipes(cluster_assignments, sub_clusters, activity_percentages, final_sleep_labels)
sleep_recipes


# ### K-Means - KL Divergence
# 
# Here we apply K-Means on the data with K-L Divergence as the distance metric

# #### Defining the distance function using the K-L Divergence

# In[19]:


def k_l_distance(x, y):
    return (entropy(x, y) + entropy(y, x))/ 2


# In[81]:


kl_dist_heart = cdist(ordered_heart_trends, ordered_heart_trends, metric=k_l_distance)
kl_dist_calories = cdist(ordered_calories_trends, ordered_calories_trends, metric=k_l_distance)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(kl_dist_heart, xticklabels=137, yticklabels=137, ax=ax[0])
ax[0].set_title('K-L Divergence Cross Matrix for Heart Trends')
sns.heatmap(kl_dist_calories, xticklabels=137, yticklabels=137, ax=ax[1])
ax[1].set_title('K-L Divergence Cross Matrix for Calories Trends')


# #### Best Model

# In[20]:


kl_best_mod = get_purest_clustering_model(lambda num_clusters: KL_Kmeans(num_clusters), reduced_heart_trends, 
                                          final_sleep_labels)


# #### Fitting the Model

# In[75]:


# Set the seed so that get the same clustering everytime
# random.seed(2)
# np.random.seed(1000)
# Performing the Clustering
# randomcentres = randomsample(reduced_heart_trends, kl_best_mod.get_num_clusters())
randomcentres = randomsample(reduced_heart_trends, 6)
centres, cluster_assignments, dist = kmeans(reduced_heart_trends, randomcentres, metric=k_l_distance, maxiter=200)
sil_score = silhouette_score(reduced_heart_trends, cluster_assignments, metric=k_l_distance)
print(len(centres), sil_score)
np.unique(cluster_assignments, return_counts=True)


# In[76]:


get_all_clusters_sleep_purity(cluster_assignments, final_sleep_labels, measure='gini')


# In[77]:


# Update the number of activity clusters based on the minimum amount of records assigned to a cluster
num_activity_clusters = min(num_activity_clusters, *(np.unique(cluster_assignments, return_counts=True)[1]))
print('Updated Number of activity clusters:', num_activity_clusters)


# In[78]:


# Visualizing the number of points in each cluster
sns.distplot(cluster_assignments, kde=False)


# #### Visualization of Clusters

# In[37]:


# Simple Cluster Visualization
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
plt.figure(figsize=(7, 5))
sns.scatterplot(pca_heart[:, 0], pca_heart[:, 1], hue=cluster_assignments)#, style=cluster_assignments)
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')
plt.title('Clusters Visualized')
# plt.legend([f'Cluster: {i+1}' for i in range(8)])
plt.legend([])


# In[71]:


# Cluster Visualization based on Sleep Efficiency
pca_mod = PCA(2)
pca_heart = pca_mod.fit_transform(reduced_heart_trends)
plt.figure(figsize=(7, 5))
sns.scatterplot(pca_heart[final_sleep_labels, 0], pca_heart[final_sleep_labels, 1], marker='X', hue=cluster_assignments[final_sleep_labels])
sns.scatterplot(pca_heart[~final_sleep_labels, 0], pca_heart[~final_sleep_labels, 1], marker='o', hue=cluster_assignments[~final_sleep_labels])#, hue=cluster_assignments)
# plt.scatter(pca_heart[:, 0], pca_heart[:, 1], s=np.array(final_sleep_labels, dtype=np.int)*10, cmap='viridis', c=cluster_assignments)
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')
# plt.colorbar(label='Cluster')
plt.title('Clusters Visualized with Good Sleep Labels')
plt.legend(['Good Sleep', 'Poor Sleep'])


# #### Cluster Purity
# 
# Finding cluster purity based on the sleep labels

# In[79]:


# Clustering Purity is defined by ratio of dominant class of sleep label instance in the cluster 
# to total number of instances in the cluster
for master_cluster_num in range(len(centres)):
    cluster_sleep_labels = final_sleep_labels[cluster_assignments == master_cluster_num]
    pos_sleep_label_purity = sum(cluster_sleep_labels) / cluster_sleep_labels.shape[0]
    print(f'Cluster Number: {master_cluster_num}, Purity:', max(pos_sleep_label_purity, 1 - pos_sleep_label_purity))


# #### Sub-Clustering on Activity Data

# In[80]:


sub_clusters = activity_percentage_clusterer(KL_Kmeans(num_clusters=12), cluster_assignments, activity_percentages)


# In[81]:


# Sanity Check for the number of points in each cluster
print(np.unique(cluster_assignments, return_counts=True))
for sub_cluster in sub_clusters:
    print(sub_cluster.shape)


# ##### Cluster Purity in each subcluster

# In[82]:


# Clustering Purity is defined by ratio of dominant class of sleep label instance in the cluster
# to total number of instances in the cluster
for index, sub_cluster in enumerate(sub_clusters):
    print('Master Cluster:', index+1)
    cluster_sleep_labels = final_sleep_labels[(cluster_assignments == index)]
    for sub_cluster_assignment in range(num_activity_clusters):
        sub_cluster_sleep_labels = cluster_sleep_labels[(sub_cluster==sub_cluster_assignment)]
        try:
            pos_sleep_label_purity = sum(sub_cluster_sleep_labels) / sub_cluster_sleep_labels.shape[0]
            print(f'Sub Cluster Number: {sub_cluster_assignment}, Purity:', max(pos_sleep_label_purity, 1 - pos_sleep_label_purity))
            print(f'Sub Cluster Number: {sub_cluster_assignment}, Good Sleep %:', pos_sleep_label_purity)
        except:
            print(f'Sub Cluster Number: {sub_cluster_assignment}, No Points assigned')


# In[83]:


sleep_recipes = get_good_sleep_recipes(cluster_assignments, sub_clusters, activity_percentages, final_sleep_labels, good_sleep_ratio=1.)
sleep_recipes


# In[84]:


for i, sleep_recipe in enumerate(sleep_recipes):
    plt.figure(i)
    plt.bar(['Sedentary', 'Light', 'Moderate', 'Vigorous'], (sleep_recipe * 720 / 100))
    plt.ylabel('Minutes')
    plt.title('Activity Recipes for Sleep')


# In[ ]:




