"""
This file is responsible to perform clustering over the heart rate time series and activity and return the
corresponding results.
"""
import numpy as np
import random


def activity_percentage_clusterer(clusterer, heart_data, heart_data_cluster_assignments, activity_percentage_data):
	sub_cluster_assignments = []
	for cluster_num in np.unique(heart_data_cluster_assignments):
		heart_cluster_data = activity_percentage_data[heart_data_cluster_assignments == cluster_num]
		random.seed(10)
		np.random.seed(10)
		sub_cluster_assignment = clusterer.fit_predict(heart_cluster_data)
		sub_cluster_assignments.append(sub_cluster_assignment)
	return sub_cluster_assignments
