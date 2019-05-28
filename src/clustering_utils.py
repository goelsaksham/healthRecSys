"""
This file is responsible to perform clustering over the heart rate time series and activity and return the
corresponding results.
"""
import numpy as np
import random
from sklearn.metrics import silhouette_score
from utils.general_utils import get_all_clusters_sleep_purity


def get_pure_num_clusters(orig_data, cluster_getter_func, number_of_cluster_range, sleep_labels,
                          measure='gini'):

	max_purity = -1
	best_num_clusters = 1
	for num_clusters in number_of_cluster_range:
		clusterer = cluster_getter_func(num_clusters)
		cluster_labels = clusterer.fit_predict(orig_data)
		purity_val = get_all_clusters_sleep_purity(cluster_labels, sleep_labels, measure=measure)
		if purity_val > max_purity:
			max_purity = purity_val
			best_num_clusters = num_clusters
	return best_num_clusters, max_purity


def get_purest_clustering_model(cluster_model_getter, data, sleep_labels,
                                cluster_range=range(2, 9),
                                purity_measure='gini'):
	best_number_clusters, max_purity = \
		get_pure_num_clusters(data, cluster_model_getter, cluster_range, sleep_labels,
		                      purity_measure)
	if best_number_clusters == list(cluster_range)[-1]:
		best_number_clusters, max_purity = \
			get_pure_num_clusters(data, cluster_model_getter,
			                      range(list(cluster_range)[-0], list(cluster_range)[-1] + 20),
			                      sleep_labels, purity_measure)
	return cluster_model_getter(best_number_clusters)


def get_best_num_clusters(orig_data, cluster_getter_func, number_of_cluster_range,
                          sil_score_distance_metric='euclidean'):
	max_sil_score = -1
	best_num_clusters = 1
	for num_clusters in number_of_cluster_range:
		clusterer = cluster_getter_func(num_clusters)
		cluster_labels = clusterer.fit_predict(orig_data)
		sil_score = silhouette_score(orig_data, cluster_labels, metric=sil_score_distance_metric)
		if sil_score > max_sil_score:
			max_sil_score = sil_score
			best_num_clusters = num_clusters
	return best_num_clusters, max_sil_score


def get_best_clustering_model(cluster_model_getter, data, cluster_range=range(2, 9),
                              sil_score_distance_metric='euclidean'):
	best_number_clusters, max_sil_score = \
		get_best_num_clusters(data, cluster_model_getter, cluster_range,
		                      sil_score_distance_metric)
	if best_number_clusters == list(cluster_range)[-1]:
		best_number_clusters, max_sil_score = \
			get_best_num_clusters(data, cluster_model_getter,
			                      range(list(cluster_range)[-0], list(cluster_range)[-1] + 5),
			                      sil_score_distance_metric)
	return cluster_model_getter(best_number_clusters)


def activity_percentage_clusterer(clusterer, heart_data_cluster_assignments, activity_percentage_data):
	sub_cluster_assignments = []
	for cluster_num in np.unique(heart_data_cluster_assignments):
		heart_cluster_data = activity_percentage_data[heart_data_cluster_assignments == cluster_num]
		# random.seed(10)
		# np.random.seed(10)
		sub_cluster_assignment = clusterer.fit_predict(heart_cluster_data)
		sub_cluster_assignments.append(sub_cluster_assignment)
	return sub_cluster_assignments


def get_good_sleep_ratio(sleep_labels):
	good_sleep_num = np.sum(sleep_labels)
	bad_sleep_num = np.sum(~sleep_labels)
	if bad_sleep_num == 0:
		return np.inf
	else:
		return good_sleep_num / bad_sleep_num


def get_good_sleep_recipes(master_cluster_assignments_array, sub_cluster_assignments_array_list,
                           activity_percentage_array, sleep_label_array, good_sleep_ratio = 2.):
	sleep_recipes = []
	for master_cluster_index, sub_cluster_assignments_array in enumerate(sub_cluster_assignments_array_list):
		master_cluster_activity = activity_percentage_array[master_cluster_assignments_array == master_cluster_index]
		master_cluster_sleep_labels = sleep_label_array[master_cluster_assignments_array == master_cluster_index]
		for sub_cluster_num in np.unique(sub_cluster_assignments_array):
			sub_cluster_activity = master_cluster_activity[sub_cluster_assignments_array == sub_cluster_num]
			sub_cluster_sleep_labels = master_cluster_sleep_labels[sub_cluster_assignments_array ==
			                                                            sub_cluster_num]
			if get_good_sleep_ratio(sub_cluster_sleep_labels) >= good_sleep_ratio:
				print(f'Cluster: {master_cluster_index}, Sub Cluster: {sub_cluster_num}, '
				      f'Good Ratio: {get_good_sleep_ratio(sub_cluster_sleep_labels)}')
				sleep_recipes.append(np.nanmean(sub_cluster_activity[sub_cluster_sleep_labels], axis=0))
	return np.array(sleep_recipes)
