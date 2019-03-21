"""
This file is responsible to hold various general utility for the whole project
"""
import numpy as np
from datetime import datetime
from datetime import timedelta
from datetime import time


def get_attribute_val_list(data_dictionary_list: list, attribute):
	return [dictionary[attribute] for dictionary in data_dictionary_list]


def get_attribute_val_array(data_dictionary_list: list, attribute):
	return np.array(get_attribute_val_list(data_dictionary_list, attribute))


def get_nan_numpy_array(arr_shape):
	"""
	Returns a numpy array of shape provided as the parameter with each value as nan
	:param arr_shape:
	:return:
	"""
	ret_arr = np.zeros(arr_shape)
	ret_arr[:] = np.nan
	return ret_arr


def get_data_date_timestamp(timestamp_val_map):
	"""
	Returns the 12:00 AM date timestamp of the date for which the data has been collected in the time map
	:param timestamp_val_map:
	:return:
	"""
	if timestamp_val_map:
		return datetime.combine(list(timestamp_val_map.keys())[0].date(), time(0, 0))
	return datetime.today()


def get_timestamp_to_val_mapping(data_dictionary_list: list, attribute, timestamp_key='timestamp',
                                 time_stamp_formatting='%Y-%m-%d %H:%M:%S'):
	"""
	Returns a dictionary mapping the datetime.timestamp object to the value extracted from the data dictionary list
	:param data_dictionary_list:
	:param attribute:
	:param timestamp_key:
	:param time_stamp_formatting:
	:return:
	"""
	return {datetime.strptime(dictionary[timestamp_key], time_stamp_formatting): dictionary[attribute]
	        for dictionary in data_dictionary_list}


def get_attribute_val_array_by_minute(data_dictionary_list: list, attribute):
	"""
	Returns a numpy array with values of the attributes corresponding to the index of the minute time in the array.
	The indexing works such that 12:00 AM maps to 0 and 11:59 maps to 1439th index
	:param data_dictionary_list:
	:param attribute:
	:return:
	"""
	NUM_MINUTES_IN_DAY = 24 * 60
	ret_arr = get_nan_numpy_array((NUM_MINUTES_IN_DAY,))

	time_val_map = get_timestamp_to_val_mapping(data_dictionary_list, attribute)
	data_date_timestamp = get_data_date_timestamp(time_val_map)
	for minute_delta in range(NUM_MINUTES_IN_DAY):
		add_minutes = timedelta(minutes=minute_delta)
		current_date = data_date_timestamp + add_minutes
		if current_date in time_val_map:
			ret_arr[minute_delta] = time_val_map[current_date]
	return ret_arr


def get_all_dates_numpy_array_minute_mean(all_dates_numpy_array):
	"""
	Find the mean of the values for each minute in a day as a 24 by 60 shape array.
	:param all_dates_numpy_array:
	:return:
	"""
	return np.nanmean(all_dates_numpy_array.reshape(-1, 24, 60), axis=0)


def get_all_dates_numpy_array_hour_mean(all_dates_numpy_array):
	return np.nanmean(get_all_dates_numpy_array_minute_mean(all_dates_numpy_array), axis=1)


def remove_nans_from_array(all_dates_numpy_array):
	"""
	This function removes nan values from the minute time series array.
	:param all_dates_numpy_array:
	:return:
	"""
	temp = all_dates_numpy_array.reshape(-1, 24, 60)
	minute_means = get_all_dates_numpy_array_minute_mean(all_dates_numpy_array)
	for day_data in temp:
		day_data[np.isnan(day_data)] = minute_means[np.isnan(day_data)]
	return temp.reshape(-1, 1440)


def reduce_time_series_dimension(time_series_array, time_window_length):
	NUM_MINUTES = 24 * 60
	return np.nanmean(time_series_array.reshape(-1, NUM_MINUTES // time_window_length, time_window_length), axis=2)


def activity_percentage_finder(activity_label_time_series_data):
	ratio_frac = 100 / (24 * 60)
	level_0_activity = np.sum(activity_label_time_series_data[:, :] == 0, axis=1) * ratio_frac
	level_1_activity = np.sum(activity_label_time_series_data[:, :] == 1, axis=1) * ratio_frac
	level_2_activity = np.sum(activity_label_time_series_data[:, :] == 2, axis=1) * ratio_frac
	level_3_activity = np.sum(activity_label_time_series_data[:, :] == 3, axis=1) * ratio_frac
	return np.round(np.array([level_0_activity, level_1_activity, level_2_activity, level_3_activity],
	                         dtype=np.float16).T,
	                decimals=2)