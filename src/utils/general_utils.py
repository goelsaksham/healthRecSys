"""
This file is responsible to hold various general utility for the whole project
"""
import numpy as np
from datetime import datetime
from datetime import timedelta
from datetime import time


def get_nan_numpy_array(arr_shape):
	ret_arr = np.zeros(arr_shape)
	ret_arr[:] = np.nan
	return ret_arr


def get_data_date_timestamp(time_val_map):
	if time_val_map:
		return datetime.combine(list(time_val_map.keys())[0].date(), time(0, 0))
	return datetime.today()


def get_attribute_val_list(data_dictionary_list: list, attribute):
	return [dictionary[attribute] for dictionary in data_dictionary_list]


def get_attribute_val_array(data_dictionary_list: list, attribute):
	return np.array(get_attribute_val_list(data_dictionary_list, attribute))


def get_time_to_val_mapping(data_dictionary_list: list, attribute, timestamp_key='timestamp',
                            time_stamp_formatting='%Y-%m-%d %H:%M:%S'):
	return {datetime.strptime(dictionary[timestamp_key], time_stamp_formatting): dictionary[attribute]
	        for dictionary in data_dictionary_list}


def get_attribute_val_array_by_minute(data_dictionary_list: list, attribute):
	NUM_MINUTES_IN_DAY = 24 * 60
	ret_arr = get_nan_numpy_array((NUM_MINUTES_IN_DAY,))

	time_val_map = get_time_to_val_mapping(data_dictionary_list, attribute)
	data_date_timestamp = get_data_date_timestamp(time_val_map)
	for minute_delta in range(NUM_MINUTES_IN_DAY):
		add_minutes = timedelta(minutes=minute_delta)
		current_date = data_date_timestamp + add_minutes
		if current_date in time_val_map:
			ret_arr[minute_delta] = time_val_map[current_date]
	return ret_arr
