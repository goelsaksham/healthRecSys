"""
This file contains several utility functions to help preprocess sleep_stages_ts data
"""

from data_preprocessor.get_user_data import get_all_sleeps_summary, get_entire_sleep_summary


def get_sleep_level_total_time_secs(sleep_stages_data, sleep_level):
	return sum([data_dictionary['sec'] for data_dictionary in sleep_stages_data if data_dictionary['level'] == sleep_level])


def get_all_sleep_levels_total_time_secs(sleep_stages_data):
	return {f'level_{sleep_level}': get_sleep_level_total_time_secs(sleep_stages_data, sleep_level) for sleep_level in range(4)}


def get_all_sleep_levels_total_time_mins(sleep_stages_data):
	return {key: value/60 for key, value in get_all_sleep_levels_total_time_secs(sleep_stages_data).items()}


def get_all_sleep_levels_total_time_hrs(sleep_stages_data):
	return {key: value/60 for key, value in get_all_sleep_levels_total_time_mins(sleep_stages_data).items()}


def get_all_sleep_levels_total_time_hrs_mins(sleep_stages_data):
	return {key: (value//60, value % 60) for key, value in
	        get_all_sleep_levels_total_time_mins(sleep_stages_data).items()}


def get_fitbit_sleep_efficiency(sleep_data, user_id):
	all_sleeps_summary = get_all_sleeps_summary(sleep_data, user_id)
	for sleep_summary in all_sleeps_summary:
		if sleep_summary['main_sleep']:
			return sleep_summary['efficiency']
	return None


def get_time_asleep_to_time_in_bed_ratio(sleep_data, user_id):
	sleep_summary = get_entire_sleep_summary(sleep_data, user_id)
	return sleep_summary['total_minutes_asleep'] / sleep_summary['total_time_in_bed']