"""
This file contains several utility functions to help preprocess sleep_stages_ts data
"""
from data_preprocessor.get_user_data import *
from utils.general_utils import *
from datetime import datetime


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


def main():
	USER_ID, CLIENT_SECRET, server = instantiate_server('Saksham')
	ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
	auth_client = get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)
	user_id = get_fitbit_user_id(get_user_information(server))
	# import pickle
	# pickle.dump(get_sleep_data(auth_client, '2019-03-13'), open('sleep_file.pickle', 'wb'))
	# print(get_sleep_data(auth_client, '2019-03-13'))
	# print('\n\n\n')
	# print(get_sleep_stages_data(get_sleep_data(auth_client, '2019-03-13'), user_id))
	# print('\n\n\n')
	# print(get_all_sleeps_summary(get_sleep_data(auth_client, '2019-03-13'), user_id))
	# print('\n\n\n')
	# print(get_entire_sleep_summary(get_sleep_data(auth_client, '2019-03-13'), user_id))
	# print('\n\n\n')
	# print(get_sleep_)
	# print(get_all_sleep_levels_total_time_hrs_mins(get_sleep_stages_data(get_sleep_data(auth_client, '2019-03-13'),
	#                                                                  user_id)))
	start_date_str = '2019-02-25'
	# print(get_sleep_stages_summary(get_sleep_data(auth_client, start_date_str)))
	# print(get_all_sleep_levels_total_time_mins(get_sleep_stages_data(get_sleep_data(auth_client, start_date_str),
	#                                                                  user_id)))
	start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
	ls = []
	for num_days in range(20):
		current_date = start_date + timedelta(days=num_days)
		current_date_str = current_date.strftime('%Y-%m-%d')
		try:
			ls.append(get_time_asleep_to_time_in_bed_ratio(get_sleep_data(auth_client, current_date_str), user_id))
		# print(get_time_asleep_to_time_in_bed_ratio(get_sleep_data(auth_client, current_date_str), user_id))
		except:
			pass
		# print(get_fitbit_sleep_efficiency(get_sleep_data(auth_client, current_date_str), user_id))
	print(ls)

if __name__ == '__main__':
    main()
