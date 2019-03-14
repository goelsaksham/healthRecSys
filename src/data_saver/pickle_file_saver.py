from data_preprocessor.get_user_data import *
import pickle
from utils.general_utils import *
from utils.directory_utils import *
from datetime import timedelta
from datetime import datetime
import time


def get_all_attributes_path(user_directory):
	return construct_path(user_directory, 'sleep'), construct_path(user_directory, 'activity'), \
	       construct_path(user_directory, 'calories'), construct_path(user_directory, 'heart'), \
	       construct_path(user_directory, 'steps')


def save_user_sleep_data(user_id, auth_client, date_str, attr_directory_path):
	if not check_output_directory(attr_directory_path):
		return
	file_name = f'{date_str}.pickle'
	file_path = construct_path(attr_directory_path, file_name)
	sleep_data = get_sleep_data(auth_client, date_str)
	if sleep_data:
		with open(file_path, 'wb') as file_writer:
			pickle.dump(get_all_sleeps_summary(sleep_data, user_id), file_writer)


def save_user_calories_data(user_id, auth_client, date_str, attr_directory_path):
	if not check_output_directory(attr_directory_path):
		return
	file_name = f'{date_str}.pickle'
	file_path = construct_path(attr_directory_path, file_name)
	calories_data = get_calories_data(auth_client, date_str)
	if calories_data:
		with open(file_path, 'wb') as file_writer:
			pickle.dump(get_calories_intraday(calories_data, user_id), file_writer)


def save_user_heart_data(user_id, auth_client, date_str, attr_directory_path):
	if not check_output_directory(attr_directory_path):
		return
	file_name = f'{date_str}.pickle'
	file_path = construct_path(attr_directory_path, file_name)
	heart_data = get_heart_data(auth_client, date_str)
	if heart_data:
		with open(file_path, 'wb') as file_writer:
			pickle.dump(get_heart_intraday(heart_data, user_id), file_writer)


def save_user_steps_data(user_id, auth_client, date_str, attr_directory_path):
	if not check_output_directory(attr_directory_path):
		return
	file_name = f'{date_str}.pickle'
	file_path = construct_path(attr_directory_path, file_name)
	steps_data = get_steps_data(auth_client, date_str)
	if steps_data:
		with open(file_path, 'wb') as file_writer:
			pickle.dump(get_steps_intraday(steps_data, user_id), file_writer)


def date_iterator(user_id, auth_client, user_directory, start_date, end_date, date_format='%Y-%m-%d'):
	if not check_output_directory(user_directory):
		return
	for num_days in range((end_date - start_date).days + 1):
		current_date = start_date + timedelta(days=num_days)
		current_date_str = current_date.strftime(date_format)
		sleep_directory, activity_directory, calories_directory, heart_directory, steps_directory = \
			get_all_attributes_path(user_directory)
		save_user_calories_data(user_id, auth_client, current_date_str, calories_directory)
		save_user_heart_data(user_id, auth_client, current_date_str, heart_directory)
		save_user_sleep_data(user_id, auth_client, current_date_str, sleep_directory)
		save_user_steps_data(user_id, auth_client, current_date_str, steps_directory)
		if num_days % 10 == 0:
			print("Going to sleep for 100 seconds")
			time.sleep(100)


def save_user_data_in_pickle_files(user_name, data_dump_directory_path):
	USER_ID, CLIENT_SECRET, server = instantiate_user()
	ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
	auth_client = get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)
	user_id = get_fitbit_user_id(get_user_information(server))
	date_iterator(user_id, auth_client, f'../../data/data_files/Saksham',
	              datetime.today() - timedelta(days=41), datetime.today())


def main():
	#time.sleep(3600)
	save_user_data_in_pickle_files('', '')

if __name__ == '__main__':
	main()
