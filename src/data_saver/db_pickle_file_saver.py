from data_preprocessor.get_user_data import *
from data_preprocessor.push_user_data import *
import pickle
from utils.general_utils import *
from utils.directory_utils import *
from datetime import timedelta
from datetime import datetime
import time
import traceback


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
			pickle.dump(sleep_data, file_writer)

		sleep_summary_data = get_entire_sleep_summary(sleep_data,user_id)
		record = [
			"'" + sleep_summary_data["user_id"] + "'",
			"'" + date_str + "'",
			sleep_summary_data["total_minutes_asleep"],
			sleep_summary_data["total_sleep_records"],
			sleep_summary_data["total_time_in_bed"]
		]
		run_insert_query(database_connection, "sleep_summary", record)

		insert_sleep_cycles_data(sleep_data,user_id)

		raw_sleep_data = get_data_for_dump(sleep_data,'sleep',user_id,date_str)#get_sleep_data_for_dump(auth_client, user_id, date_str)
		record = [
			"'" + raw_sleep_data["user_id"] + "'",
			"'" + (str(raw_sleep_data["data"])).replace(r"'", r'"').replace("True", '"true"').replace("False", '"false"') + "'",
			"timestamp '" + raw_sleep_data["pull_time"] + "'",
			"'" + raw_sleep_data["date"] + "'",
			"'" + raw_sleep_data["data_type"] + "'"
		]
		run_insert_query(database_connection, "raw_device_data", record)

		sleep_intraday_data = get_sleep_stages_data(sleep_data, user_id)
		for value in sleep_intraday_data:
			record = [
				f"'{user_id}'",
				f"timestamp '{value['timestamp']}'",
				value["sec"],
				value["level"]
			]
			run_insert_query(database_connection, "sleep_intraday_data", record)

def save_user_calories_data(user_id, auth_client, date_str, attr_directory_path):
	if not check_output_directory(attr_directory_path):
		return
	file_name = f'{date_str}.pickle'
	file_path = construct_path(attr_directory_path, file_name)
	calories_data = get_calories_data(auth_client, date_str)
	if calories_data:
		print(calories_data)
		activity_values = get_calories_intraday(calories_data, user_id)

		with open(file_path, 'wb') as file_writer:
			pickle.dump(activity_values, file_writer)

		for value in activity_values:
			date_time = "timestamp '" + value["timestamp"] + "'"
			record = [
				"'" + value["user_id"] + "'",
				date_time,
				value["value"],
				value["level"],
				value["mets"]
			]
			run_insert_query(database_connection, "activity_intraday_data", record)

		activity_summary = get_calories_summary(calories_data, user_id)
		record = [
			"'" + activity_summary['user_id'] + "'",
			activity_summary['value'],
			"'" + activity_summary['date'] + "'",
		]
		run_insert_query(database_connection, "activity_type_summary", record)

		s_data = get_data_for_dump(calories_data, 'activity', user_id, date_str)
		record = [
			"'" + s_data["user_id"] + "'",
			"'" + (str(s_data["data"])).replace(r"'", r'"') + "'",
			"timestamp '" + s_data["pull_time"] + "'",
			"'" + s_data["date"] + "'",
			"'" + s_data["data_type"] + "'"
		]
		run_insert_query(database_connection, "raw_device_data", record)

def save_user_heart_data(user_id, auth_client, date_str, attr_directory_path):
	if not check_output_directory(attr_directory_path):
		return
	file_name = f'{date_str}.pickle'
	file_path = construct_path(attr_directory_path, file_name)
	heart_data = get_heart_data(auth_client, date_str)
	if heart_data:
		heart_rate_data = get_heart_intraday(heart_data, user_id)

		with open(file_path, 'wb') as file_writer:
			pickle.dump(heart_rate_data, file_writer)

		for data in heart_rate_data:
			record = [
				"'" + user_id + "'",
				"timestamp '" + data["timestamp"] + "'",
				data["value"]
			]
			run_insert_query(database_connection, "heart_rate_intraday_data", record)

def save_user_steps_data(user_id, auth_client, date_str, attr_directory_path):
	if not check_output_directory(attr_directory_path):
		return
	file_name = f'{date_str}.pickle'
	file_path = construct_path(attr_directory_path, file_name)
	steps_data = get_steps_data(auth_client, date_str)
	print(date_str, steps_data)
	if steps_data:
		steps_intraday_data = get_steps_intraday(steps_data, user_id)

		with open(file_path, 'wb') as file_writer:
			pickle.dump(steps_intraday_data, file_writer)

		s_data = get_data_for_dump(steps_data,'steps',user_id,date_str)
		record = [
			"'" + s_data["user_id"] + "'",
			"'" + (str(s_data["data"])).replace(r"'", r'"') + "'",
			"timestamp '" + s_data["pull_time"] + "'",
			"'" + s_data["date"] + "'",
			"'" + s_data["data_type"] + "'"
		]
		run_insert_query(database_connection, "raw_device_data", record)

		for data in steps_intraday_data:
			record = [
				"'" + user_id + "'",
				"timestamp '" + data["timestamp"] + "'",
				data["value"]
			]
			run_insert_query(database_connection, "steps_intraday_data", record)


def date_iterator(user_id, auth_client, user_directory, start_date, end_date, date_format='%Y-%m-%d'):
	if not check_output_directory(user_directory):
		print("error")
		return
	current_date = None
	for num_days in range((end_date - start_date).days + 1):
		current_date = start_date + timedelta(days=num_days)
		current_date_str = current_date.strftime(date_format)
		sleep_directory, activity_directory, calories_directory, heart_directory, steps_directory = \
			get_all_attributes_path(user_directory)
		save_user_heart_data(user_id, auth_client, current_date_str, heart_directory)
		save_user_steps_data(user_id, auth_client, current_date_str, steps_directory)
		save_user_sleep_data(user_id, auth_client, current_date_str, sleep_directory)
		save_user_calories_data(user_id, auth_client, current_date_str, calories_directory)
		if num_days != 0 and num_days % 30 == 0:
			print("Going to sleep for 100 seconds to avoid delays")
			time.sleep(100)
	return current_date


def save_user_data_in_pickle_files(user_name='Meghna', data_dump_directory_path=f'../../data/data_files/Meghna'):
	USER_ID, CLIENT_SECRET, server = instantiate_server(user_name)
	ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
	auth_client = get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)
	user_id = get_fitbit_user_id(get_user_information(server))
	break_loop = False
	start_date = datetime(2019, 1, 14)
	end_date = datetime.today() #datetime(2019, 1, 1)#
	print(start_date)

	while not break_loop:
		# Adding this loop along with the try except block to
		try:
			last_date = date_iterator(user_id, auth_client, data_dump_directory_path,
			              start_date, end_date)
			break_loop = (last_date - end_date).days >= -1
		except Exception as e:
			print(e)
			traceback.print_tb(e.__traceback__)
			# sleeping
			print('Sleeping so that no too many requests!')
			time.sleep(1800)
			USER_ID, CLIENT_SECRET, server = instantiate_server(user_name)
			ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
			auth_client = get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)
			user_id = get_fitbit_user_id(get_user_information(server))


def main():
	save_user_data_in_pickle_files()


if __name__ == '__main__':
	main()
