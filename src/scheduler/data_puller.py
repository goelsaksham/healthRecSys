from data_preprocessor.get_user_data import *
from data_preprocessor.push_user_data import *
from last_sync_reader import *
from client_secrets.user_ids import all_users_info
from datetime import timedelta
import schedule
import time


def collect_user_data(user_name, sync_file_obj, date_format='%Y-%m-%d'):
	CLIENT_ID, CLIENT_SECRET, server = instantiate_user(user_name)
	ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
	auth_client = get_auth_client(CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)
	user_id = get_fitbit_user_id(get_user_information(server))

	#TODO: Change functionality
	date_last_sync_str = sync_file_obj.get_user_last_sync_time(user_id)
	date_last_sync = datetime.strptime(date_last_sync_str, date_format)
	todays_date = datetime.today()
	todays_date_str = todays_date.strftime(date_format)

	number_of_days_since_sync = todays_date - date_last_sync
	# TODO: Need to check if able to get data for that particular date???
	for i in range(number_of_days_since_sync.days + 1):
		current_date = date_last_sync + timedelta(days=i)
		print(current_date)
		current_date_str = current_date.strftime(date_format)
		try:
			# insert_sleep_cycles_data(current_date_str, auth_client, user_id)
			# insert_activity_intraday_data(current_date_str, auth_client, user_id)
			insert_sleep_raw_data(current_date_str, auth_client, user_id)
			# insert_heart_rate_intraday_data(current_date_str, auth_client, user_id)
			insert_activity_summary_data(current_date_str, auth_client, user_id)
			insert_sleep_intraday_data(current_date_str, auth_client, user_id)
			insert_sleep_summary_data(current_date_str, auth_client, user_id)
		except:
			pass
	# TODO: Do stuff with the current date for the current user
	# TODO: Add if condition to write date upto which the data is present and not the current date
	sync_file_obj.update_user_last_sync_time(user_id, todays_date_str, date_format)


def collect_all_users_data(user_info, sync_file_obj):
	print('Pulling Data for all users at Timestamp:', datetime.now())
	for user_name in user_info:
		print('Hello World')
		# Uncomment this for testing the correction of script
		collect_user_data(user_name, sync_file_obj)


def main():
	user_info = all_users_info
	sync_file_obj = CSVLastSync('../../data/sync_times/sync_times.csv')
	collect_all_users_data(user_info, sync_file_obj)


if __name__ == '__main__':
	schedule.every(1).second.do(main)
	# Uncomment this for testing the correction of script
	# schedule.every().day.at("23:30").do(main)

	while True:
		schedule.run_pending()
		time.sleep(1)
