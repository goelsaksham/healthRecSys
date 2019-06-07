from data_preprocessor.get_user_data import *
# from data_preprocessor.push_user_data import *
from last_sync_reader import *
from client_secrets.user_ids import *
from datetime import timedelta
import schedule
import time
from token_reader import UserTokenGenerator


def collect_user_data(user_name, sync_file_obj, token_obj: UserTokenGenerator, date_format='%Y-%m-%d'):
	user_id, ACCESS_TOKEN, REFRESH_TOKEN = token_obj.get_tokens_using_user_names(user_name)
	auth_client = get_auth_client(get_fitbit_client_id(), get_fitbit_client_secret(), ACCESS_TOKEN, REFRESH_TOKEN)
	print('User Id:', user_id, 'User Name:', user_name, 'Access Token:', ACCESS_TOKEN, 'Refresh Token:',
	      REFRESH_TOKEN)
	print(get_heart_intraday(get_heart_data(auth_client, '2019-02-27'), user_id))

	# TODO: Change functionality
	# date_last_sync_str = sync_file_obj.get_user_last_sync_time(user_id)
	# date_last_sync = datetime.strptime(date_last_sync_str, date_format)
	# todays_date = datetime.today()
	# todays_date_str = todays_date.strftime(date_format)
	#
	# number_of_days_since_sync = todays_date - date_last_sync
	# # TODO: Need to check if able to get data for that particular date???
	# for i in range(number_of_days_since_sync.days + 1):
	# 	current_date = date_last_sync + timedelta(days=i)
	# 	print(current_date)
	# 	current_date_str = current_date.strftime(date_format)
	# 	try:
	# 		# insert_sleep_cycles_data(current_date_str, auth_client, user_id)
	# 		# insert_activity_intraday_data(current_date_str, auth_client, user_id)
	# 		insert_sleep_raw_data(current_date_str, auth_client, user_id)
	# 		# insert_heart_rate_intraday_data(current_date_str, auth_client, user_id)
	# 		insert_activity_summary_data(current_date_str, auth_client, user_id)
	# 		insert_sleep_intraday_data(current_date_str, auth_client, user_id)
	# 		insert_sleep_summary_data(current_date_str, auth_client, user_id)
	# 	except:
	# 		pass
	# # TODO: Do stuff with the current date for the current user
	# # TODO: Add if condition to write date upto which the data is present and not the current date
	# sync_file_obj.update_user_last_sync_time(user_id, todays_date_str, date_format)


def collect_all_users_data(user_info, sync_file_obj, token_obj):
	print('Pulling Data for all users at Timestamp:', datetime.now())
	for user_name in user_info:
		print(f'UserName: {user_name}')
		# Uncomment this for testing the correction of script
		collect_user_data(user_name, sync_file_obj, token_obj)


def main():
	user_info = all_users_info
	sync_file_obj = CSVLastSync('../../data/sync_times/sync_times.csv')
	token_obj = UserTokenGenerator(f'../../data/user_tokens/tokens.csv')
	collect_all_users_data(user_info, sync_file_obj, token_obj)


if __name__ == '__main__':
	# main()
	# schedule.every(100).minutes.do(main)
	# # Uncomment this for testing the correction of script
	# # schedule.every().day.at("23:30").do(main)
	#
	# while True:
	# 	schedule.run_pending()
	# 	time.sleep(1)
	sync_file_obj = CSVLastSync('../../data/sync_times/sync_times.csv')
	token_obj = UserTokenGenerator(f'../../data/user_tokens/tokens.csv')
	collect_user_data('Saksham', sync_file_obj, token_obj)
