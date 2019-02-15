"""
This file is responsible to define various classes to get the datetime format of the user for which we want to get the
last sync time so that we can get data for the particular user. The file defines various classes and then an object
of a class can be used to get the data for a particular user using the User ID.
"""

import pandas as pd
import os
from datetime import datetime


class CSV_Last_Sync:
	def __init__(self, csv_file_path='../../data/sync_times/sync_times.csv', date_formatting='%Y-%M-%d'):
		if not os.path.isfile(csv_file_path):
			raise FileNotFoundError(f'Invalid csv file path provided: {csv_file_path}')
		self.__csv_file_path = csv_file_path
		self.__date_formatting = date_formatting

	def get_csv_file_path(self):
		return self.__csv_file_path

	def set_csv_file_path(self, new_csv_file_path):
		self.__csv_file_path = new_csv_file_path

	def get_date_formatting(self):
		return self.__date_formatting

	def set_date_formatting(self, new_date_formatting):
		self.__date_formatting = new_date_formatting

	def get_user_last_sync_time(self, user_id):
		sync_time_df = pd.read_csv(self.get_csv_file_path(), index_col='user_id')
		if user_id in sync_time_df.index:
			if pd.isna(sync_time_df.loc[user_id]['last_sync_time']):
				return datetime.today().strftime(self.get_date_formatting())
			else:
				return sync_time_df.loc[user_id]['last_sync_time']
		else:
			return None

	def update_user_last_sync_time(self, user_id, date, date_formatting='%Y-%M-%d'):
		sync_time_df = pd.read_csv(self.get_csv_file_path(), index_col='user_id')
		date_str = datetime.strptime(date, date_formatting).strftime(self.get_date_formatting())
		if user_id in sync_time_df.index:
			sync_time_df.loc[user_id]['last_sync_time'] = date_str
		else:
			sync_time_df.loc[user_id] = {'last_sync_time': date_str}
		sync_time_df.to_csv(self.get_csv_file_path())


# TODO: Complete the class for Database reading
# class DB_Last_Sync:


if __name__ == '__main__':
	obj = CSV_Last_Sync()
	print(obj.get_user_last_sync_time('6WQRF5'))
	obj.update_user_last_sync_time('6WQRF5', '2018:09:09', '%Y:%M:%d')
	obj.update_user_last_sync_time('6WQRF6', '2018:09:09', '%Y:%M:%d')
	obj.update_user_last_sync_time('6WQRF7', '2018:09:09', '%Y:%M:%d')
	obj.update_user_last_sync_time('6WQRF8', '2018:09:09', '%Y:%M:%d')