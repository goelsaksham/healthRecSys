from data_preprocessor.get_user_data import *
import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt


def main():
	USER_ID, CLIENT_SECRET, server = instantiate_user()
	ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
	auth_client = get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

	user_id = get_fitbit_user_id(get_user_information(server))
	date_str = '2019-02-25'
	# print(get_sleep_data_for_dump(auth_client, user_id, '2018-11-10'))
	# print(get_all_sleeps_summary(get_sleep_data(auth_client, '2018-11-15'), user_id))
	# print(get_calories_data(auth_client, '2018-11-10'))
	# print(get_calories_intraday(get_calories_data(auth_client, '2018-11-10'), user_id))
	# print(get_heart_data(auth_client, '2018-11-10'))
	# print(get_heart_summary(get_heart_data(auth_client, '2018-11-10'), user_id))
	# print(get_heart_intraday(get_heart_data(auth_client, '2019-02-27'), user_id))
	# print(get_heart_data(auth_client, '2018-02-27'))
	heart_data = get_heart_intraday(get_heart_data(auth_client, date_str), user_id)
	# print(heart_data)
	calories_data = get_calories_intraday(get_calories_data(auth_client, date_str), user_id)
	# print(calories_data)
	steps_data = get_steps_intraday(get_steps_data(auth_client, date_str), user_id)
	# print(steps_data)


	heart_beat_vals = np.array([minute_heart['value'] for minute_heart in heart_data])
	heart_beat_vals = heart_beat_vals - np.mean(heart_beat_vals)
	heart_beat_vals = heart_beat_vals / np.std(heart_beat_vals)
	calories_vals = np.array([minute_calories['value'] for minute_calories in calories_data])
	calories_vals = calories_vals - np.mean(calories_vals)
	calories_vals = calories_vals / np.std(calories_vals)
	activity_vals = np.array([minute_calories['level'] for minute_calories in calories_data])
	activity_vals = activity_vals - np.mean(activity_vals)
	activity_vals = activity_vals / np.std(activity_vals)
	steps_vals = np.array([minute_steps['value'] for minute_steps in steps_data])
	steps_vals = steps_vals - np.mean(steps_vals)
	steps_vals = steps_vals / np.std(steps_vals)


	all_vals = np.array(list(zip(heart_beat_vals, calories_vals, steps_vals))).reshape(-1, 3)
	print(all_vals.shape)
	heart_beat_vals = np.array(heart_beat_vals)
	calories_vals = np.array(calories_vals)
	activity_vals = np.array(activity_vals)
	steps_vals = np.array(steps_vals)

	algo = rpt.Pelt(model="l2", min_size=10).fit(heart_beat_vals)
	result = algo.predict(pen=10)
	rpt.display(heart_beat_vals, result)
	plt.gcf().axes[0].set_title('Heart Beat')
	plt.savefig('../data/plots/changepoint/heart.png')
	# plt.show()

	algo = rpt.Pelt(model="l2", min_size=10).fit(calories_vals)
	result = algo.predict(pen=10)
	rpt.display(calories_vals, result)
	plt.gcf().axes[0].set_title('Calories')
	plt.savefig('../data/plots/changepoint/calories.png')
	# plt.show()

	algo = rpt.Pelt(model="l2", min_size=10).fit(steps_vals)
	result = algo.predict(pen=10)
	rpt.display(steps_vals, result)
	plt.gcf().axes[0].set_title('Steps')
	plt.savefig('../data/plots/changepoint/steps.png')
	# plt.show()

	algo = rpt.Pelt(model="l2", min_size=10).fit(activity_vals)
	result = algo.predict(pen=10)
	rpt.display(activity_vals, result)
	plt.gcf().axes[0].set_title('Activity')
	plt.savefig('../data/plots/changepoint/activity.png')
	# plt.show()

	algo = rpt.Pelt(model="l2", min_size=10).fit(all_vals)
	result = algo.predict(pen=10)
	rpt.display(all_vals, result)
	plt.gcf().axes[0].set_title('Heart Rate')
	plt.gcf().axes[1].set_title('Calories')
	plt.gcf().axes[2].set_title('Steps')
	plt.savefig('../data/plots/changepoint/all.png')
	# plt.show()


if __name__ == '__main__':
	main()
