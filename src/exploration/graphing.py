from get_user_data import *
import matplotlib.pyplot as plt
import numpy as np


def get_data_from_dictionary_list(list_of_dictionaries, dictionary_key):
    return [dictionary[dictionary_key] for dictionary in list_of_dictionaries]


def plot_activity_vs_attribute(axis, activity_list, attribute_list):
    axis.plot(attribute_list)
    twin_axis = axis.twinx()
    twin_axis.plot(activity_list, 'r')


def plot_activity_vs_all(date):
    USER_ID, CLIENT_SECRET, server = instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
    auth_client = get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    user_id = get_fitbit_user_id(get_user_information(server))

    calories_intraday = get_calories_intraday(get_calories_data(auth_client, date), user_id)
    heart_intraday = get_heart_intraday(get_heart_data(auth_client, date), user_id)

    mets_list = get_data_from_dictionary_list(calories_intraday, 'mets')
    cals_list = get_data_from_dictionary_list(calories_intraday, 'value')
    heart_list = get_data_from_dictionary_list(heart_intraday, 'value')
    activity = get_data_from_dictionary_list(calories_intraday, 'level')

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    plot_activity_vs_attribute(ax[0], activity, mets_list)
    plot_activity_vs_attribute(ax[1], activity, cals_list)
    plot_activity_vs_attribute(ax[2], activity, heart_list)
    plt.show()


def plot_all_on_one(date):
    USER_ID, CLIENT_SECRET, server = instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
    auth_client = get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    user_id = get_fitbit_user_id(get_user_information(server))

    calories_intraday = get_calories_intraday(get_calories_data(auth_client, date), user_id)
    heart_intraday = get_heart_intraday(get_heart_data(auth_client, date), user_id)

    mets_list = get_data_from_dictionary_list(calories_intraday, 'mets')
    cals_list = get_data_from_dictionary_list(calories_intraday, 'value')
    calories_arr = np.array(cals_list)
    norm_calroies_arr = (calories_arr - np.min(calories_arr))/(np.max(calories_arr) - np.min(calories_arr))
    heart_list = get_data_from_dictionary_list(heart_intraday, 'value')
    heart_arr = np.array(heart_list)
    norm_health_arr = (heart_arr - np.min(heart_arr))/(np.max(heart_arr) - np.min(heart_arr))
    activity = get_data_from_dictionary_list(calories_intraday, 'level')

    fig, ax = plt.subplots(1, 1, figsize=(10, 15))
    twin_axis = ax.twinx()
    ax.plot(norm_calroies_arr)
    ax.plot(norm_health_arr, 'g')
    twin_axis.plot(activity, 'r')
    plt.show()


def main():
    plot_activity_vs_all('2018-11-28')
    plot_all_on_one('2018-11-28')


if __name__ == '__main__':
    main()
