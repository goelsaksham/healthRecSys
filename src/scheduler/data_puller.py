from data_prepocessor.get_user_data import *
from data_prepocessor.push_user_data import *
from last_sync_reader import *
from client_secrets.user_ids import all_users_info


def collect_user_data(user_name):
    print('Hello world!')


def collect_all_users_data(user_id_list):
    for user_name in user_id_list:
        collect_user_data(user_name)
    print(f'Hello world!')


if __name__ == '__main__':
    collect_all_users_data([])



