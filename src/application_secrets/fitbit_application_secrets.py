"""
This script contains the fitbit application client secrets which we use to access the fitbit data. These application
secrets should not be shared with anyone else
"""

all_fitbit_app_secrets = {
    'Saksham': {
        #'Client ID': '22DJZD',
        #'Client Secret': '5e6d1b8a6f9c43c097a1299458b66b06'
        'Client ID': '22D8XW',
        'Client Secret': '39823c504eeb610fed6be977d3806cee'
    },
    'Abhiraj': {
        'Client ID': '22DJV5',
        'Client Secret': 'ed40ec900fd38110e83767b7a5206008'
        # 'Client ID': '22D8XW',
        # 'Client Secret': '39823c504eeb610fed6be977d3806cee'
    },
    'Meghna': {
        'Client ID': '227ZN2',
        'Client Secret': '81170691ca6e7094c9c64b7e473351f6'
        # 'Client ID': '22D8XW',
        # 'Client Secret': '39823c504eeb610fed6be977d3806cee'
    },
    'Prof': {
        'Client ID': '22DDKS',
        'Client Secret': 'de3b99541ee111cf4133a450a80f4038'
        # 'Client ID': '22D8XW',
        # 'Client Secret': '39823c504eeb610fed6be977d3806cee'
    }
}


def get_user_fitbit_secrets(user_name):
    if user_name in all_fitbit_app_secrets.keys():
        return all_fitbit_app_secrets[user_name]
    else:
        raise KeyError(f'Inavlid key when accessing user fitbit secrets. Given: {user_name}. Allowed: {all_fitbit_app_secrets.keys()}')


def get_fitbit_client_id(user_name):
    return get_user_fitbit_secrets(user_name)['Client ID']


def get_fitbit_client_secret(user_name):
    return get_user_fitbit_secrets(user_name)['Client Secret']
