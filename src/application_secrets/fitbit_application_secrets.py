"""
This script contains the fitbit application client secrets which we use to access the fitbit data. These application
secrets should not be shared with anyone else
"""

fitbit_app_secrets = {
    'Client ID': '22D8XW',
    'Client Secret': 'fe88c75924d8aa3348e670cd3670ba68'
}


def get_fitbit_client_id():
    return fitbit_app_secrets['Client ID']


def get_fitbit_client_secret():
    return fitbit_app_secrets['Client Secret']
