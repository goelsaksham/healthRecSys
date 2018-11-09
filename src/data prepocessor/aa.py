import fitbit
import gather_keys_oauth2 as Oauth2
import numpy as np
import pandas as pd
import datetime
from application_secrets.fitbit_application_secrets import *

""" define functions """


def loadkeys(filename):
    """"
    load parrt's keys/tokens from CSV file with form
    consumer_key, consumer_secret, access_token, access_token_secret
    """
    with open(filename) as f:
        items = f.readline().strip().split(' ')
        return items


USER_ID, CLIENT_SECRET = get_fitbit_client_id(), get_fitbit_client_secret()

"""for obtaining Access-token and Refresh-token"""

server = Oauth2.OAuth2Server(USER_ID, CLIENT_SECRET)
server.browser_authorize()

user_id = server.fitbit.user_profile_get()['user']['encodedId']
print(user_id)

ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])

auth2_client = fitbit.Fitbit(USER_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN,
                             refresh_token=REFRESH_TOKEN)


def get_sleep_data():
    sleep_data = auth2_client.get_sleep(date=(datetime.datetime.now() - datetime.timedelta(days=2)))
    sleep_data = sleep_data['sleep'][0]['levels']['data']
    processed_sleep_data = []
    level_enum = {'wake': 0, 'light': 1, 'rem': 2, 'deep': 3}

    print(sleep_data)
    for sleep in sleep_data:
        processed_sleep_data.append({
            'user_id': user_id,
            'date': sleep['dateTime'].split('T')[0],
            'time': sleep['dateTime'].split('T')[-1],
            'sec': sleep['seconds'],
            'level': level_enum[sleep['level']]
        })

    return processed_sleep_data


def get_activity_intraday():
    calories_data = auth2_client.intraday_time_series(resource='calories')
    date = calories_data['activities-calories'][0]['dateTime']
    calories_data = calories_data['activities-calories-intraday']['dataset']

    processed_calories_data = []
    for calories in calories_data:
        processed_calories_data.append({
            'user_id': user_id,
            'date': date,
            'minute': calories['time'],
            'calories': calories['value'],
            'activity_level': calories['level'],
            'activity_met': calories['mets']
        })
    return processed_calories_data


print(get_activity_intraday())