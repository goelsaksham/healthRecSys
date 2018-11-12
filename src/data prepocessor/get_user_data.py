import fitbit
import gather_keys_oauth2 as Oauth2
import datetime
from application_secrets.fitbit_application_secrets import *


def instantiate_user():
    USER_ID, CLIENT_SECRET = get_fitbit_client_id(), get_fitbit_client_secret()
    server = Oauth2.OAuth2Server(USER_ID, CLIENT_SECRET)
    server.browser_authorize()
    return USER_ID, CLIENT_SECRET, server


def get_access_token(server):
    return server.fitbit.client.session.token['access_token']


def get_refresh_token(server):
    return server.fitbit.client.session.token['refresh_token']


def get_user_information(server):
    return server.fitbit.user_profile_get()


def get_fitbit_user_id(user_profile_information):
    return user_profile_information['user']['encodedId']


def get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN):
    return fitbit.Fitbit(USER_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)


def get_sleep_data(auth_client, date_str):
    return auth_client.get_sleep(date=date_str)


def get_all_sleeps_summary(sleep_data):
    entire_sleep_data = sleep_data['sleep']
    full_sleep_summary = []
    for current_sleep_data in entire_sleep_data:
        current_sleep_summary = dict()
        current_sleep_summary['date_of_sleep'] = current_sleep_data['dateOfSleep']
        current_sleep_summary['start_time'] = current_sleep_data['startTime']
        current_sleep_summary['time_in_bed'] = current_sleep_data['timeInBed']
        current_sleep_summary['duration'] = current_sleep_data['duration']
        current_sleep_summary['efficiency'] = current_sleep_data['efficiency']
        current_sleep_summary['end_time_stamp'] = current_sleep_data['endTime']
        current_sleep_summary['main_sleep'] = current_sleep_data['isMainSleep']
        current_sleep_summary['minutes_after_wakeup'] = current_sleep_data['minutesAfterWakeup']
        current_sleep_summary['minutes_asleep'] = current_sleep_data['minutesAsleep']
        current_sleep_summary['minutes_awake'] = current_sleep_data['minutesAwake']
        current_sleep_summary['minutes_to_fall_asleep'] = current_sleep_data['minutesToFallAsleep']
        full_sleep_summary.append(current_sleep_summary)
    return full_sleep_summary


def get_entire_sleep_summary(sleep_data):
    entire_summary = sleep_data['summary']
    entire_sleep_summary = {
        'deep_sleep_minutes': entire_summary['stages']['deep'],
        'light_sleep_minutes': entire_summary['stages']['light'],
        'rem_sleep_minutes': entire_summary['stages']['rem'],
        'wake_sleep_minutes': entire_summary['stages']['wake'],
        'total_minutes_asleep': entire_summary['totalMinutesAsleep'],
        'total_sleep_records': entire_summary['totalSleepRecords'],
        'total_time_in_bed': entire_summary['totalTimeInBed'],
    }
    return entire_sleep_summary


def get_sleep_enum():
    return {'wake': 0, 'light': 1, 'rem': 2, 'deep': 3, 'asleep': 4, 'restless': 5, 'awake': 6}


def get_sleep_stages_data(sleep_data, level_enum = get_sleep_enum()):
    entire_sleep_data = sleep_data['sleep']
    sleep_stages_data = []
    for sleep in entire_sleep_data['levels']:
        print(sleep)
        for sleep_stage in sleep:
            sleep_stages_data.append({
                #'user_id': user_id,
                'timestamp': sleep_stage['dateTime'].replace('T', ''),
                'sec': sleep_stage['seconds'],
                'level': level_enum[sleep_stage['level']]
            })
    return sleep_stages_data


def main():
    USER_ID, CLIENT_SECRET, server = instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
    print(get_sleep_stages_data(get_sleep_data(get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN),
                                           '2018-11-10')))


if __name__ == '__main__':
    main()


"""
def get_sleep_data():
    sleep_data =
    print(sleep_data)
    sleep_data = sleep_data['sleep'][0]['levels']['data']
    processed_sleep_data = []
    level_enum = {'wake': 0, 'light': 1, 'rem': 2, 'deep': 3}

#    print(sleep_data)
    for sleep in sleep_data:
        processed_sleep_data.append({
            'user_id': user_id,
            'date': sleep['dateTime'].split('T')[0],
            'time': sleep['dateTime'].split('T')[-1],
            'sec': sleep['seconds'],
            'level': level_enum[sleep['level']]
        })

    return processed_sleep_data

get_sleep_data()

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


#print(get_activity_intraday())
"""