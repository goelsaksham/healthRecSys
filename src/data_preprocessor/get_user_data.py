import fitbit
import gather_keys_oauth2 as Oauth2
from application_secrets.fitbit_application_secrets import *
import time


def instantiate_user(user_name='Saksham'):
    CLIENT_ID, CLIENT_SECRET = get_fitbit_client_id(user_name), get_fitbit_client_secret(user_name)
    server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
    server.browser_authorize()
    return CLIENT_ID, CLIENT_SECRET, server


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


def get_data_for_dump(data, data_type, user_id, date_str):
    from time import gmtime, strftime

    return {
        'user_id': user_id,
        'data_type': data_type,
        'data': data,
        'pull_time': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        'date': date_str
    }


###################################################################################
#                               SLEEP DATA
###################################################################################
def get_sleep_data(auth_client, date_str):
    return auth_client.get_sleep(date=date_str)


def get_sleep_data_for_dump(auth_client, user_id, date_str):
    from time import gmtime, strftime

    sleep_json = get_sleep_data(auth_client, date_str)
    return {
        'user_id': user_id,
        'data_type': 'sleep_stages_ts',
        'data': sleep_json,
        'pull_time': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        'date': date_str
    }


def get_all_sleeps_summary(sleep_data, user_id):
    entire_sleep_data = sleep_data['sleep']
    full_sleep_summary = []
    for current_sleep_data in entire_sleep_data:
        current_sleep_summary = dict()
        current_sleep_summary['user_id'] = user_id
        current_sleep_summary['date_of_sleep'] = current_sleep_data['dateOfSleep']
        current_sleep_summary['start_time'] = current_sleep_data['startTime'].replace('T', ' ')
        current_sleep_summary['time_in_bed'] = current_sleep_data['timeInBed']
        current_sleep_summary['duration'] = current_sleep_data['duration']/(60000)
        current_sleep_summary['efficiency'] = current_sleep_data['efficiency']
        current_sleep_summary['end_time_stamp'] = current_sleep_data['endTime'].replace('T', ' ')
        current_sleep_summary['main_sleep'] = int(current_sleep_data['isMainSleep'])
        current_sleep_summary['minutes_after_wakeup'] = current_sleep_data['minutesAfterWakeup']
        current_sleep_summary['minutes_asleep'] = current_sleep_data['minutesAsleep']
        current_sleep_summary['minutes_awake'] = current_sleep_data['minutesAwake']
        current_sleep_summary['minutes_to_fall_asleep'] = current_sleep_data['minutesToFallAsleep']
        full_sleep_summary.append(current_sleep_summary)
    return full_sleep_summary


def get_entire_sleep_summary(sleep_data, user_id):
    entire_summary = sleep_data['summary']
    try:
        if 'stages' in entire_summary:
            entire_sleep_summary = {
                'user_id': user_id,
                'deep_sleep_minutes': entire_summary['stages']['deep'],
                'light_sleep_minutes': entire_summary['stages']['light'],
                'rem_sleep_minutes': entire_summary['stages']['rem'],
                'wake_sleep_minutes': entire_summary['stages']['wake'],
                'total_minutes_asleep': entire_summary['totalMinutesAsleep'],
                'total_sleep_records': entire_summary['totalSleepRecords'],
                'total_time_in_bed': entire_summary['totalTimeInBed']
            }
        else:
            entire_sleep_summary = {
                'user_id': user_id,
                'total_minutes_asleep': entire_summary['totalMinutesAsleep'],
                'total_sleep_records': entire_summary['totalSleepRecords'],
                'total_time_in_bed': entire_summary['totalTimeInBed']
            }
        return entire_sleep_summary
    except KeyError:
        return dict()


def get_sleep_stages_summary(sleep_data):
    sleep_summary = sleep_data['summary']
    try:
        sleep_stages_summary = {
            'deep': sleep_summary['stages']['deep'],
            'light': sleep_summary['stages']['light'],
            'rem': sleep_summary['stages']['rem'],
            'wake': sleep_summary['stages']['wake'],
        }
        return sleep_stages_summary
    except KeyError:
        return dict()


def get_sleep_enum():
    return {'wake': 0, 'light': 1, 'rem': 2, 'deep': 3, 'asleep': 4, 'restless': 5, 'awake': 6, 'unknown': 7}


def get_reverse_sleep_enum():
    return {0: 'wake', 1: 'light', 2: 'rem', 3: 'deep', 4: 'asleep', 5: 'restless', 6: 'awake', 7: 'unknown'}


def get_sleep_stages_data(sleep_data, user_id, level_enum = get_sleep_enum()):
    entire_sleep_data = sleep_data['sleep']
    sleep_stages_data = []
    for sleep in entire_sleep_data:
        for sleep_stage in sleep['levels']['data']:
            sleep_stages_data.append({
                'user_id': user_id,
                'timestamp': sleep_stage['dateTime'].replace('T', ' '),
                'sec': sleep_stage['seconds'],
                'level': level_enum[sleep_stage['level']]
            })
    return sleep_stages_data


###################################################################################
#                               CALORIES DATA
###################################################################################
def get_calories_data(auth_client, date_str):
    return auth_client.intraday_time_series(resource='calories', base_date=date_str)


def get_steps_data(auth_client, date_str):
    return auth_client.intraday_time_series(resource='steps', base_date=date_str)


def get_distance_data(auth_client, date_str):
    return auth_client.intraday_time_series(resource='distance', base_date=date_str)


def get_floors_data(auth_client, date_str):
    return auth_client.intraday_time_series(resource='floors', base_date=date_str)


def get_elevation_data(auth_client, date_str):
    return auth_client.intraday_time_series(resource='elevation', base_date=date_str)


def get_calories_json(auth_client, user_id, date_str):
    from time import gmtime, strftime
    calories_json = get_calories_data(auth_client, date_str)
    return {
        'user_id': user_id,
        'data_type': 'calories',
        'data': calories_json,
        'pull_time': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        'date': date_str
    }


def get_calories_summary(calories_data, user_id):
    return {
        'user_id': user_id,
        'date': calories_data['activities-calories'][0]['dateTime'],
        'value': calories_data['activities-calories'][0]['value']
    }


def get_calories_intraday(calories_data, user_id):
    date = calories_data['activities-calories'][0]['dateTime']
    intraday_calories_data = calories_data['activities-calories-intraday']['dataset']
    return_calories_intraday_data = []
    for minute_data in intraday_calories_data:
        return_calories_intraday_data.append({
            'user_id': user_id,
            'timestamp': f'{date} {minute_data["time"]}',
            'level': minute_data['level'],
            'mets': minute_data['mets'],
            'value': minute_data['value']
        })
    return return_calories_intraday_data


def get_steps_intraday(steps_data, user_id):
    date = steps_data['activities-steps'][0]['dateTime']
    intraday_steps_data = steps_data['activities-steps-intraday']['dataset']
    return_steps_intraday_data = []
    for minute_data in intraday_steps_data:
        return_steps_intraday_data.append({
            'user_id': user_id,
            'timestamp': f'{date} {minute_data["time"]}',
            'value': minute_data['value']
        })
    return return_steps_intraday_data


def get_activity_level_to_activity_label_map():
    return {0: 'sedentary', 1: 'light', 2: 'moderate', 3: 'high'}


###################################################################################
#                               HEART DATA
###################################################################################
def get_heart_data(auth_client, date_str):
    return auth_client.intraday_time_series(resource='heart', base_date=date_str)


def get_heart_json(auth_client, user_id, date_str):
    from time import gmtime, strftime
    heart_json = get_heart_data(auth_client, date_str)
    return {
        'user_id': user_id,
        'data_type': 'heart',
        'data': heart_json,
        'pull_time': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        'date': date_str
    }


def get_heart_summary(heart_data, user_id):
    return {
        'date': heart_data['activities-heart'][0]['dateTime'],
        'value': heart_data['activities-heart'][0]['value']
    }


def get_heart_intraday(heart_data, user_id):
    date = heart_data['activities-heart'][0]['dateTime']
    intraday_heart_data = heart_data['activities-heart-intraday']['dataset']
    return_heart_intraday_data = []
    for minute_data in intraday_heart_data:
        return_heart_intraday_data.append({
            'user_id': user_id,
            'timestamp': f'{date} {minute_data["time"]}',
            'value': minute_data['value']
        })
    return return_heart_intraday_data


def main():
    CLIENT_ID, CLIENT_SECRET, server = instantiate_user('Saksham')
    ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
    auth_client = get_auth_client(CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    user_id = get_fitbit_user_id(get_user_information(server))
    #print(get_sleep_data_for_dump(auth_client, user_id, '2018-11-10'))
    #print(get_all_sleeps_summary(get_sleep_data(auth_client, '2018-11-15'), user_id))
    #print(get_calories_data(auth_client, '2018-11-10'))
    #print(get_calories_intraday(get_calories_data(auth_client, '2018-11-10'), user_id))
    #print(get_heart_data(auth_client, '2018-11-10'))
    #print(get_heart_summary(get_heart_data(auth_client, '2018-11-10'), user_id))
    print(get_heart_intraday(get_heart_data(auth_client, '2019-02-27'), user_id))
    #print(get_heart_data(auth_client, '2018-02-27'))
    #heart_data = get_heart_intraday(get_heart_data(auth_client, '2018-11-05'), user_id)
    #print(heart_data)
    #import matplotlib.pyplot as plt
    #heart_beat = []
    #for minute_heart in heart_data:
    #    heart_beat.append(minute_heart['value'])
    #plt.plot(heart_beat)
    #plt.show()


if __name__ == '__main__':
    time.sleep(120)
    main()
