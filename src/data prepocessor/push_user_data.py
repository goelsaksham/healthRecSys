import psycopg2
import get_user_data

# routine to run a insert query
def run_insert_query(conn, table, values):
    cur = conn.cursor()
    command = "INSERT INTO " + table + "("

    cur.execute("Select * FROM "+table)
    colnames = [desc[0] for desc in cur.description]
    # adding column names
    for column in colnames:
        command = command + column + ", "
    command = command[:-2] + ") VALUES ("

    for value in values:
        command = command + str(value) + ", "
    command = command[:-2] + ") RETURNING "+table+"."+colnames[0]
    cur.execute(command)
    conn.commit()
    # print(cur.fetchall())


# routine to run a delete query
def run_delete_query(conn, table, column, value):
    cur = conn.cursor()
    command = "DELETE FROM " + table + " WHERE " + table + "." + column + " = " + str(value)
    cur.execute(command)
    conn.commit()


# routine to run a select query
def run_select_query(conn, table, column=None):
    cur = conn.cursor()
    if column:
        command = "SELECT " + column + "FROM " + table
    else:
        command = "SELECT * FROM " + table
    cur.execute(command)
    # print(cur.fetchall())


connection_config = {
    "hostname": 'localhost',
    "username": 'postgres',
    "password": '123',
    "database": 'postgres'
}

# connecting to postgres database
database_connection = psycopg2.connect(host=connection_config["hostname"], user=connection_config["username"], password=connection_config["password"],
                                       dbname=connection_config["database"])

# sample usage
insert_values = ["'6WQRF5'", 1997, "'male'", "'fitbit'", "'CDT'", "'Saksham Goel'", "NULL", "NULL"]

# inserting sleep_intraday_data data
# values = aa.get_sleep_data()
# # print(values)
#
# for value in values:
#     date = value["date"]+" "+value["time"]
#     start_time = "timestamp '"+date+"'"
#     record = [
#         "'"+value["user_id"]+"'",
#         "'" + value["date"] + "'",
#         start_time,
#         value["sec"],
#         value["level"]
#     ]
#     run_insert_query(database_connection, "sleep_intraday_data", record)


# # inserting activity_intraday_data data
# activity_values = aa.get_activity_intraday()
# # print(activity_values)
#
# for value in activity_values:
#     date = value["date"]
#     duration = value["minute"].split(":")[0]
#     record = [
#         "'"+value["user_id"]+"'",
#         "'" + date + "'",
#         duration,
#         value["calories"],
#         value["activity_level"],
#         "NULL",
#         value['activity_met']
#     ]
#     run_insert_query(database_connection, "activity_intraday_data", record)

# function to insert
def insert_sleep_cycles_data(date):
    USER_ID, CLIENT_SECRET, server = get_user_data.instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_user_data.get_access_token(server), get_user_data.get_refresh_token(server)
    auth_client = get_user_data.get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    user_id = get_user_data.get_fitbit_user_id(get_user_data.get_user_information(server))
    sleep_cycles_data = get_user_data.get_all_sleeps_summary(get_user_data.get_sleep_data(auth_client, date),
                                                          user_id)
    print(sleep_cycles_data[0])
    for data in sleep_cycles_data:
        start_time = data["start_time"].split("T")
        starttime = "timestamp '"+start_time[0]+" "+start_time[1]+"'"
        if data["main_sleep"]:
            is_main_sleep = 1
        else:
            is_main_sleep = 0

        record = [
            "'"+data["user_id"]+"'",
            "'"+data["date_of_sleep"]+"'",
            data["duration"],
            data["efficiency"],
            is_main_sleep,
            data["minutes_after_wakeup"],
            data["minutes_asleep"],
            data["minutes_awake"],
            data["minutes_to_fall_asleep"],
            starttime,
            data["time_in_bed"]
        ]
        run_insert_query(database_connection, "sleep_cycles", record)


def insert_sleep_summary_data(date):
    USER_ID, CLIENT_SECRET, server = get_user_data.instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_user_data.get_access_token(server), get_user_data.get_refresh_token(server)
    auth_client = get_user_data.get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    user_id = get_user_data.get_fitbit_user_id(get_user_data.get_user_information(server))
    sleep_summary_data = get_user_data.get_entire_sleep_summary(get_user_data.get_sleep_data(auth_client, date),
                                                              user_id)
    print(sleep_summary_data)
    record = [
        "'"+sleep_summary_data["user_id"]+"'",
        "'"+date+"'",
        sleep_summary_data["total_minutes_asleep"],
        sleep_summary_data["total_sleep_records"],
        sleep_summary_data["total_time_in_bed"]
    ]
    run_insert_query(database_connection, "sleep_summary", record)


def insert_sleep_intraday_data(date):
    USER_ID, CLIENT_SECRET, server = get_user_data.instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_user_data.get_access_token(server), get_user_data.get_refresh_token(server)
    auth_client = get_user_data.get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    values = get_user_data.get_sleep_data(auth_client, date)
    # print(values)
    for value in values:
        date_time = value["date"]+" "+value["time"]
        start_time = "timestamp '"+date_time+"'"
        record = [
            "'"+value["user_id"]+"'",
            "'" + value["date"] + "'",
            start_time,
            value["sec"],
            value["level"]
        ]
        run_insert_query(database_connection, "sleep_intraday_data", record)


def insert_activity_intraday_data(date):
    USER_ID, CLIENT_SECRET, server = get_user_data.instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_user_data.get_access_token(server), get_user_data.get_refresh_token(server)
    auth_client = get_user_data.get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    user_id = get_user_data.get_fitbit_user_id(get_user_data.get_user_information(server))
    activity_values = get_user_data.get_calories_intraday(get_user_data.get_calories_data(auth_client, date),
                                                             user_id)

    print(activity_values)

    for value in activity_values:
        date_time = "timestamp '"+value["timestamp"]+"'"
        record = [
            "'"+value["user_id"]+"'",
            value["value"],
            value["level"],
            "NULL",
            value['mets'],
            date_time
        ]
        run_insert_query(database_connection, "activity_intraday_data", record)


def insert_activity_summary_data(date):
    USER_ID, CLIENT_SECRET, server = get_user_data.instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_user_data.get_access_token(server), get_user_data.get_refresh_token(server)
    auth_client = get_user_data.get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    user_id = get_user_data.get_fitbit_user_id(get_user_data.get_user_information(server))
    activity_values = get_user_data.get_calories_summary(get_user_data.get_calories_data(auth_client, date),
                                                          user_id)
    record = [
        "'"+activity_values['user_id']+"'",
        "NULL",
        activity_values['value'],
        "'"+activity_values['date']+"'",

    ]
    run_insert_query(database_connection, "activity_type_summary", record)


def insert_heart_rate_intraday_data(date):
    USER_ID, CLIENT_SECRET, server = get_user_data.instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_user_data.get_access_token(server), get_user_data.get_refresh_token(server)
    auth_client = get_user_data.get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    user_id = get_user_data.get_fitbit_user_id(get_user_data.get_user_information(server))
    heart_rate_data = get_user_data.get_heart_intraday(get_user_data.get_heart_data(auth_client, date),
                                                       user_id)
    for data in heart_rate_data:
        record = [
            "'"+user_id+"'",
            "timestamp '"+data["timestamp"]+"'",
            data["value"]
        ]
        run_insert_query(database_connection, "heart_rate_intraday_data", record)


def insert_sleep_raw_data(date):
    USER_ID, CLIENT_SECRET, server = get_user_data.instantiate_user()
    ACCESS_TOKEN, REFRESH_TOKEN = get_user_data.get_access_token(server), get_user_data.get_refresh_token(server)
    auth_client = get_user_data.get_auth_client(USER_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)

    user_id = get_user_data.get_fitbit_user_id(get_user_data.get_user_information(server))
    raw_sleep_data = get_user_data.get_calories_json(auth_client, user_id, date)
    record = [
        "'"+raw_sleep_data["user_id"]+"'",
        raw_sleep_data["data_type"],
        "'"+(str(raw_sleep_data["data"])).replace(r"'", r'"').replace("True", '"true"')+"'",
        "timestamp '"+raw_sleep_data["pull_time"]+"'",
        "'"+raw_sleep_data["date"]+"'"
    ]
    run_insert_query(database_connection, "raw_device_data", record)


log_dates = ['2018-11-10', '2018-11-11','2018-11-12','2018-11-13','2018-11-14','2018-11-15','2018-11-16',
             '2018-11-17','2018-11-18','2018-11-19','2018-11-20','2018-11-21']

# insert_heart_rate_intraday_data(log_dates[-1])
for date in log_dates:
    # insert_sleep_raw_data(date)
    insert_activity_intraday_data(date)
database_connection.close()
