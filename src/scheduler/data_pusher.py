from data_preprocessor.get_user_data import *
#from data_preprocessor.push_user_data import *
from last_sync_reader import *
from client_secrets.user_ids import all_users_info
from datetime import timedelta
import schedule
import time
from data_preprocessor.push_user_data import *

CONNECTION_CONFIG = {
    "hostname": 'localhost',
    "username": 'postgres',
    "password": '123',
    "database": 'postgres'
}

# connecting to postgres database
DATABASE_CONNECTION = psycopg2.connect(host=connection_config["hostname"], user=connection_config["username"], password=connection_config["password"],
                                       dbname=connection_config["database"])

# TODO: Make a mapping for table_title to the function to insert data
insert_func_map = {}

def push_user_data(table_titles, user_name, date, data):
    print("Hello")
    for title in table_titles:
        insert_func_map[title](user_name, date, data)
        #TODO: Build function to handle all of these queries
