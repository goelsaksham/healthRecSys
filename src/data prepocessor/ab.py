#!/usr/bin/python

import aa
import psycopg2
import datetime
import time


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
    print(cur.fetchall())


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
    print(cur.fetchall())


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
print(database_connection)
# run_delete_query(database_connection, "subject", "id", "'6WQRF5'")
# run_insert_query(database_connection, "subject", insert_values)
run_select_query(database_connection, "subject")

values = aa.get_sleep_data()
print(values)

for value in values:
    date = value["date"]+" "+value["time"]
    start_time = "timestamp '"+date+"'"
    record = [
        "'"+value["user_id"]+"'",
        "'" + value["date"] + "'",
        start_time,
        value["sec"],
        value["level"]
    ]
    run_insert_query(database_connection, "sleep_intraday_data", record)

database_connection.close()
