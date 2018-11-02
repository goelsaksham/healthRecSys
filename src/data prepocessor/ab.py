#!/usr/bin/python

import psycopg2


# routine to run a insert query
def run_insert_query(conn, table):
    cur = conn.cursor()
    command = "INSERT INTO " + table
    cur.execute(command + "(id, birth_year, gender, provider, timezone, user_name, created_at, updated_at)"
                " VALUES (1234, 1997, 'male', 'fitbit', 'CDT', 'mohan056', NULL, NULL) RETURNING subject.id")
    conn.commit()
    print(cur.fetchall())


# routine to run a delete query
def run_delete_query(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM subject WHERE subject.id = 1234")
    conn.commit()


# routine to run a select query
def run_select_query(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM subject")
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
print(database_connection)
run_delete_query(database_connection)
run_insert_query(database_connection, "subject")
run_select_query(database_connection)
database_connection.close()
