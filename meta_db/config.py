import json
import mysql.connector

config = {}

with open('config.json') as fd:
    config = json.load(fd)

_con = mysql.connector.connect(
    host = config['mysql']['host'],
    user = config['mysql']['user'],
    password = config['mysql']['password'],
)
_cursor = _con.cursor(buffered = True)
_cursor.execute("SHOW DATABASES;")
if not ( ('paje',) in _cursor):
    _cursor.execute("CREATE DATABASE paje;")
