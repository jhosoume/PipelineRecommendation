import sys
import mysql
from meta_db import config

class DBHelper:
    def __init__():
        db_config = config['mysql']
        self._con = mysql.connector.connect(
            host = db_config['host'],
            user = db_config['user'],
            password = db_config['password'],
            database = db_config['database']
        )
        self.__cursor = self.__connection.cursor()

    def create_table():
        with self.__cursor:
            sql_create = """
                CREATE TABLE metadata (id INT PRIMARY KEY AUTO_INCREMENT,
                                        )

            """
