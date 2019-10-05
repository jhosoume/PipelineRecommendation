import sys
import mysql
from meta_db.config import config

class DBHelper:
    def __init__(self):
        db_config = config['mysql']
        self.__con = mysql.connector.connect(
            host = db_config['host'],
            user = db_config['user'],
            password = db_config['password'],
            database = db_config['database']
        )
        self.__cursor = self.__con.cursor()
        self.__rows = []

    def __get_rows(self):
        if not self.__rows:
            from sklearn.datasets import load_iris
            from pymfe.mfe import MFE
            data = load_iris()
            mfe = MFE()
            mfe.fit(data.data, data.target)
            ft = mfe.extract()
            self.__rows = [feature.replace(".", "_") for feature in ft[0]]
        return self.__rows

    def create_table(self):
        sql_create = """
            CREATE TABLE metadata (id INT PRIMARY KEY AUTO_INCREMENT,
                                   name VARCHAR(255){}
                                  );
        """
        for feature in self.__get_rows():
            if feature == "int":
                feature = "int_"
            sql_create = sql_create.format(""", {} DOUBLE{}""").format(feature, {})
        sql_create = sql_create.format("")
        self.__cursor.execute(sql_create)

    def drop_table(self):
        self.__cursor.execute("DROP TABLE metadata;")
