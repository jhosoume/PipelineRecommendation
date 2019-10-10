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
        self.__feats = []

    def __get_feats(self):
        if not self.__feats:
            from sklearn.datasets import load_iris
            from pymfe.mfe import MFE
            data = load_iris()
            mfe = MFE()
            mfe.fit(data.data, data.target)
            ft = mfe.extract()
            self.__feats = [feature.replace(".", "_") for feature in ft[0]]
        return self.__feats

    def create_metadata_table(self):
        sql_create = """
            CREATE TABLE metadata (id INT PRIMARY KEY AUTO_INCREMENT,
                                   name VARCHAR(255) NOT NULL UNIQUE{}
                                  );
        """
        for feature in self.__get_feats():
            if feature == "int":
                feature = "intt"
            sql_create = sql_create.format(""", {} DOUBLE{}""").format(feature, {})
        sql_create = sql_create.format("")
        self.__cursor.execute(sql_create)

    def create_models_table(self,
        scores = ["recall_micro", "recall_macro", "accuracy", "precision_micro",
                  "precision_macro", "f1_micro", "f1_macro", "fit_time", "score_time"
                 ]):
        sql_create = """
            CREATE TABLE models (id INT PRIMARY KEY AUTO_INCREMENT,
                                 name VARCHAR(255) NOT NULL,
                                 model VARCHAR(255) NOT NULL{}
                                 );
        """
        measures = ["mean", "std"]
        for score in scores:
            for measure in measures:
                info = "{}_{}".format(score, measure)
                sql_create = sql_create.format(""", {} DOUBLE{}""").format(info, {})
        sql_create = sql_create.format("")
        self.__cursor.execute(sql_create)
        # Creating unique pair of name (dataset) and model
        sql_unique = "ALTER TABLE models ADD UNIQUE INDEX (name, model);"
        self.__cursor.execute(sql_unique)


    def drop_table(self, name = "metadata"):
        self.__cursor.execute("DROP TABLE {};".format(name))

    def add_metadata_record(self, types, values):
        if (len(types) > 0 and len(types) != len(values)):
            raise ValueError("List of types and values must be of same length")
        sql_insert = """ INSERT INTO metadata ({}) VALUES ({});"""

        # Getting types in the format suitable for inclusion
        valid_types = ""
        for type in types:
            if type == "int":
                type = "intt"
            valid_types += type.replace(".", "_") + ", "
        valid_types = valid_types[:-2]
        # Including fields to be substituted by the values
        to_subst = ""
        for indx in range(len(values)):
            to_subst += "%s, "
        to_subst = to_subst[:-2] # Elimineting comma and empty space
        print(sql_insert.format(valid_types, to_subst))
        self.__cursor.execute(sql_insert.format(valid_types, to_subst), values)
        self.__con.commit()
        print(self.__cursor.rowcount, "record inserted.")

    def add_model_record(self, types, values):
        if (len(types) > 0 and len(types) != len(values)):
            raise ValueError("List of types and values must be of same length")
        sql_insert = """ INSERT INTO models ({}) VALUES ({});"""

        # Including fields to be substituted by the values
        to_subst = ""
        for indx in range(len(values)):
            to_subst += "%s, "
        to_subst = to_subst[:-2] # Elimineting comma and empty space
        print(sql_insert.format(types, to_subst))
        self.__cursor.execute(sql_insert.format(",".join(types), to_subst), values)
        self.__con.commit()
        print(self.__cursor.rowcount, "record inserted.")

    def get_all_metadata(self):
        self.__cursor.execute("SELECT * FROM metadata;")
        return self.__cursor.fetchall()

    def get_all_models(self):
        self.__cursor.execute("SELECT * FROM models;")
        return self.__cursor.fetchall()

    def get_metadata_record(self, name):
        self.__cursor.execute("SELECT * FROM metadata WHERE name = %s", (name,))
        return self.__cursor.fetchall()

    def get_model_record(self, name):
        self.__cursor.execute("SELECT * FROM models WHERE name = %s", (name,))
        return self.__cursor.fetchall()
