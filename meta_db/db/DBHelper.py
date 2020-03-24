import sys
import mysql
import pandas as pd
import itertools
from config import config
import constants

class NumpyMySQLConverter(mysql.connector.conversion.MySQLConverter):
    """ A mysql.connector Converter that handles Numpy types """

    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)

class DBHelper:
    def __init__(self):
        db_config = config['mysql']
        self.__con = mysql.connector.connect(
            host = db_config['host'],
            user = db_config['user'],
            password = db_config['password'],
            database = db_config['database'],
            use_pure = True,
            connect_timeout = 31536000
        )
        self.__con.set_converter_class(NumpyMySQLConverter)
        self.__cursor = self.__con.cursor()
        self.__feats = []

    def __restart(self):
        self.__con = mysql.connector.connect(
            host = db_config['host'],
            user = db_config['user'],
            password = db_config['password'],
            database = db_config['database'],
            use_pure = True,
            connect_timeout = 31536000
        )
        self.__con.set_converter_class(NumpyMySQLConverter)
        self.__cursor = self.__con.cursor()

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

    def start(self):
        self.create_scores_table()
        self.create_models_table()
        self.create_metadata_table()
        self.create_regressor_table()
        self.create_combination_table()
        self.create_preperformance_table()
        self.create_regressor_preperformance_table()

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

    def create_scores_table(self,
        models = constants.CLASSIFIERS):
        sql_create = """
            CREATE TABLE scores (id INT PRIMARY KEY AUTO_INCREMENT,
                                   name VARCHAR(255) NOT NULL UNIQUE{}
                                  );
        """
        for model in models:
            sql_create = sql_create.format(""", {} DOUBLE{}""").format(model, {})
        sql_create = sql_create.format("")
        self.__cursor.execute(sql_create)

    def create_models_table(self,
        scores = constants.CLASSIFIERS_SCORES):
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

    def create_combination_table(self):
        sql_create = """
            CREATE TABLE combinations (id INT PRIMARY KEY AUTO_INCREMENT,
                                      classifier VARCHAR(255) NOT NULL,
                                      num_preprocesses  INT,
                                      preprocesses VARCHAR(255) NOT NULL
                                      );
        """
        self.__cursor.execute(sql_create)
        # Creating unique pair of name (dataset) and model
        sql_unique = "ALTER TABLE combinations ADD UNIQUE INDEX (classifier, preprocesses);"
        self.__cursor.execute(sql_unique)

    def create_preperformance_table(self,
        scores = constants.CLASSIFIERS_SCORES):
        sql_create = """
            CREATE TABLE preperformance (id INT PRIMARY KEY AUTO_INCREMENT,
                                 name VARCHAR(255) NOT NULL,
                                 combination_id INT,
                                 FOREIGN KEY(combination_id) REFERENCES combinations(id){}
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
        sql_unique = "ALTER TABLE preperformance ADD UNIQUE INDEX (name, combination_id);"
        self.__cursor.execute(sql_unique)


    def create_regressor_table(self,
        scores = constants.REGRESSORS_SCORES):
        sql_create = """
            CREATE TABLE regressor (id INT PRIMARY KEY AUTO_INCREMENT,
                                    name VARCHAR(255) NOT NULL,
                                    classifier VARCHAR(255) NOT NULL,
                                    score VARCHAR(255) NOT NULL,
                                    model_id INT{}
                                   );
        """
        for score in scores:
            info = "{}".format(score)
            sql_create = sql_create.format(""", {} DOUBLE{}""").format(info, {})
        sql_create = sql_create.format("")
        self.__cursor.execute(sql_create)
        # Creating unique pair of regressor (name),classifier and score
        sql_unique = "ALTER TABLE regressor ADD UNIQUE INDEX (name, classifier, score, model_id);"
        self.__cursor.execute(sql_unique)

    def create_regressor_preperformance_table(self,
        scores = constants.REGRESSORS_SCORES):
        sql_create = """
            CREATE TABLE regressor_preperformance (id INT PRIMARY KEY AUTO_INCREMENT,
                                                   name VARCHAR(255) NOT NULL,
                                                   score VARCHAR(255) NOT NULL,
                                                   combination_id INT,
                                                   FOREIGN KEY(combination_id) REFERENCES combinations(id),
                                                   model_id INT{}
                                                  );
        """
        for score in scores:
            info = "{}".format(score)
            sql_create = sql_create.format(""", {} DOUBLE{}""").format(info, {})
        sql_create = sql_create.format("")
        self.__cursor.execute(sql_create)
        # Creating unique pair of regressor (name),classifier and score
        sql_unique = "ALTER TABLE regressor_preperformance ADD UNIQUE INDEX (name, combination_id, score, model_id);"
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
        # print(sql_insert.format(valid_types, to_subst))
        try:
            self.__cursor.execute(sql_insert.format(valid_types, to_subst), values)
            self.__con.commit()
            print(self.__cursor.rowcount, "record inserted.")
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                print("Dataset features are already in the database, skipping...")
            else:
                raise

    def add_model_record(self, types, values):
        if (len(types) > 0 and len(types) != len(values)):
            raise ValueError("List of types and values must be of same length")
        sql_insert = """ INSERT INTO models ({}) VALUES ({});"""

        # Including fields to be substituted by the values
        to_subst = ""
        for indx in range(len(values)):
            to_subst += "%s, "
        to_subst = to_subst[:-2] # Elimineting comma and empty space
        # print(sql_insert.format(types, to_subst))
        try:
            self.__cursor.execute(sql_insert.format(",".join(types), to_subst), values)
            self.__con.commit()
            print(self.__cursor.rowcount, "record inserted.")
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                print("Dataset features are already in the database, skipping...")
            else:
                raise err

    def add_regressor_record(self, types, values):
        if (len(types) > 0 and len(types) != len(values)):
            raise ValueError("List of types and values must be of same length")
        sql_insert = """ INSERT INTO regressor ({}) VALUES ({});"""

        # Including fields to be substituted by the values
        to_subst = ""
        for indx in range(len(values)):
            to_subst += "%s, "
        to_subst = to_subst[:-2] # Elimineting comma and empty space
        # print(sql_insert.format(types, to_subst))
        try:
            self.__cursor.execute(sql_insert.format(",".join(types), to_subst), values)
            self.__con.commit()
            print(self.__cursor.rowcount, "record inserted.")
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                print("Regressor performance is already in the database, skipping...")
            else:
                raise err

    def add_regressor_preperformance_record(self, types, values):
        if (len(types) > 0 and len(types) != len(values)):
            raise ValueError("List of types and values must be of same length")
        sql_insert = """ INSERT INTO regressor_preperformance ({}) VALUES ({});"""

        # Including fields to be substituted by the values
        to_subst = ""
        for indx in range(len(values)):
            to_subst += "%s, "
        to_subst = to_subst[:-2] # Elimineting comma and empty space
        # print(sql_insert.format(types, to_subst))
        try:
            self.__cursor.execute(sql_insert.format(",".join(types), to_subst), values)
            self.__con.commit()
            print(self.__cursor.rowcount, "record inserted.")
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                print("Regressor performance is already in the database, skipping...")
            else:
                raise err

    def add_scores_record(self, types, values):
        if (len(types) > 0 and len(types) != len(values)):
            raise ValueError("List of types and values must be of same length")
        sql_insert = """ INSERT INTO scores ({}) VALUES ({});"""

        # Including fields to be substituted by the values
        to_subst = ""
        for indx in range(len(values)):
            to_subst += "%s, "
        to_subst = to_subst[:-2] # Elimineting comma and empty space
        # print(sql_insert.format(types, to_subst))
        try:
            self.__cursor.execute(sql_insert.format(",".join(types), to_subst), values)
            self.__con.commit()
            print(self.__cursor.rowcount, "record inserted.")
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                print("Dataset features are already in the database, skipping...")
            else:
                raise err

    def add_combination_record(self, types, values):
        if (len(types) > 0 and len(types) != len(values)):
            raise ValueError("List of types and values must be of same length")
        sql_insert = """ INSERT INTO combinations ({}) VALUES ({});"""

        # Including fields to be substituted by the values
        to_subst = ""
        for indx in range(len(values)):
            to_subst += "%s, "
        to_subst = to_subst[:-2] # Elimineting comma and empty space

        try:
            self.__cursor.execute(sql_insert.format(",".join(types), to_subst), values)
            self.__con.commit()
            print(self.__cursor.rowcount, "record inserted.")
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                print("Pre-processing combination is already in the database, skipping...")
            else:
                raise err

    def add_preperformance_record(self, types, values):
        if (len(types) > 0 and len(types) != len(values)):
            raise ValueError("List of types and values must be of same length")
        sql_insert = """ INSERT INTO preperformance ({}) VALUES ({});"""

        # Including fields to be substituted by the values
        to_subst = ""
        for indx in range(len(values)):
            to_subst += "%s, "
        to_subst = to_subst[:-2] # Elimineting comma and empty space

        try:
            self.__cursor.execute(sql_insert.format(",".join(types), to_subst), values)
            self.__con.commit()
            print(self.__cursor.rowcount, "record inserted.")
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                print("Pre-processing performance is already in the database, skipping...")
            else:
                raise err

    def get_datasets_names(self):
        self.__cursor.execute("SELECT name FROM metadata;")
        return self.__cursor.fetchall()

    def get_all_metadata(self):
        self.__cursor.execute("SELECT * FROM metadata;")
        return self.__cursor.fetchall()

    def get_all_models(self):
        self.__cursor.execute("SELECT * FROM models;")
        return self.__cursor.fetchall()

    def get_models_indx(self):
        self.__cursor.execute("SELECT name, model FROM models;")
        return self.__cursor.fetchall()

    def get_all_scores(self):
        self.__cursor.execute("SELECT * FROM scores;")
        return self.__cursor.fetchall()

    def get_all_combinations(self):
        self.__cursor.execute("SELECT * FROM combinations;")
        return self.__cursor.fetchall()

    def get_combination(self, id):
        self.__cursor.execute("SELECT * FROM combinations WHERE id = %s;", (id,))
        return self.__cursor.fetchall()

    def get_all_preperformance(self):
        self.__cursor.execute("SELECT * FROM preperformance;")
        return self.__cursor.fetchall()

    def get_preperformance_combination(self, combination_id):
        self.__cursor.execute("SELECT * FROM preperformance WHERE combination_id = %s;", (combination_id,))
        return self.__cursor.fetchall()

    def get_all_regressors(self):
        self.__cursor.execute("SELECT * FROM regressor;")
        return self.__cursor.fetchall()

    def get_all_regressors_preperformance(self):
        self.__cursor.execute("select name, score, max_error, mean_absolute_error, r2_score, median_absolute_error, classifier, preprocesses from regressor_preperformance, combinations where combination_id = combinations.id;")
        return self.__cursor.fetchall()

    def get_metadata_record(self, name):
        self.__cursor.execute("SELECT * FROM metadata WHERE name = %s;", (name,))
        return self.__cursor.fetchall()

    def get_combination_record(self, preprocesses):
        self.__cursor.execute("SELECT * FROM combinations WHERE preprocesses = %s;", (preprocesses,))
        return self.__cursor.fetchall()

    def get_combination_per_indx(self, id):
        self.__cursor.execute("SELECT * FROM combinations WHERE id = %s;", (id,))
        comb = self.__cursor.fetchall()
        if len(comb) > 0:
            return comb[0]
        else:
            return comb

    def get_model_record_per_dataset(self, name):
        self.__cursor.execute("SELECT * FROM models WHERE name = %s;", (name,))
        return self.__cursor.fetchall()

    def get_model_record_per_model(self, model):
        self.__cursor.execute("SELECT * FROM models WHERE model = %s;", (model,))
        return self.__cursor.fetchall()

    def get_combination(self, classifier, preprocesses):
        self.__cursor.execute("SELECT * FROM combinations \
                               WHERE classifier = %s AND preprocesses = %s;",
                                    (classifier, preprocesses))
        result = self.__cursor.fetchall()
        if len(result) > 0:
            return result[0]
        else:
            self.add_combination_record(["classifier", "num_preprocesses", "preprocesses"],
                                        [classifier, len(preprocesses.split()), preprocesses])
            return self.get_combination(classifier, preprocesses)

    def get_metadata_datasets(self):
        self.__cursor.execute("SELECT name FROM metadata;")
        datasets = self.__cursor.fetchall()
        return list(itertools.chain.from_iterable(datasets))

    def get_preperformance_done(self):
        self.__cursor.execute("SELECT name, combination_id FROM preperformance;")
        return self.__cursor.fetchall()

    def get_preperformance_per_combination(self):
        self.__cursor.execute("SELECT * FROM preperformance WHERE;")
        return self.__cursor.fetchall()

    def metadata_columns(self):
        self.__cursor.execute("SELECT * FROM metadata LIMIT 0;")
        self.__cursor.fetchall()
        return self.__cursor.column_names

    def models_columns(self):
        self.__cursor.execute("SELECT * FROM models LIMIT 0;")
        self.__cursor.fetchall()
        return self.__cursor.column_names

    def combinations_columns(self):
        self.__cursor.execute("SELECT * FROM combinations LIMIT 0;")
        self.__cursor.fetchall()
        return self.__cursor.column_names

    def regressor_columns(self):
        self.__cursor.execute("SELECT * FROM regressor LIMIT 0;")
        self.__cursor.fetchall()
        return self.__cursor.column_names

    def preperformance_columns(self):
        self.__cursor.execute("SELECT * FROM preperformance LIMIT 0;")
        self.__cursor.fetchall()
        return self.__cursor.column_names
