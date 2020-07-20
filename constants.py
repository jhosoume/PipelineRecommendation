RANDOM_STATE = 42
# CLASSIFIERS = ["svm", "linear_discriminant", "kneighbors",
#                "decision_tree", "gaussian_nb", "gradient_boosting"]
CLASSIFIERS = ["svm", "logistic_regression", "linear_discriminant", "kneighbors",
               "decision_tree", "gaussian_nb", "random_forest", "gradient_boosting"]
# CLASSIFIERS = ["svm", "kneighbors",
#                "decision_tree", "gaussian_nb", "random_forest", "gradient_boosting"]
# CLASSIFIERS = ["svm", "kneighbors",
#                "decision_tree", "gaussian_nb", "random_forest"]
               #"neural_network"]
# REGRESSORS = [ "svm", "knn","gradient_boosting", "random_forest", "decision_tree", "random", "default"]
REGRESSORS = [ "ann", "cart", "randomForest", "dwnn", "random", "default"]
CLASSIFIERS_SCORES = ["recall_micro", "recall_macro", "accuracy", "precision_micro",
                      "precision_macro", "f1_micro", "f1_macro",
                      "balanced_accuracy",
                      "fit_time", "score_time"]

REGRESSORS_SCORES = ["max_error", "mean_absolute_error", "r2_score", "median_absolute_error",
                      "mean_squared_error"]

# PRE_PROCESSES = ["SMOTE", "RandomUnder", "HARF", "AENN"]

PRE_PROCESSES = ["SMOTE", "RandomUnder", "HARF", "ENN"]

# PRE_PROCESSES = ["SMOTE", "ADASYN", "HARF", "AENN"]

# CLASSIFIERS_SCORES = ["recall_micro", "recall_macro", "accuracy", "precision_micro",
#                       "precision_macro", "f1_micro", "f1_macro", "fit_time", "score_time"]

REG_ORDER = ["SVM", "kNN", "GB", "RF", "DT", "Random", "Default"]
