RANDOM_STATE = 7
CLASSIFIERS = ["svm", "logistic_regression", "linear_discriminant", "kneighbors",
               "decision_tree", "gaussian_nb", "random_forest", "gradient_boosting"]
REGRESSORS = []
CLASSIFIERS_SCORES = ["recall_micro", "recall_macro", "accuracy", "precision_micro",
                      "precision_macro", "f1_micro", "f1_macro", "fit_time", "score_time"]
REGRESSORS_SCORES = ["max_error", "neg_mean_absolute_error", "r2", "neg_median_absolute_error"]
