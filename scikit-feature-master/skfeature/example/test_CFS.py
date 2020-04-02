import scipy.io
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from skfeature.function.statistical_based import CFS


def main():
    # load data
    mat = scipy.io.loadmat('../data/colon.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features

    # split data into 10 folds
    ss = KFold(10, shuffle=True)

    # perform evaluation on classification task
    num_fea = 100    # number of selected features
    clf = svm.LinearSVC()    # linear SVM

    correct = 0
    for train, test in ss.split(X):
        # obtain the index of selected features on training set
        idx = CFS.cfs(X[train], y[train])

        # obtain the dataset on the selected features
        selected_features = X[:, idx[0:num_fea]]

        import pdb; pdb.set_trace()

        # train a classification model with the selected features on the training dataset
        clf.fit(selected_features[train], y[train])

        # predict the class labels of test data
        y_predict = clf.predict(selected_features[test])

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc

    # output the average classification accuracy over all 10 folds
    print('Accuracy:', float(correct)/10)

if __name__ == '__main__':
    main()
