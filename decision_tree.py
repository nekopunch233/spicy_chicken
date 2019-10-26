import pumpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score

def decision_tree(X,y):
    # stratified cross_validation
    kf = StratifiedKFold(n_splits=5)
    fid = 0
    for train_index, test_index in kf.split(X,y):
        fid += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('fold # {}, y_train_mean: {}, y_test_mean: {}'.format(fid, np.mean(y_train), np.mean(y_test)))
        # decision tree
        clf = DecisionTreeClassifier(max_depth=10, max_features=None, max_leaf_nodes=20, min_samples_leaf=0.05,
                                     class_weight={0: class_weights[0], 1: class_weights[1]})
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores = [accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred)]
        print("fold # {}, accuracy: {}, recall: {}, precision: {}".format(fid, scores[0], scores[1], scores[2])
