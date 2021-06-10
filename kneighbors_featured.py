from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import FeatureSelect as fs


def kNeighbors(df):
    def runModel(X, y, selected_feat, name=""):
        print(name)
        print("Selected Features: ")
        print(selected_feat)

        tree = KNeighborsClassifier()
        param_dist = {"n_neighbors": np.arange(1, 25)}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        tree_cv = RandomizedSearchCV(tree, param_dist, cv=10)
        tree_cv.fit(X_train, y_train)
        y_pred = tree_cv.best_estimator_.predict(X_test)
        precision, recall, fscore, support = score(y_test, y_pred)
        results = confusion_matrix(y_test, y_pred)

        print("Confusion Matrix")
        print(results)
        print("Precision", precision)
        print("Recall", recall)
        print("F-score", fscore)
        print("Support", support)
        print("Tuned KNeighborsClassifier Parameters: {}".format(tree_cv.best_params_))
        print("Best score is {}".format(tree_cv.best_score_))
        print('Test set score:   {}'.format(tree_cv.score(X_test, y_test)))
        print("accuracy : ", accuracy_score(y_test, y_pred))


    #run models
    y = df['WORKER'].values
    X = df.drop('WORKER', axis=1).values
    columns = df.drop('WORKER', axis=1)

    print("*************kNeighbors*************")
    X_mutual, selected_feat = fs.mutualInf(X, y, columns, kBest=15)
    runModel(X_mutual, y, selected_feat, name="-----Mutual Info Feature Selection-----")

    X_chi, selected_feat = fs.chi(X, y, columns, kBest=15)
    runModel(X_chi, y, selected_feat, name="-----Chi Feature Selection-----")

    X_trees, selected_feat = fs.extraTrees(X, y, columns, treeEst=150)
    runModel(X_trees, y, selected_feat, name="-----Extratrees Feature Selection-----")

    runModel(X, y, df.columns, name="-----Without Feature Selection-----")