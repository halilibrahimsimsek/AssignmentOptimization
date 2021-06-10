from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
import FeatureSelect as fs
from sklearn.metrics import accuracy_score

def decisionTree(df):
    def runModel(X, y, selected_feat, name=""):
        print(name)
        print("Selected Features: ")
        print(selected_feat)

        # Setup the parameters : param_dist
        param_dist = {"max_depth": [3, None],
                      "max_features": randint(1, 9),
                      "min_samples_leaf": randint(1, 9),
                      "criterion": ["gini", "entropy"]}

        # Instantiate a Decision Tree classifier: tree
        tree = DecisionTreeClassifier()

        # Instantiate the RandomizedSearchCV object: tree_cv
        tree_cv = RandomizedSearchCV(tree, param_dist, cv=10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        # Fit it to the data
        tree_cv.fit(X_train, y_train)

        y_pred = tree_cv.best_estimator_.predict(X_test)
        results = confusion_matrix(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)

        print("Confusion Matrix")
        print(results)
        print("Precision", precision)
        print("Recall", recall)
        print("F-Score", fscore)
        print("Support", support)
        print("accuracy : ", accuracy_score(y_test, y_pred))

        print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
        print("Best score is {}".format(tree_cv.best_score_))
        print('Test set score:   {}'.format(tree_cv.score(X_test, y_test)))

    y = df['WORKER'].values
    X = df.drop('WORKER', axis=1).values
    columns = df.drop('WORKER', axis=1)

    print("*************DecisionTree*************")
    runModel(X, y, df.columns, name="-----Without Feature Selection-----")

    X_mutual, selected_feat = fs.mutualInf(X, y, columns, kBest=15)
    runModel(X_mutual, y, selected_feat, name="-----Mutual Info Feature Selection-----")

    X_chi, selected_feat = fs.chi(X, y, columns, kBest=15)
    runModel(X_chi, y, selected_feat, name="-----Chi Feature Selection-----")

    X_trees, selected_feat = fs.extraTrees(X, y, columns, treeEst=150)
    runModel(X_trees, y, selected_feat, name="-----Extratrees Feature Selection-----")

