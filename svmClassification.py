from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC as svc
from sklearn.metrics import precision_recall_fscore_support as score
import FeatureSelect as fs


def svm_method(df):

    def predicSvm(x, y, selected_feat, name=""):

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        clf = svc(kernel='rbf')
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        precision, recall, fscore, support = score(y_test, y_pred)
        print(precision)
        """print(name)
        print("Selected Features: ")
        print(selected_feat)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        svc_model = svc(kernel='rbf')
        svc_model.fit(x_train, y_train)
        y_pred = svc_model.predict(x_test)

        results = confusion_matrix(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)
        print("Confusion Matrix")
        print(results)
        print("Precision", precision)
        print("Recall", recall)
        print("F-Score", fscore)
        print("Support", support)
        print("accuracy : ", accuracy_score(y_test, y_pred))"""

    y = df['WORKER'].values
    X = df.drop('WORKER', axis=1).values
    columns = df.drop('WORKER', axis=1)
    #print("*************SVM*************")
    #print(X.shape)
    predicSvm(X, y, df.columns, name="-----Without Feature Selection-----")
    """
    X_mutual, selected_feat = fs.mutualInf(X, y, columns, kBest=15)
    predicSvm(X_mutual, y, selected_feat, name="-----Mutual Info Feature Selection-----")

    X_chi, selected_feat = fs.chi(X, y, columns, kBest=15)
    predicSvm(X_chi, y, selected_feat, name="-----Chi Feature Selection-----")

    X_trees, selected_feat = fs.extraTrees(X, y, columns, treeEst=150)
    predicSvm(X_trees, y, selected_feat, name="-----Extratrees Feature Selection-----")

    #predicSvm(X, y, df.columns, name="-----Without Feature Selection-----")

"""