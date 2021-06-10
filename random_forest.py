from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import FeatureSelect as fs

class randomForest:

    def predicRandomForest(self, x, y, selected_feat, name=""):
        print(name)
        print("Selected Features: ")
        print(selected_feat)

        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()

        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, cv=10)
        # Fit the random search model

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        rf_random.fit(x_train, y_train)

        y_pred = rf_random.best_estimator_.predict(x_test)
        results = confusion_matrix(y_test, y_pred)
        precision, recall, fscore, support = score(y_test, y_pred)
        print("Confusion Matrix")
        print(results)
        print("Precision", precision)
        print("Recall", recall)
        print("F-Score", fscore)
        print("Support", support)
        print("accuracy : ", accuracy_score(y_test, y_pred))

        # Print the tuned parameters and score
        print("Tuned Decision Tree Parameters: {}".format(rf_random.best_params_))
        print("Best score is {}".format(rf_random.best_score_))
        print('Test set score:   {}'.format(rf_random.score(x_test, y_test)))


    def rf_with_feature_select(self, df):
        # run models
        y = df['WORKER'].values
        X = df.drop('WORKER', axis=1).values
        columns = df.drop('WORKER', axis=1)

        X_mutual, selected_feat = fs.mutualInf(X, y, columns, kBest=15)
        self.predicRandomForest(X_mutual, y, selected_feat, name="-----Mutual Info Feature Selection-----")

        X_chi, selected_feat = fs.chi(X, y, columns, kBest=15)
        self.predicRandomForest(X_chi, y, selected_feat, name="-----Chi Feature Selection-----")

        X_trees, selected_feat = fs.extraTrees(X, y, columns, treeEst=150)
        self.predicRandomForest(X_trees, y, selected_feat, name="-----Extratrees Feature Selection-----")

        self.predicRandomForest(X, y, df.columns, name="-----Without Feature Selection-----")
