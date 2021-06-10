from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif,chi2
# feature selection impl
def extraTrees(X,y,columns,treeEst=100):
    #Tree-based feature selection
    clf = ExtraTreesClassifier(n_estimators=treeEst)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    selected_feat = columns.columns[model.get_support(indices=True)].tolist()
    return X_new, selected_feat

def chi(X, y, columns, kBest=10):
    model = SelectKBest(chi2, k=13)
    X_new = model.fit_transform(X, y)
    selected_feat = columns.columns[model.get_support(indices=True)].tolist()
    return X_new, selected_feat

def mutualInf(X, y, columns, kBest=10):
    model = SelectKBest(mutual_info_classif, k=13)
    X_new = model.fit_transform(X, y)
    selected_feat = columns.columns[model.get_support(indices=True)].tolist()
    return X_new, selected_feat