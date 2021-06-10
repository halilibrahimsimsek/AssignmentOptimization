import pandas as pd
import csv
import numpy as np
import preprocess
import decisionTree_featured as dt
import svmClassification as sc
import kneighbors_featured as kn
from random_forest import randomForest

#read preprocessed data
df = pd.read_csv('dataframe_to_use.csv')

df = df.drop("ISSUE_SUB_CATEGORY", axis=1)
data_df = pd.read_csv('data.csv')

#merge issue categories under super categories
data_df[['ISSUE_CATEGORY']] = data_df[['ISSUE_CATEGORY']].replace(
    dict.fromkeys(['AP', 'AR', 'FA', 'GL', 'GRC', 'PA', 'FAH'], 'FINANS'))
data_df[['ISSUE_CATEGORY']] = data_df[['ISSUE_CATEGORY']].replace(
    dict.fromkeys(['Database', 'LINUX', 'Development', 'Custom', 'XTR', 'Org Pub', 'Sysadmin', 'IT', 'BI'], 'DEV'))
data_df[['ISSUE_CATEGORY']] = data_df[['ISSUE_CATEGORY']].replace(
    dict.fromkeys(['INV', 'QA', 'PO', 'IPROC', 'ISUPPLIER', 'OPM Costing', 'EAM'], 'LOJISTIK'))
data_df[['ISSUE_CATEGORY']] = data_df[['ISSUE_CATEGORY']].replace(dict.fromkeys(['PIM', 'WIP', 'OE'], 'URETIM'))

df["category"] = data_df['ISSUE_CATEGORY']

#select top workers related to issue category
#manually check all categories value counts and determine best workers count
#this part should be done in for loop for all categories
filterinfDataframe = df[(df['category'] == "HR")]
finans_list = filterinfDataframe["WORKER"].value_counts().head(4).index.tolist()
model_df = filterinfDataframe.loc[filterinfDataframe['WORKER'].isin(finans_list)]
model_df = model_df.drop("category", axis=1)
#

#run models
dt.decisionTree(model_df)
sc.svm_method(model_df)
kn.kNeighbors(model_df)
randomForestObj = randomForest()
randomForestObj.rf_with_feature_select(model_df)
