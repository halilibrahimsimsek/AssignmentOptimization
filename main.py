import pandas as pd
import csv
import numpy as np
import preprocess
import decisionTree_featured as dt
import svmClassification as sc

df = pd.read_csv('data.csv')

#merge issue categories
df[['ISSUE_CATEGORY']] = df[['ISSUE_CATEGORY']].replace(dict.fromkeys(['AP','AR','FA','GL','GRC','PA','FAH'], 'FINANS'))
df[['ISSUE_CATEGORY']] = df[['ISSUE_CATEGORY']].replace(dict.fromkeys(['Database','LINUX','Development', 'Custom', 'XTR','Org Pub','Sysadmin','IT','BI'], 'DEV'))
df[['ISSUE_CATEGORY']] = df[['ISSUE_CATEGORY']].replace(dict.fromkeys(['INV','QA','PO','IPROC','ISUPPLIER','OPM Costing','EAM'], 'LOJISTIK'))
df[['ISSUE_CATEGORY']] = df[['ISSUE_CATEGORY']].replace(dict.fromkeys(['PIM','WIP','OE'], 'URETIM'))

df = df.drop("ISSUE_ID", axis=1)
df = df.drop("EUAS_KATEGORI", axis=1)
df = df.drop("JIRANAME", axis=1)

df.to_csv("afters.csv", sep='\t', encoding='utf-8')

categorical_columns = ['REPORTER', "ISSUE_TYPE", 'PRIORTY', 'COMPNAME', 'WORKER', 'EMPLOYEE_TYPE', 'ISSUE_CATEGORY',
                       'ISSUE_SUB_CATEGORY', 'EUAS_ONCELIK']
columns_to_scale = ['WORK_LOG', "WORK_LOG_TOTAL"]


df = preprocess.fill_missing(df, categorical_columns)
df = preprocess.factorize(df, categorical_columns)




