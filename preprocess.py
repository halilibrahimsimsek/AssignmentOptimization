import pandas as pd

def factorize(df,columns_to_factorize):
    for col in columns_to_factorize:
        df[col] = pd.factorize(df[col])[0]

    return df

def fill_missing(df, nominal_columns):

    for i in nominal_columns:
        df[i].fillna(value='Not Given', inplace=True)

    return df
