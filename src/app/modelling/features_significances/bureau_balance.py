import pandas as pd
from src.app.utils.stat_tests import (mann_whitney_significance, chi2_significance, ordinal_encoder, verdict,
                                      bootstraping_mean)

def stats(read_path, save_path):
    """
    Проверяет значимость признаков и оставляет только значимые
    """
    bureau_balance = pd.read_csv(f"{read_path}/bureau_balance.csv")
    bureau = pd.read_csv(f"{read_path}/bureau.csv", usecols=['SK_ID_CURR', 'SK_ID_BUREAU'])
    application = pd.read_csv(f"{read_path}/application_train.csv", usecols=['TARGET', 'SK_ID_CURR'])
    bureau_balance = bureau_balance.merge(bureau, on='SK_ID_BUREAU')
    bureau_balance = bureau_balance.merge(application, on='SK_ID_CURR')


    categorical = bureau_balance.select_dtypes(include='object').columns
    numeric = bureau_balance.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU']))

    for col in numeric:
        bureau_balance[col] = bureau_balance[col].fillna(bureau_balance[col].median())
        bureau_balance = mann_whitney_significance(bureau_balance, col)
    for col in categorical:
        bureau_balance[col] = bureau_balance[col].fillna('NA')
        bureau_balance[col] = ordinal_encoder(bureau_balance, col)
        bureau_balance = chi2_significance(bureau_balance,col)

    bureau_balance.to_csv(f"{save_path}/bureau_balance.csv")

def stat_with_bootsrap(read_path, save_path):
    """
    Проверяет значимость категориальных и числовых признаков и оставляет только значимые
    """
    bureau_balance = pd.read_csv(f"{read_path}/bureau_balance.csv")
    categorical = bureau_balance.select_dtypes(include='object').columns
    numeric = bureau_balance.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU']))

    for col in categorical:
        bureau_balance[col] = bureau_balance[col].fillna('NA')
        bureau_balance[col] = ordinal_encoder(bureau_balance, col)
        bureau_balance = chi2_significance(bureau_balance,col)

    bureau_balance[numeric] = bureau_balance[numeric].fillna(bureau_balance[col].median())
    important_features = [col for col in numeric if verdict(bootstraping_mean(bureau_balance, bureau_balance['TARGET'], feat_name=col)) == 1]
    numeric_and_not_important = set(bureau_balance.columns) - set(categorical) - set(important_features)
    bureau_balance.drop(labels=numeric_and_not_important, axis=1, inplace=True)
    bureau_balance.to_csv(f"{save_path}/bureau_balance_bootstrap.csv")

read_path = '/Users/vi/home-credit-default-risk'
save_path = '/Users/vi/home-credit-default-risk/signinficance'

stats(read_path, save_path)
#stat_with_bootsrap(read_path, save_path)
