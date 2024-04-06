import pandas as pd
from src.app.utils.stat_tests import (mann_whitney_significance, chi2_significance, ordinal_encoder, verdict,
                                      bootstraping_mean)

def stats(read_path, save_path):
    """
    Проверяет значимость признаков и оставляет только значимые
    """
    bureau = pd.read_csv(f"{read_path}/bureau.csv")
    application = pd.read_csv(f"{read_path}/application_train.csv", usecols=['TARGET','SK_ID_CURR'])
    bureau = bureau.merge(application, on='SK_ID_CURR')

    categorical = bureau.select_dtypes(include='object').columns
    numeric = bureau.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR']))

    for col in numeric:
        bureau[col] = bureau[col].fillna(bureau[col].median())
        bureau = mann_whitney_significance(bureau, col)
    for col in categorical:
        bureau[col] = bureau[col].fillna('NA')
        bureau[col] = ordinal_encoder(bureau, col)
        bureau = chi2_significance(bureau,col)

    bureau.to_csv(f"{save_path}/bureau.csv")

def stat_with_bootsrap(read_path, save_path):
    """
    Проверяет значимость категориальных и числовых признаков и оставляет только значимые
    """
    bureau = pd.read_csv(f"{read_path}/bureau.csv")
    categorical = bureau.select_dtypes(include='object').columns
    numeric = bureau.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR']))

    for col in categorical:
        bureau[col] = bureau[col].fillna('NA')
        bureau[col] = ordinal_encoder(bureau, col)
        bureau = chi2_significance(bureau,col)

    bureau[numeric] = bureau[numeric].fillna(bureau[col].median())
    important_features = [col for col in numeric if verdict(bootstraping_mean(bureau, bureau['TARGET'], feat_name=col)) == 1]
    numeric_and_not_important = set(bureau.columns) - set(categorical) - set(important_features)
    bureau.drop(labels=numeric_and_not_important, axis=1, inplace=True)
    bureau.to_csv(f"{save_path}/bureau_bootstrap.csv")

read_path = '/Users/vi/home-credit-default-risk'
save_path = '/Users/vi/home-credit-default-risk/signinficance'

stats(read_path, save_path)
#stat_with_bootsrap(read_path, save_path)
