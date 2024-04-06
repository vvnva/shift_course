import pandas as pd
from src.app.utils.stat_tests import (mann_whitney_significance, chi2_significance, ordinal_encoder, verdict,
                                      bootstraping_mean)

def stats(read_path, save_path):
    """
    Проверяет значимость признаков и оставляет только значимые
    """
    previous_application = pd.read_csv(f"{read_path}/previous_application.csv")
    application = pd.read_csv(f"{read_path}/application_train.csv", usecols=['TARGET', 'SK_ID_CURR'])
    previous_application = previous_application.merge(application, on='SK_ID_CURR')

    categorical = previous_application.select_dtypes(include='object').columns
    numeric = previous_application.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR', 'SK_ID_PREV']))

    for col in numeric:
        previous_application[col] = previous_application[col].fillna(previous_application[col].median())
        previous_application = mann_whitney_significance(previous_application, col)
    for col in categorical:
        previous_application[col] = previous_application[col].fillna('NA')
        previous_application[col] = ordinal_encoder(previous_application, col)
        previous_application = chi2_significance(previous_application, col)

    previous_application.to_csv(f"{save_path}/previous_application.csv")


def stat_with_bootsrap(read_path, save_path):
    """
    Проверяет значимость категориальных и числовых признаков и оставляет только значимые
    """
    previous_application = pd.read_csv(f"{read_path}/previous_application.csv")
    categorical = previous_application.select_dtypes(include='object').columns
    numeric = previous_application.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR', 'SK_ID_PREV']))

    for col in categorical:
        previous_application[col] = previous_application[col].fillna('NA')
        previous_application[col] = ordinal_encoder(previous_application, col)
        previous_application = chi2_significance(previous_application, col)

    previous_application[numeric] = previous_application[numeric].fillna(previous_application[col].median())
    important_features = [col for col in numeric if verdict(bootstraping_mean(previous_application,
                                                                              previous_application['TARGET'],
                                                                              feat_name=col)) == 1]
    numeric_and_not_important = set(previous_application.columns) - set(categorical) - set(important_features)
    previous_application.drop(labels=numeric_and_not_important, axis=1, inplace=True)
    previous_application.to_csv(f"{save_path}/previous_application_bootstrap.csv")


read_path = '/Users/vi/home-credit-default-risk'
save_path = '/Users/vi/home-credit-default-risk/signinficance'

stats(read_path, save_path)
#stat_with_bootsrap(read_path, save_path)

#%%
