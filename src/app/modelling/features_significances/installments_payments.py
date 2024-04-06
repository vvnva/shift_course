import pandas as pd
from src.app.utils.stat_tests import (mann_whitney_significance, chi2_significance, ordinal_encoder, verdict,
                                      bootstraping_mean)

def stats(read_path, save_path):
    """
    Проверяет значимость признаков и оставляет только значимые
    """
    installments_payments = pd.read_csv(f"{read_path}/installments_payments.csv")
    application = pd.read_csv(f"{read_path}/application_train.csv", usecols=['TARGET','SK_ID_CURR', 'SK_ID_PREV'])
    installments_payments = installments_payments.merge(application, on='SK_ID_CURR')

    categorical = installments_payments.select_dtypes(include='object').columns
    numeric = installments_payments.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR']))

    for col in numeric:
        installments_payments[col] = installments_payments[col].fillna(installments_payments[col].median())
        installments_payments = mann_whitney_significance(installments_payments, col)
    for col in categorical:
        installments_payments[col] = installments_payments[col].fillna('NA')
        installments_payments[col] = ordinal_encoder(installments_payments, col)
        installments_payments = chi2_significance(installments_payments,col)

    installments_payments.to_csv(f"{save_path}/installments_payments.csv")

def stat_with_bootsrap(read_path, save_path):
    """
    Проверяет значимость категориальных и числовых признаков и оставляет только значимые
    """
    installments_payments = pd.read_csv(f"{read_path}/installments_payments.csv")
    categorical = installments_payments.select_dtypes(include='object').columns
    numeric = installments_payments.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR', 'SK_ID_PREV']))

    for col in categorical:
        installments_payments[col] = installments_payments[col].fillna('NA')
        installments_payments[col] = ordinal_encoder(installments_payments, col)
        installments_payments = chi2_significance(installments_payments,col)

    installments_payments[numeric] = installments_payments[numeric].fillna(installments_payments[col].median())
    important_features = [col for col in numeric if verdict(bootstraping_mean(installments_payments, installments_payments['TARGET'], feat_name=col)) == 1]
    numeric_and_not_important = set(installments_payments.columns) - set(categorical) - set(important_features)
    installments_payments.drop(labels = numeric_and_not_important, axis=1, inplace=True)
    installments_payments.to_csv(f"{save_path}/installments_payments_bootstrap.csv")


read_path = '/Users/vi/home-credit-default-risk'
save_path = '/Users/vi/home-credit-default-risk/signinficance'

stats(read_path, save_path)
#stat_with_bootsrap(read_path, save_path)

#%%
