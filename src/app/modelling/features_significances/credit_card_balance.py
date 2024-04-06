import pandas as pd
from src.app.utils.stat_tests import (mann_whitney_significance, chi2_significance, ordinal_encoder, verdict,
                                      bootstraping_mean)

def stats(read_path, save_path):
    """
    Проверяет значимость признаков и оставляет только значимые
    """
    credit_card_balance = pd.read_csv(f"{read_path}/credit_card_balance.csv")
    application = pd.read_csv(f"{read_path}/application_train.csv", usecols=['TARGET','SK_ID_CURR'])
    credit_card_balance = credit_card_balance.merge(application, on='SK_ID_CURR')

    categorical = credit_card_balance.select_dtypes(include='object').columns
    numeric = credit_card_balance.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR', 'SK_ID_PREV']))

    for col in numeric:
        credit_card_balance[col] = credit_card_balance[col].fillna(credit_card_balance[col].median())
        credit_card_balance = mann_whitney_significance(credit_card_balance, col)
    for col in categorical:
        credit_card_balance[col] = credit_card_balance[col].fillna('NA')
        credit_card_balance[col] = ordinal_encoder(credit_card_balance, col)
        credit_card_balance = chi2_significance(credit_card_balance,col)

    credit_card_balance.to_csv(f"{save_path}/credit_card_balance.csv")

def stat_with_bootsrap(read_path, save_path):
    """
    Проверяет значимость категориальных и числовых признаков и оставляет только значимые
    """
    credit_card_balance = pd.read_csv(f"{read_path}/credit_card_balance.csv")
    categorical = credit_card_balance.select_dtypes(include='object').columns
    numeric = credit_card_balance.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR', 'SK_ID_PREV']))

    for col in categorical:
        credit_card_balance[col] = credit_card_balance[col].fillna('NA')
        credit_card_balance[col] = ordinal_encoder(credit_card_balance, col)
        credit_card_balance = chi2_significance(credit_card_balance,col)

    credit_card_balance[numeric] = credit_card_balance[numeric].fillna(credit_card_balance[col].median())
    important_features = [col for col in numeric if verdict(bootstraping_mean(credit_card_balance, credit_card_balance['TARGET'], feat_name=col)) == 1]
    numeric_and_not_important = set(credit_card_balance.columns) - set(categorical) - set(important_features)
    credit_card_balance.drop(labels=numeric_and_not_important, axis=1, inplace=True)
    credit_card_balance.to_csv(f"{save_path}/credit_card_balance_bootstrap.csv")

read_path = '/Users/vi/home-credit-default-risk'
save_path = '/Users/vi/home-credit-default-risk/signinficance'

stats(read_path, save_path)
#stat_with_bootsrap(read_path, save_path)

#%%
