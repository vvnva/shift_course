import pandas as pd
from src.app.utils.stat_tests import mann_whitney_significance, chi2_significance, ordinal_encoder, verdict, bootstraping_mean

def stats(read_path, save_path):
    """
    Проверяет значимость признаков и оставляет только значимые
    """
    application = pd.read_csv(f"{read_path}/application_train.csv")

    categorical = application.select_dtypes(include='object').columns
    numeric = application.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR']))

    for col in numeric:
        application[col] = application[col].fillna(application[col].median())
        application = mann_whitney_significance(application, col)
    for col in categorical:
        application[col] = application[col].fillna('NA')
        application[col] = ordinal_encoder(application, col)
        application = chi2_significance(application, col)

    application.to_csv(f"{save_path}/application.csv")

def stat_with_bootsrap(read_path, save_path):
    """
    Проверяет значимость категориальных и числовых признаков и оставляет только значимые
    """
    application = pd.read_csv(read_path)
    categorical = application.select_dtypes(include='object').columns
    numeric = application.columns.difference(set(list(categorical) + ['TARGET', 'SK_ID_CURR']))

    for col in categorical:
        application[col] = application[col].fillna('NA')
        application[col] = ordinal_encoder(application, col)
        application = chi2_significance(application, col)

    application[numeric] = application[numeric].fillna(application[col].median())
    important_features = [col for col in numeric if verdict(bootstraping_mean(application, application['TARGET'], feat_name=col)) == 1]
    numeric_and_not_important = set(application.columns) - set(categorical) - set(important_features)
    application.drop(labels=numeric_and_not_important, axis=1, inplace=True)
    application.to_csv(f"{save_path}/application_bootstrap.csv")

read_path = '/Users/vi/home-credit-default-risk'
save_path = '/Users/vi/home-credit-default-risk/signinficance'

stats(read_path, save_path)
#stat_with_bootsrap(read_path, save_path)
