import pandas as pd
import numpy as np

def remove_columns_with_excessive_nas(data, threshold=0.3):
        """
        Удаляет столбцы из DataFrame, где доля пропущенных значений превышает заданный порог.
        Возвращает: None, изменения применяются к переданному DataFrame на месте.
        """
        na_count = data.isnull().sum()
        total_rows = len(data)
        columns_to_drop = na_count[na_count > total_rows * threshold].index
        data.drop(columns=columns_to_drop, inplace=True)

def delete_correlated(data, threshold=0.9):
        """
        Удаляет мультиколлинеарные столбцы.
        threshold - пороговое значение коэффициентов корреляции Пирсона.
        Возвращает DataFrame без высококоррелированных столбцов.
        """
        corr_matrix = data.select_dtypes(include=np.number).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        high_correlated = [column for column in upper.columns if any(upper[column] > threshold)]
        return data.drop(columns=high_correlated)

def main(data_path, save_path):
        application = pd.read_csv(f"{data_path}/application_features_train_test.csv")
        bureau = pd.read_csv(f"{data_path}/bureau_features.csv")
        bureau_balance = pd.read_csv(f"{data_path}/bureau_balance_features.csv")
        credit_card_balance = pd.read_csv(f"{data_path}/credit_card_balance_features.csv")
        previous_application = pd.read_csv(f"{data_path}/previous_application_features.csv", index_col=0)
        installments_payments = pd.read_csv(f"{data_path}/installments_payments_features.csv", index_col=0)
        target_train = pd.read_csv(f"{data_path}/application_train.csv", usecols=['SK_ID_CURR', 'TARGET'])
        target_test = pd.read_csv(f"{data_path}/application_test.csv", usecols=['SK_ID_CURR'])
        target = pd.concat([target_train, target_test], ignore_index=True)

        remove_columns_with_excessive_nas(application)
        remove_columns_with_excessive_nas(bureau)
        remove_columns_with_excessive_nas(bureau_balance)
        remove_columns_with_excessive_nas(credit_card_balance)
        remove_columns_with_excessive_nas(previous_application)
        remove_columns_with_excessive_nas(installments_payments)

        data = application.merge(bureau, on='SK_ID_CURR', how='left')
        data = data.merge(bureau_balance, on='SK_ID_CURR', how='left')
        data = data.merge(credit_card_balance, on='SK_ID_CURR', how='left')
        data = data.merge(previous_application, on='SK_ID_CURR', how='left')
        data = data.merge(installments_payments, on='SK_ID_CURR', how='left')
        data = data.merge(target, on='SK_ID_CURR', how='left')

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        #Заполнение пропусков
        binary_columns = [col for col in data.columns if data[col].nunique() == 2 and col != 'TARGET']
        for column in binary_columns:
                 data[column] = data[column].fillna(data[column].mode()[0])

        non_binary_columns = [col for col in data.columns if col not in binary_columns and col != 'TARGET']
        for column in non_binary_columns:
                data[column] = data[column].fillna(data[column].median())

        # Удаление мультиколлинеарных столбцов
        data = delete_correlated(data)

        # Приведение к одинаковому названию столбцов
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.lower()

        # Сохранение файлов
        data.to_csv(f"{save_path}/data_preprocessed_train_test.csv" , index=False)

data_path = '/Users/vi/home-credit-default-risk/features'
save_path = '/Users/vi/home-credit-default-risk/modelling'
main(data_path, save_path)
