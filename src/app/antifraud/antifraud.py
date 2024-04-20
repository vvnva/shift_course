import numpy as np
import pandas as pd

def delete_fraud(data):
    '''
    Удаляет строки из DataFrame, которые соответствуют одному или нескольким условиям, указывающим
    на потенциальное мошенничество.

    Условия для удаления строк:
        1. 'weighted_ext' < 0.15: Взвешенный скоринговый показатель от внешних источников ниже 0.15.
        2. 'share_for_loan' > 0.4: Доля платежей клиента по займу превышает 40% его дохода.
        3. 'min_debt' > 8000: Минимальная сумма просроченной задолженности превышает 8000.
    '''

    # Определение условий мошенничества
    data['fraud1'] = np.where(data['weighted_ext'] < 0.15, 1, 0)
    data['fraud2'] = np.where(data['share_for_loan'] > 0.4, 1, 0)
    data['fraud3'] = np.where(data['min_debt'] > 8000, 1, 0)

    print(f"Доля дефолта у клиентов с 2 фродным признаком: {data[data['fraud1'] == 1]['target'].mean() * 100:.2f}%")
    print(f"Доля дефолта у клиентов с 2 фродным признаком: {data[data['fraud2'] == 1]['target'].mean() * 100:.2f}%")
    print(f"Доля дефолта у клиентов с 2 фродным признаком: {data[data['fraud3'] == 1]['target'].mean() * 100:.2f}%")

    # Фильтрация и удаление строк, удовлетворяющих условиям мошенничества
    initial_data_count = len(data)
    clean_data = data[(data['fraud1'] == 0) & (data['fraud2'] == 0) & (data['fraud3'] == 0)]
    removed_data_count = initial_data_count - len(clean_data)
    removed_percentage = (removed_data_count / initial_data_count) * 100

    # Вывод информации о проценте удаленных данных
    print(f"Всего удалено строк: {removed_data_count} ({removed_percentage:.2f}% от исходного количества данных)")

    # Удаление столбцов fraud1, fraud2, fraud3 перед возвратом очищенного датасета
    clean_data = clean_data.drop(columns=['fraud1', 'fraud2', 'fraud3'])

    return clean_data
