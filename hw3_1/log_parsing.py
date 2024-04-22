import pandas as pd
import json
from dataclasses import dataclass

@dataclass
class PosCashBalanceIDs:
    SK_ID_PREV: int
    SK_ID_CURR: int
    NAME_CONTRACT_STATUS: str

@dataclass
class AmtCredit:
    CREDIT_CURRENCY: str
    AMT_CREDIT_MAX_OVERDUE: float
    AMT_CREDIT_SUM: float
    AMT_CREDIT_SUM_DEBT: float
    AMT_CREDIT_SUM_LIMIT: float
    AMT_CREDIT_SUM_OVERDUE: float
    AMT_ANNUITY: float

def process_pos_cash_balance(data):
    """
    Обработка/парсинг записей из POS_CASH_balance.
    """
    df = pd.DataFrame(data).explode('records').reset_index(drop=True)
    records = pd.json_normalize(df['records'])
    records['PosCashBalanceIDs'] = records['PosCashBalanceIDs'].apply(lambda x: eval(x))
    for field in ['SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_STATUS']:
        records[field] = records['PosCashBalanceIDs'].apply(lambda x: getattr(x, field))
    records.drop(columns='PosCashBalanceIDs', inplace=True)
    return pd.concat([df.drop(columns='records'), records], axis=1)

def process_bureau(data):
    """
    Обработка/парсинг записей из bureau.
    """
    df = pd.DataFrame(data)
    records = pd.json_normalize(df['record'])
    records['AmtCredit'] = records['AmtCredit'].apply(lambda x: eval(x))
    for field in AmtCredit.__annotations__.keys():
        records[field] = records['AmtCredit'].apply(lambda x: getattr(x, field))
    records.drop(columns='AmtCredit', inplace=True)
    return pd.concat([df.drop(columns='record'), records], axis=1)

def main(log_file_path, pos_cash_csv_path, bureau_csv_path):
    """
    Функция чтения лог файла, вызова функций по парсингу, записи готовых датафремов в csv.
    """
    pos_cash_records, bureau_records = [], []

    with open(log_file_path, 'r') as file:
        for line in file:
            log_record = json.loads(line)
            if log_record['type'] == 'POS_CASH_balance':
                pos_cash_records.append(log_record['data'])
            elif log_record['type'] == 'bureau':
                bureau_records.append(log_record['data'])

    pos_cash_df = process_pos_cash_balance(pos_cash_records)
    bureau_df = process_bureau(bureau_records)

    pos_cash_df.to_csv(pos_cash_csv_path, index=False)
    bureau_df.to_csv(bureau_csv_path, index=False)


log_file_path = '/Users/vi/DataspellProjects/шифт_обучение/27.03_Обработка_данных/logs_parsing/POS_CASH_balance_plus_bureau-001.log'
pos_cash_csv_path = 'hw3_1/POS_CASH_balance_parsed.csv'
bureau_csv_path = 'hw3_1/bureau_parsed.csv'
main(log_file_path, pos_cash_csv_path, bureau_csv_path)
