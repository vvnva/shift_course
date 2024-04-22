import pandas as pd
import src.app.utils.db_connector as db_connector
import src.config.db_params as db_params

def main(path):
    connector = db_connector.database_connector(db_params.DB_ARGS)
    sql_query = """
    SET search_path TO credit_scoring; 
    SELECT * FROM credit_card_balance"""
    credit_card_balance = connector.get_df_from_query(sql_query)

    credit_card_balance.columns = [x.upper() for x in credit_card_balance.columns]
    columns = credit_card_balance.columns.drop(['SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
    credit_card_balance_features = pd.DataFrame(index=pd.Index(credit_card_balance['SK_ID_CURR'].unique(), name='SK_ID_CURR'))
    aggregations = ['sum', 'mean', 'median', 'min', 'max', 'std', 'var']

    for agg_method in aggregations:
        #Посчитайте все возможные аггрегаты по картам.
        agg_full = credit_card_balance.groupby('SK_ID_CURR')[columns].agg(agg_method)
        agg_full.columns = [f"{col}_full_{agg_method}" for col in columns]
    
        #Посчитайте как меняются аггрегаты. например отношение аггрегата за все время к аггрегату за последние 3 месяца или к данных за последний месяц.
        agg_last_3m = credit_card_balance[credit_card_balance['MONTHS_BALANCE'] >= -3].groupby('SK_ID_CURR')[columns].agg(agg_method)
        agg_last_3m.columns = [f"{col}_3m_{agg_method}" for col in columns]

        ratio = pd.DataFrame(index=agg_full.index)
        for col in columns:
            ratio[f"{col}_ratio_{agg_method}"] = agg_full[f"{col}_full_{agg_method}"].divide(agg_last_3m[f"{col}_3m_{agg_method}"], fill_value=1)

        credit_card_balance_features = credit_card_balance_features.merge(agg_full, how='left', left_index=True, right_index=True)
        credit_card_balance_features = credit_card_balance_features.merge(agg_last_3m, how='left', left_index=True, right_index=True)
        credit_card_balance_features = credit_card_balance_features.merge(ratio, how='left', left_index=True, right_index=True)

    credit_card_balance_features.reset_index(inplace=True)
    credit_card_balance_features.set_index('SK_ID_CURR', inplace=True)
    credit_card_balance_features.to_csv(f"{path}/credit_card_balance_features.csv")

path = '/Users/vi/home-credit-default-risk/features'
main(path)