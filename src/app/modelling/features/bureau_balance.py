import pandas as pd
import src.app.utils.db_connector as db_connector
import src.config.db_params as db_params

def main(path):
    connector = db_connector.database_connector(db_params.DB_ARGS)
    sql_query = """
    SET search_path TO credit_scoring; 
    SELECT * FROM bureau_balance"""
    bureau_balance = connector.get_df_from_query(sql_query)

    bureau_balance.columns = [x.upper() for x in bureau_balance.columns]
    bureau_balance_features = pd.DataFrame()
    bureau_balance_features['SK_ID_BUREAU'] = pd.DataFrame(bureau_balance['SK_ID_BUREAU'].unique())

    #Кол-во открытых кредитов
    pivot_table = bureau_balance.pivot_table(index="SK_ID_BUREAU", columns="STATUS", aggfunc="count").fillna(0)["MONTHS_BALANCE"]
    bureau_balance_features = bureau_balance_features.merge(pivot_table,on='SK_ID_BUREAU')
    bureau_balance_features['NUM_OPENED_CREDITS'] = bureau_balance_features.loc[:,"0":"5"].sum(axis=1)

    # Кол-во закрытых кредитов
    bureau_balance_features['NUM_CLOSED_CREDITS'] = bureau_balance_features['C']

    # Кол-во просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
    bureau_balance_features = bureau_balance_features.rename(columns={'0': 'NO_DPD','1': 'DPD_30','2': 'DPD_60','3': 'DPD_90','4':'DPD_120','5': 'DPD_120+_OR_SOLD'})

    # Кол-во кредитов
    num_credits = bureau_balance.groupby('SK_ID_BUREAU')['SK_ID_BUREAU'].count().rename('NUM_CREDITS')
    bureau_balance_features = bureau_balance_features.join(num_credits, on='SK_ID_BUREAU')

    # Доля закрытых кредитов
    bureau_balance_features['SHARE_CLOSED'] = bureau_balance_features['NUM_CLOSED_CREDITS'] / bureau_balance_features['NUM_CREDITS']

    # Доля открытых кредитов
    bureau_balance_features['SHARE_OPENED'] = bureau_balance_features['NUM_OPENED_CREDITS'] / bureau_balance_features['NUM_CREDITS']

    # Доля просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
    dpd_columns = ['DPD_30', 'DPD_60', 'DPD_90', 'DPD_120', 'DPD_120+_OR_SOLD']
    total_dpd = bureau_balance_features[dpd_columns].sum(axis=1)
    for column in dpd_columns:
        share_column_name = 'SHARE_' + column
        bureau_balance_features[share_column_name] = bureau_balance_features[column] / total_dpd
        bureau_balance_features[share_column_name] = bureau_balance_features[share_column_name].fillna(0)

    # Интервал между последним закрытым кредитом и текущей заявкой
    last_closed = (bureau_balance.loc[bureau_balance['STATUS'] == 'C'].groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max().reset_index().rename(columns={'MONTHS_BALANCE': 'LAST_CLOSED_CREDIT'}))
    bureau_balance_features = bureau_balance_features.merge(last_closed, on='SK_ID_BUREAU')

    # Интервал между взятием последнего активного займа и текущей заявкой
    active_credits = bureau_balance.loc[(bureau_balance['MONTHS_BALANCE'] == -1) & (~bureau_balance['STATUS'].isin(['C', 'X']))]
    last_take = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min().reset_index()
    last_take.rename(columns={'MONTHS_BALANCE': 'LAST_OPEN_CREDIT'}, inplace=True)
    last_take = last_take[last_take['SK_ID_BUREAU'].isin(active_credits['SK_ID_BUREAU'])]
    bureau_balance_features = bureau_balance_features.merge(last_take, how='left', on='SK_ID_BUREAU')

    bureau_balance_features.set_index('SK_ID_BUREAU', inplace=True)
    bureau_balance_features.to_csv(f"{path}/bureau_balance_features.csv")

path = '/Users/vi/home-credit-default-risk/features'
main(path)
#%%
