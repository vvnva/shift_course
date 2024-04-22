import pandas as pd
import src.app.utils.db_connector as db_connector
import src.config.db_params as db_params

def main(path):
    connector = db_connector.database_connector(db_params.DB_ARGS)
    sql_query = """
    SET search_path TO credit_scoring; 
    SELECT * FROM installments_payments"""
    installments_payments = connector.get_df_from_query(sql_query)

    installments_payments.columns = [x.upper() for x in installments_payments.columns]

    installments_payments_features = pd.DataFrame()
    installments_payments_features['SK_ID_CURR'] = installments_payments['SK_ID_CURR']
    installments_payments_features['SK_ID_PREV'] = installments_payments['SK_ID_PREV']

    #признаки
    installments_payments_features['DIFF_DAYS'] = installments_payments['DAYS_INSTALMENT'] - installments_payments['DAYS_ENTRY_PAYMENT']
    installments_payments_features['DIFF_INSTALMENT'] = installments_payments['AMT_PAYMENT'] - installments_payments['AMT_INSTALMENT']

    #аггрегации
    aggregation_functions = ['sum', 'mean', 'median', 'min', 'max', 'std', 'var']
    aggregations = {col: aggregation_functions for col in ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT']}
    agg_full = installments_payments.groupby('SK_ID_CURR').agg(aggregations)
    agg_full.columns = [f"{col}_{agg_type}" for col, agg_type in agg_full.columns]
    installments_payments_features = installments_payments_features.merge(agg_full, how='left', left_index=True, right_on='SK_ID_CURR')
    installments_payments_features.to_csv(f"{path}/installments_payments_features.csv")

path = '/Users/vi/home-credit-default-risk/features'
main(path)

