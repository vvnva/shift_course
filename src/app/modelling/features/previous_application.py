import pandas as pd
import src.app.utils.db_connector as db_connector
import src.config.db_params as db_params

def main(path):
    connector = db_connector.database_connector(db_params.DB_ARGS)
    sql_query = """
    SET search_path TO credit_scoring; 
    SELECT * FROM previous_application"""
    previous_application = connector.get_df_from_query(sql_query)

    previous_application.columns = [x.upper() for x in previous_application.columns]

    previous_application_features = pd.DataFrame()
    previous_application_features['SK_ID_CURR'] = previous_application['SK_ID_CURR']
    previous_application_features['SK_ID_PREV'] = previous_application['SK_ID_PREV']

    #признаки
    previous_application_features['APPLICATION_CREDIT_RATIO'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']
    previous_application_features['CREDIT_GOODS_RATIO'] = previous_application['AMT_CREDIT'] / previous_application['AMT_GOODS_PRICE']

    #аггрегации
    temp_col=previous_application.dtypes[(previous_application.dtypes != 'object') & (previous_application.columns!='SK_ID_PREV') & (previous_application.columns!='SK_ID_CURR')].index
    aggregation_functions = ['sum', 'mean', 'median', 'min', 'max', 'std', 'var']
    aggregations = {col: aggregation_functions for col in temp_col}
    agg_full = previous_application.groupby('SK_ID_CURR').agg(aggregations)
    agg_full.columns = [f"{col}_{agg_type}" for col, agg_type in agg_full.columns]
    previous_application_features = previous_application_features.merge(agg_full, how='left', left_index=True, right_on='SK_ID_CURR')

    previous_application_features.to_csv(f"{path}/previous_application_features.csv")

path = '/Users/vi/home-credit-default-risk/features'
main(path)
