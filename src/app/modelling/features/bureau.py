import pandas as pd
import src.app.utils.db_connector as db_connector
import src.config.db_params as db_params

def main(path):
    connector = db_connector.database_connector(db_params.DB_ARGS)
    sql_query = """
    SET search_path TO credit_scoring; 
    SELECT * FROM bureau"""
    bureau = connector.get_df_from_query(sql_query)

    bureau.columns = [x.upper() for x in bureau.columns]
    bureau_features = pd.DataFrame(index=bureau['SK_ID_CURR'].unique())
    bureau_features['SK_ID_CURR']=bureau['SK_ID_CURR']

    # Максимальная сумма просрочки
    max_debt = bureau.groupby('SK_ID_CURR').agg(MAX_DEBT=('AMT_CREDIT_SUM_DEBT', 'max'))

    # Минимальная сумма просрочки
    min_debt = bureau.groupby('SK_ID_CURR').agg(MIN_DEBT=('AMT_CREDIT_SUM_DEBT', 'min'))

    #Какую долю суммы от открытого займа просрочил
    active_debt_percentage = bureau.loc[bureau['CREDIT_ACTIVE'] == 'Active'].assign(SHARE_ACTIVE_DEBT=lambda x: x['AMT_CREDIT_SUM_DEBT'] / x['AMT_CREDIT_SUM'])
    active_debt_feature = active_debt_percentage.groupby('SK_ID_CURR')['SHARE_ACTIVE_DEBT'].max()

    # Кол-во кредитов определенного типа
    pivot_credit_types = pd.pivot_table(bureau, index="SK_ID_CURR", columns="CREDIT_TYPE", values="DAYS_CREDIT_UPDATE", aggfunc="count", fill_value=0)

    # Кол-во просрочек кредитов определенного типа
    bureau['FLAG'] = (bureau['CREDIT_DAY_OVERDUE'] > 0).astype(int)
    pivot_overdue_credit = pd.pivot_table(bureau, index='SK_ID_CURR', columns='CREDIT_TYPE', values='FLAG', aggfunc='sum', fill_value=0)

    #Кол-во закрытых кредитов определенного типа
    bureau['FLAG'] = bureau['DAYS_ENDDATE_FACT'].notna().astype(int)
    pivot_closed_credit = pd.pivot_table(bureau, index='SK_ID_CURR', columns='CREDIT_TYPE', values='FLAG', aggfunc='sum', fill_value=0)
    feature_frames = [max_debt, min_debt, active_debt_feature, pivot_credit_types, pivot_overdue_credit, pivot_closed_credit]
    bureau_features = pd.concat([bureau_features] + feature_frames, axis=1)

    bureau_features.set_index('SK_ID_CURR', inplace=True)
    bureau_features.to_csv(f"{path}/bureau_features.csv")

path = '/Users/vi/home-credit-default-risk/features'
main(path)
