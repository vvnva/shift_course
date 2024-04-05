import pandas as pd
import numpy as np
import src.app.utils.db_connector as db_connector
import src.config.db_params as db_params

def main(path):
    connector = db_connector.database_connector(db_params.DB_ARGS)
    sql_query = """
    SET search_path TO credit_scoring; 
    SELECT * FROM application"""
    application = connector.get_df_from_query(sql_query)

    application.columns = [x.upper() for x in application.columns]
    application_features = pd.DataFrame()
    application_features['SK_ID_CURR'] = application['SK_ID_CURR']


    # Кол-во документов
    document_columns = application.filter(regex='^FLAG_DOCUMENT_').columns
    application_features['NUM_DOCUMENTS'] = application[document_columns].sum(axis=1)

    # Есть ли полная информация о доме.
    house_columns = ['APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG',
                     'COMMONAREA_AVG','ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG',
                     'LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG',
                     'APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE',
                     'COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE',
                     'LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE',
                     'NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI',
                     'YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI',
                     'FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI',
                     'NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI','FONDKAPREMONT_MODE','HOUSETYPE_MODE',
                     'TOTALAREA_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE']

    application_features['COMPLETE_HOME_INFO'] = np.where(application[house_columns].isnull().sum(axis=1) < 30, 1, 0)

    # Кол-во полных лет
    application_features['AGE'] = application['DAYS_BIRTH'] // -365

    # Сколько лет назад был сменён документ
    application_features['DOC_CHANGE_YEARS_AGO'] = (application['DAYS_ID_PUBLISH'] // -365).astype(int)

    # В каком возрасте клиент сменил документ
    application_features['DOC_CHANGE_AGE'] = application_features['AGE'] - application_features['DOC_CHANGE_YEARS_AGO']

    # Признак задержки смены документа
    application_features['DOC_CHANGE_DELAY'] = np.where((application_features['DOC_CHANGE_AGE']==14)|
                                                  (application_features['DOC_CHANGE_AGE']==20)|
                                                  (application_features['DOC_CHANGE_AGE']==45), 0, 1)

    # Доля денег которые клиент отдает на займ за год
    application_features['SHARE_FOR_LOAN']=application['AMT_ANNUITY']/application['AMT_INCOME_TOTAL']

    # Среднее кол-во детей в семье на одного взрослого
    application_features['AVG_CHILDREN_PER_ADULT']=application['CNT_CHILDREN']/(application['CNT_FAM_MEMBERS'] - application['CNT_CHILDREN'])

    # Средний доход на ребенка
    application_features['AVG_INCOME_PER_CHILD']=application['AMT_INCOME_TOTAL'] / application['CNT_CHILDREN']
    application_features['AVG_INCOME_PER_CHILD'] = application_features['AVG_INCOME_PER_CHILD'].replace([np.inf, -np.inf], np.nan)

    # Средний доход на взрослого
    application_features['AVG_INCOME_PER_ADULT']=application['AMT_INCOME_TOTAL'] / (application['CNT_FAM_MEMBERS'] - application['CNT_CHILDREN'])

    # Процентная ставка
    application_features['INTEREST_RATE']=(application['AMT_CREDIT']/application['AMT_GOODS_PRICE']-1)*100

    # Взвешенный скор внешних источников
    application_features['WEIGHTED_EXT']=application[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)

    # разница между средним доходом в группе и доходом заявителя
    average_income_by_group = application.groupby(['CODE_GENDER', 'NAME_EDUCATION_TYPE'])['AMT_INCOME_TOTAL'].transform('mean')
    application_features['INCOME_DIFF'] = application['AMT_INCOME_TOTAL'] - average_income_by_group


    application_features.set_index('SK_ID_CURR', inplace=True)
    application_features.to_csv(f"{path}/application_features.csv")

path = '/Users/vi/home-credit-default-risk/features'
main(path)