import pandas as pd
import numpy as np

def main(path):
    application=pd.read_csv('/Users/vi/home-credit-default-risk/application_train.csv')
    application_test = pd.read_csv('/Users/vi/home-credit-default-risk/application_test.csv')

    # Добавляем столбец SET_TYPE
    application['SET_TYPE'] = 'train'
    application_test['SET_TYPE'] = 'test'

    # Объединяем датафреймы
    application = pd.concat([application, application_test], ignore_index=True)

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

    # Процентная ставка - А=К*(П/(1+П)-М-1), где К – сумма кредита, П – процентная ставка, М – количество месяцев.
    application_features['N'] = application['AMT_CREDIT'] / application['AMT_ANNUITY']
    application_features['INTEREST_RATE'] = ((application['AMT_ANNUITY'] + application_features['N'] * application['AMT_CREDIT'] + application['AMT_CREDIT'])
                                             / (application['AMT_ANNUITY'] + application_features['N'] * application['AMT_CREDIT'])-1)
    application_features.drop(columns='N', inplace=True)

    # Взвешенный скор внешних источников
    application_features['WEIGHTED_EXT']=application[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)

    # разница между средним доходом в группе и доходом заявителя - добавлена группировка по тесту или трейну
    # подсчет среднего только по трейну
    average_income_by_group = application[application['SET_TYPE'] == 'train'].groupby(['CODE_GENDER', 'NAME_EDUCATION_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index()
    average_income_by_group.rename(columns={'AMT_INCOME_TOTAL': 'MEAN_AMT_INCOME_TOTAL'}, inplace=True)
    #объединение среднего по трейну со всем датасетом(трейн+тест)
    application = application.merge(average_income_by_group, on=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], how='inner')
    #подсчет признака
    application_features['INCOME_DIFF'] = application['AMT_INCOME_TOTAL'] - application['MEAN_AMT_INCOME_TOTAL']
    application.drop(columns='MEAN_AMT_INCOME_TOTAL', inplace=True)


    application_features.set_index('SK_ID_CURR', inplace=True)
    application_features.to_csv(f"{path}/application_features_train_test.csv")

path = '/Users/vi/home-credit-default-risk/features'
main(path)
