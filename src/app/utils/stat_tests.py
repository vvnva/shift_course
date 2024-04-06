import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.preprocessing import OrdinalEncoder
from tqdm.notebook import tqdm

def mann_whitney_significance(df, col):
    """
    Проверяет значимость признака по критерию Манна-Уитни
    """
    _, p_mw = mannwhitneyu(df[df['TARGET'] == 0][col], df[df['TARGET'] == 1][col])
    if p_mw >= 0.05:
        df = df.drop(col, axis=1)
    return df

def chi2_significance(df, is_significant_col):
    """
    Проверяет значимость признака по критерию хи-квадрат,
    """
    cross_tab = pd.crosstab(df[is_significant_col], df['TARGET'])
    _, p, _, _ = chi2_contingency(cross_tab)
    if p >= 0.05:
        df.drop(is_significant_col, axis=1, inplace=True)
    return df


def ordinal_encoder(df, col):
    """
    Сортирует категории признака по их среднему влиянию на целевую переменную (target)
    и применяет порядковое кодирование к этому признаку.
    """
    temp_df = df.copy()
    sorted_categories = temp_df.groupby(col)['TARGET'].mean().sort_values().index.tolist()
    ordinal = OrdinalEncoder(categories=[sorted_categories])
    encoded_column = ordinal.fit_transform(temp_df[[col]]).astype('int8')
    return pd.Series(encoded_column.flatten(), index=temp_df.index, name=col)


def bootstrap(data1, data2, n=100, func=np.mean, subtr=np.subtract, alpha=0.05):
    '''
    Бутстрап средних значений для двух групп

    data1 - выборка 1 группы
    data2 - выборка 2 группы
    n=10000 - сколько раз моделировать
    func=np.mean - функция отвыборки, например, среднее
    subtr=np.subtract,
    alpha=0.05 - 95% доверительный интервал

    return:
    ci_diff - доверительный интервал разницы средних для двух групп
    s1 - распределение средних для 1 группы
    s2 - распределение средних для 2 группы
    confidence_interval(s1, s2, n, 1 - alpha) - доверительные интервалы для двух групп
    '''
    s1, s2 = [], []
    s1_size = len(data1)
    s2_size = len(data2)


    for i in tqdm(range(n)):
        itersample1 = np.random.choice(data1, size=s1_size, replace=True)
        s1.append(func(itersample1))
        itersample2 = np.random.choice(data2, size=s2_size, replace=True)
        s2.append(func(itersample2))
    s1.sort()
    s2.sort()

    #доверительный интервал разницы
    bootdiff = subtr(s2, s1)
    bootdiff.sort()

    ci_diff = (np.round(bootdiff[np.round(n*alpha/2).astype(int)], 3),
               np.round(bootdiff[np.round(n*(1-alpha/2)).astype(int)], 3))

    return ci_diff, s1, s2


def bootstraping_mean(data, y, feat_name=None, val=[0, 1]):
    '''
    Бутстрап средних значений для любого признака

    data - датафрейм с данными
    y - таргет
    feat_name - название признака, строка

    return:
    cidiff - доверительный интервал разницы в средних значениях для двух групп
    '''
    data1 = data[(y==val[0])][feat_name]
    data2 = data[(y==val[1])][feat_name]
    s1_mean_init = np.mean(data1)
    s2_mean_init = np.mean(data2)
    cidiff, s1, s2 = bootstrap(data1, data2)

    return cidiff


def verdict(ci_diff):
    cidiff_min=0.001 #,близкое к 0
    ci_diff_abs = [abs(ele) for ele in ci_diff]
    if (min(ci_diff) <= cidiff_min <= max(ci_diff)):
        print(ci_diff,'Различия в средних статистически незначимы.')
    elif (cidiff_min >= max(ci_diff_abs) >= 0) or (cidiff_min >= min(ci_diff_abs) >= 0):
        print(ci_diff,'Различия в средних статистически незначимы.')
    else:
        print(ci_diff,'Различия в средних статистически значимы.')
        return 1