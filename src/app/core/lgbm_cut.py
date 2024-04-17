import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


path = '/Users/vi/home-credit-default-risk/modelling'
save_path = '/Users/vi/DataspellProjects/shift/src/app/core'
model = joblib.load(f'{path}/lightgbm_optuna_model.pkl')
feature_importance = model.feature_importances_
feature_names = model.booster_.feature_name()
feature_tuples = list(zip(feature_names, feature_importance))
feature_tuples.sort(key=lambda x: x[1], reverse=True)
top_15_features = feature_tuples[:15]
top_15_feature_names = [feature_name for feature_name, _ in top_15_features]
# ['interest_rate',
# 'share_for_loan',
# 'weighted_ext',
# 'age',
# 'avg_income_per_adult',
# 'doc_change_years_ago',
# 'diff_days',
# 'application_credit_ratio',
# 'credit_goods_ratio',
# 'avg_children_per_adult',
# 'complete_home_info',
# 'doc_change_delay',
# 'amt_credit_limit_actual_full_mean',
# 'num_documents',
# 'share_active_debt']

random_state = 1


def objective(trial, X, y):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 31, 200),
        'max_depth': trial.suggest_int('max_depth', -1, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': 100,
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1e-3, 1e3),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False])
    }

    clf = LGBMClassifier(**param, random_state=random_state)
    score = cross_val_score(clf, X, y, n_jobs=-1, cv=3, scoring='roc_auc').mean()
    return score


def cut_lightgbm_optuna(path, save_path):
    data = pd.read_csv(f'{path}/data_prep_train_test.csv')
    data = data[top_15_feature_names + ['target', 'sk_id_curr']]

    train = data.loc[data['target'].notna()].drop(['target', 'sk_id_curr'], axis=1)
    target = data.loc[data['target'].notna()]['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, stratify=target,
                                                        random_state=random_state)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=40, show_progress_bar=True)

    best_params = study.best_params
    best_lightgbm = LGBMClassifier(**best_params, random_state=random_state)
    best_lightgbm.fit(x_train, y_train)

    print('ROC AUC score on test:', roc_auc_score(y_test, best_lightgbm.predict_proba(x_test)[:, 1]))

    train = data.loc[data['target'].notna()].drop(['target'], axis=1)
    target = data.loc[data['target'].notna()]['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, stratify=target,
                                                        random_state=random_state)
    cut_test = pd.concat([x_test[:200], y_test[:200]], axis=1)
    cut_test.to_csv(f'{save_path}/cut_test.csv', index=False)

    joblib.dump(best_lightgbm, f'{save_path}/lightgbm_optuna_model_cut.pkl')


cut_lightgbm_optuna(path, save_path)
