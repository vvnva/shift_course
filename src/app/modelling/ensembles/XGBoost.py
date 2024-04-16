import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

random_state = 1

def objective(trial, X, y):
    param = {
        'objective': 'binary:logistic',
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': 100,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'verbosity': 0,
        'random_state': random_state
    }

    clf = XGBClassifier(**param)
    score = cross_val_score(clf, X, y, n_jobs=-1, cv=3, scoring='roc_auc').mean()
    return score

def main_xgboost_optuna(path):
    data = pd.read_csv(f'{path}/data_preprocessed_train_test.csv')

    train = data.loc[data['target'].notna()].drop(['target', 'sk_id_curr'], axis=1)
    target = data.loc[data['target'].notna()]['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, stratify=target, random_state=random_state)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=40, show_progress_bar=True)

    best_params = study.best_params
    best_xgboost = XGBClassifier(**best_params)
    best_xgboost.fit(x_train, y_train)

    print('ROC AUC score on test:', roc_auc_score(y_test, best_xgboost.predict_proba(x_test)[:, 1]))

    joblib.dump(best_xgboost, f'{path}/xgboost_optuna_model.pkl')

path = '/Users/vi/home-credit-default-risk/modelling'
main_xgboost_optuna(path)
