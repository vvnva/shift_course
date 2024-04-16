import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

random_state = 1

def objective(trial, X, y):
    param = {
        'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 1, 12),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3)
    }

    if param['bootstrap_type'] == 'Bayesian':
        param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)

    clf = CatBoostClassifier(**param, iterations=100, verbose=False, random_state=random_state)
    score = cross_val_score(clf, X, y, n_jobs=-1, cv=3, scoring='roc_auc').mean()
    return score

def main_catboost_optuna(path):
    data = pd.read_csv(f'{path}/data_preprocessed_train_test.csv')

    train = data.loc[data['target'].notna()].drop(['target', 'sk_id_curr'], axis=1)
    target = data.loc[data['target'].notna()]['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, stratify=target, random_state=random_state)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=40, show_progress_bar=True)

    best_params = study.best_params
    best_catboost = CatBoostClassifier(**best_params, random_state=random_state)
    best_catboost.fit(x_train, y_train)

    print('ROC AUC score on test:', roc_auc_score(y_test, best_catboost.predict_proba(x_test)[:, 1]))

    joblib.dump(best_catboost, f'{path}/catboost_optuna_model.pkl')

path = '/Users/vi/home-credit-default-risk/modelling'
main_catboost_optuna(path)
