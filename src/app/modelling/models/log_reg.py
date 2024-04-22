import joblib
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

random_state = 2

def main(path):
    data = pd.read_csv(f'{path}/data_preprocessed_train_test.csv')

    train = data.loc[data['target'].notna()].drop(['target', 'sk_id_curr'], axis=1).copy()
    target = data.loc[data['target'].notna()]['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, stratify=target, random_state=random_state)

    steps = [('scaler', StandardScaler()),
             ('log_reg', LogisticRegression(penalty='l1', class_weight='balanced', random_state=random_state))]
    pipeline_log_reg = Pipeline(steps)

    param_grid_log_reg = {'log_reg__C': [0.01, 0.1, 1, 10],
                          'log_reg__solver': ['liblinear']}

    grid_search_log_reg = GridSearchCV(pipeline_log_reg, param_grid_log_reg, cv=5, scoring='roc_auc', verbose=2)
    grid_search_log_reg.fit(x_train, y_train)

    print('ROC AUC score on test:', roc_auc_score(y_test, grid_search_log_reg.predict_proba(x_test)[:, 1]))

    joblib.dump(grid_search_log_reg.best_estimator_, f'{path}/log_reg.pkl')

path = '/Users/vi/home-credit-default-risk/modelling'
main(path)
