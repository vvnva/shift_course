import joblib
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

random_state = 1

def main_random_forest(path):
    data = pd.read_csv(f'{path}/data_preprocessed_train_test.csv')

    train = data.loc[data['target'].notna()].drop(['target', 'sk_id_curr'], axis=1).copy()
    target = data.loc[data['target'].notna()]['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, stratify=target, random_state=random_state)

    random_forest = RandomForestClassifier(class_weight='balanced', random_state=random_state)

    param_grid_random_forest = {
        'n_estimators': [50, 100],
        'max_depth': [2, 6],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }

    grid_search_random_forest = GridSearchCV(random_forest, param_grid_random_forest, cv=5, scoring='roc_auc', verbose=2)
    grid_search_random_forest.fit(x_train, y_train)

    cv_scores = cross_val_score(grid_search_random_forest.best_estimator_, train, target, cv=5, scoring='roc_auc')

    print('ROC AUC score on test:', roc_auc_score(y_test, grid_search_random_forest.predict_proba(x_test)[:, 1]))
    print('Average ROC AUC score from cross-validation:', cv_scores.mean())

    joblib.dump(grid_search_random_forest.best_estimator_, f'{path}/random_forest.pkl')

path = '/Users/vi/home-credit-default-risk/modelling'
main_random_forest(path)
