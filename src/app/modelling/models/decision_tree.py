import joblib
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

random_state=1

def main(path):
    data = pd.read_csv(f'{path}/data_preprocessed_train_test.csv')

    train = data.loc[data['target'].notna()].drop(['target', 'sk_id_curr'], axis=1).copy()
    target = data.loc[data['target'].notna()]['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, stratify=target, random_state=random_state)

    steps = [('scaler', StandardScaler()),
             ('decision_tree', DecisionTreeClassifier(class_weight='balanced', random_state=random_state))]
    pipeline_decision_tree = Pipeline(steps)

    param_grid_decision_tree = {'decision_tree__max_depth': [3, 5, 10, 20],
                                'decision_tree__min_samples_leaf': [1, 5, 10],
                                'decision_tree__min_samples_split': [2, 5, 10]}

    grid_search_decision_tree = GridSearchCV(pipeline_decision_tree, param_grid_decision_tree, cv=5, scoring='roc_auc', verbose=2)
    grid_search_decision_tree.fit(x_train, y_train)

    print('ROC AUC score on test:', roc_auc_score(y_test, grid_search_decision_tree.predict_proba(x_test)[:, 1]))

    joblib.dump(grid_search_decision_tree.best_estimator_, f'{path}/decision_tree.pkl')

path = '/Users/vi/home-credit-default-risk/modelling'
main(path)
