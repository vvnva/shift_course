import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

random_state = 1

def compare(path):
    data = pd.read_csv(f'{path}/data_preprocessed_train_test.csv')
    train = data.loc[data['target'].notna()].drop(['target', 'sk_id_curr'], axis=1)
    target = data.loc[data['target'].notna()]['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, stratify=target, random_state=random_state)
    
    model_paths = [
        f'{path}/random_forest.pkl',
        f'{path}/lightgbm_optuna_model.pkl',
        f'{path}/catboost_optuna_model.pkl',
        f'{path}/xgboost_optuna_model.pkl'
    ]
    
    models = [joblib.load(path) for path in model_paths]
    
    # Подготовка графика для ROC-кривых
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))  # Размер графика можно настроить по желанию
    
    # Подготовка фигуры для матриц ошибок
    fig_cm, axes_cm = plt.subplots(nrows=1, ncols=len(models), figsize=(5 * len(models), 5))
    
    model_names = ['Random Forest', 'LightGBM', 'CatBoost', 'XGBoost']
    colors = ['b', 'g', 'r', 'c']  # Цвета для ROC-кривых каждой модели
    
    for i, model in enumerate(models):
        y_probs = model.predict_proba(x_test)[:, 1]  # Предсказанные вероятности
        roc_auc = roc_auc_score(y_test, y_probs)  # Расчёт AUC
        fpr, tpr, _ = roc_curve(y_test, y_probs)  # Получение значений для ROC-кривой
    
        # Построение ROC-кривой для каждой модели на общем графике
        ax_roc.plot(fpr, tpr, label=f'{model_names[i]} (AUC = {roc_auc:.2f})', color=colors[i])
        print(f"{model_names[i]} ROC AUC: {roc_auc:.4f}")
    
        # Построение и настройка матрицы ошибок для каждой модели
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes_cm[i], cmap=plt.cm.Blues)
        axes_cm[i].set_title(f'{model_names[i]} Confusion Matrix')
    
    # Настройка графика ROC
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random chance')
    ax_roc.set_title('Combined ROC Curves')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")
    
    # Сохранение графиков
    fig_roc.tight_layout()
    fig_roc.savefig(f'{path}/roc_curves.png')
    
    fig_cm.tight_layout()
    fig_cm.savefig(f'{path}/confusion_matrices.png')
    
    # Отображение графиков
    plt.show()

path = '/Users/vi/home-credit-default-risk/modelling'
compare(path)
