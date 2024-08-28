import joblib
import pandas as pd
from typing import Dict, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def criar_pipeline(componentes: int, k_best: Optional[int] = 'all') -> Pipeline:
    """
    Cria um pipeline com as etapas de pré-processamento, redução de dimensionalidade e classificação.

    Parâmetros:
    -----------
    componentes : int
        O número de componentes principais a serem retidos durante a redução de dimensionalidade com PCA.
    k_best : int ou 'all', opcional
        Número de melhores características a serem selecionadas por `SelectKBest`. Padrão é 'all'.

    Retorna:
    --------
    pipeline : sklearn.pipeline.Pipeline
        O pipeline configurado com as etapas especificadas.
    """
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('polynomial_features', PolynomialFeatures(degree=2, interaction_only=True)),
        ('pca', PCA(n_components=componentes)),
        ('selector', SelectKBest(score_func=f_classif, k=k_best)),
        ('clf', LogisticRegression())
    ])
    return pipeline

def configurar_grid_search(pipeline: Pipeline, parametros: Dict[str, List], cv_folds: int = 5) -> GridSearchCV:
    """
    Configura o GridSearchCV com validação cruzada estratificada.

    Parâmetros:
    -----------
    pipeline : Pipeline
        O pipeline de machine learning a ser otimizado.
    parametros : Dict[str, List]
        Os hiperparâmetros a serem otimizados.
    cv_folds : int, opcional
        O número de dobras (folds) para a validação cruzada. Padrão é 5.

    Retorna:
    --------
    grid_search : GridSearchCV
        O objeto configurado para realizar a busca em grade (GridSearchCV).
    """
    kfold = StratifiedKFold(n_splits=cv_folds)
    grid_search = GridSearchCV(pipeline, parametros, cv=kfold, scoring='accuracy', n_jobs=-1)
    return grid_search

def funcao_otimizadora(dataset: pd.DataFrame, target: str, componentes: int, parametros: Dict[str, List], colunas_drop: Optional[List[str]] = None) -> None:
    """
    Realiza a otimização de um pipeline de machine learning utilizando validação cruzada 
    e busca em grade de hiperparâmetros. Em seguida, avalia o modelo otimizado em um conjunto de teste 
    e salva o modelo treinado.

    Parâmetros:
    -----------
    dataset : pd.DataFrame
        O conjunto de dados a ser utilizado. Deve conter as features e a coluna alvo.
    target : str
        O nome da coluna no `dataset` que contém o alvo (variável dependente) para a predição.
    componentes : int
        O número de componentes principais a serem retidos durante a redução de dimensionalidade com PCA.
    parametros : Dict[str, List]
        Um dicionário contendo os hiperparâmetros e os valores a serem testados na busca em grade (GridSearchCV).
    colunas_drop : Optional[List[str]], opcional
        Uma lista de colunas a serem removidas do conjunto de dados antes do treinamento. 
        Se não fornecido, nenhuma coluna adicional será removida.

    Retorno:
    --------
    None
        A função não retorna nada, mas imprime os melhores parâmetros, as métricas de avaliação do modelo
        e os scores da validação cruzada. O modelo otimizado é salvo em um arquivo chamado 'modelo_completo.joblib'.
    """
    if colunas_drop is None:
        colunas_drop = []
    
    X = dataset.drop(columns=[target, *colunas_drop])
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    pipeline = criar_pipeline(componentes)
    grid_search = configurar_grid_search(pipeline, parametros)

    logging.info("Iniciando o ajuste do GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    logging.info(f"Melhores parâmetros: {best_params}")

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"ROC AUC Score: {roc_auc}")

    cv_scores = cross_val_score(best_model, X, y, cv=grid_search.cv, scoring='accuracy')

    logging.info(f"Cross-Validation Scores: {cv_scores}")
    logging.info(f"Cross-Validation Mean Accuracy: {cv_scores.mean()}")

    joblib.dump(best_model, './modelo/modelo_completo.joblib')
    logging.info("Modelo otimizado salvo com sucesso.")

if __name__ == '__main__':
    df = pd.read_csv("./data/dados_transformados.csv")
    param_grid = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'clf__penalty': ['elasticnet'],
        'clf__solver': ['saga'], 
        'clf__tol': [0.0001, 0.001, 0.01],
        'clf__fit_intercept': [True, False],
        'clf__intercept_scaling': [1, 2, 3],
        'clf__l1_ratio': [0.15, 0.5, 0.85],
        'clf__max_iter': [4000, 6000, 8000]
    }
    funcao_otimizadora(dataset=df, target='HeartDisease', componentes=12, parametros=param_grid, colunas_drop=['RestingECG_ST', 'RestingECG_Normal', 'ChestPainType_TA', 'Cholesterol'])
