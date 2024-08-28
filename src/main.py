import pandas as pd
from typing import Dict, List, Optional
from otimizacao import funcao_otimizadora
from preprocessador import processamento_teste

def main(dataset_path: str, target: str, componentes: int, param_grid: Dict[str, List], colunas_label: Optional[List[str]] = None, colunas_hot: Optional[List[str]] = None, coluna_zero: Optional[str] = None, colunas_drop: Optional[List[str]] = None) -> None:
    # Carrega o dataset
    df = pd.read_csv(dataset_path)
    
    # Aplica o pré-processamento, se necessário
    df = processamento_teste(dataset=df, colunas_label=colunas_label, colunas_hot=colunas_hot, coluna_zero=coluna_zero)

    # Executa a otimização
    funcao_otimizadora(dataset=df, target=target, componentes=componentes, parametros=param_grid, colunas_drop=colunas_drop)

if __name__ == "__main__":
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
    main(
        dataset_path="./data/dados_transformados.csv",
        target='HeartDisease',
        componentes=12,
        param_grid=param_grid,
        colunas_label=['Sex', 'ExerciseAngina'],
        colunas_hot=['ChestPainType', 'RestingECG', 'ST_Slope'],
        coluna_zero='Cholesterol',
        colunas_drop=['RestingECG_ST', 'RestingECG_Normal', 'ChestPainType_TA', 'Cholesterol']
    )
