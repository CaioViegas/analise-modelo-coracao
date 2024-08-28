import pandas as pd
from typing import Optional, List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def _aplicar_pre_processamento(dataset: pd.DataFrame, colunas_remover: Optional[List[str]], colunas_label: Optional[List[str]], colunas_hot: Optional[List[str]], coluna_zero: Optional[str]) -> pd.DataFrame:
    """
    Aplica as transformações principais de pré-processamento no dataset.
    """
    if colunas_remover:
        dataset.drop(columns=colunas_remover, axis=1, inplace=True)

    if coluna_zero:
        dataset = dataset[dataset[coluna_zero] != 0]

    if colunas_label:
        le = LabelEncoder()
        for coluna in colunas_label:
            dataset[coluna] = le.fit_transform(dataset[coluna]).astype('int64')

    if colunas_hot:
        ohe = OneHotEncoder(drop='first', dtype='int64', sparse_output=False)
        for coluna in colunas_hot:
            encoded_cols = ohe.fit_transform(dataset[[coluna]])
            encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out([coluna]), index=dataset.index)
            dataset = pd.concat([dataset, encoded_df], axis=1)
            dataset.drop(columns=[coluna], inplace=True)

    return dataset

def funcao_processamento(dataset: pd.DataFrame, colunas_remover: Optional[List[str]] = None, colunas_label: Optional[List[str]] = None, colunas_hot: Optional[List[str]] = None, coluna_zero: Optional[str] = None) -> pd.DataFrame:
    """
    Realiza o pré-processamento de um DataFrame, incluindo a remoção de colunas,
    filtragem de linhas com valores zero, codificação de rótulos e codificação one-hot.
    O DataFrame processado é salvo como um arquivo CSV chamado 'dados_transformados.csv'.
    """
    dataset_processado = _aplicar_pre_processamento(dataset, colunas_remover, colunas_label, colunas_hot, coluna_zero)
    dataset_processado.to_csv("./data/dados_transformados.csv", index=False)
    return dataset_processado

def processamento_teste(dataset: pd.DataFrame, colunas_remover: Optional[List[str]] = None, colunas_label: Optional[List[str]] = None, colunas_hot: Optional[List[str]] = None, coluna_zero: Optional[str] = None) -> pd.DataFrame:
    """
    Realiza o pré-processamento de um DataFrame, retornando o DataFrame processado.
    """
    return _aplicar_pre_processamento(dataset, colunas_remover, colunas_label, colunas_hot, coluna_zero)

if __name__ == '__main__':
    dataset = pd.read_csv("./data/heart.csv")
    funcao_processamento(
        dataset=dataset,
        coluna_zero="Cholesterol",
        colunas_label=["Sex", "ExerciseAngina"],
        colunas_hot=["ChestPainType", "RestingECG", "ST_Slope"]
    )
