# Projeto: Análise de Dados e Previsão de Doenças Cardíacas

Este repositório contém um projeto de análise de dados e criação de um modelo de machine learning para a previsão binária do surgimento de doenças cardíacas em pacientes, utilizando um conjunto de dados que inclui diversos fatores de risco.

## Estrutura do Projeto

A estrutura de diretórios e arquivos do projeto está organizada da seguinte forma:

├── data │ ├── heart.csv │ └── dados_transformados.csv ├── modelo │ └── modelo_completo.joblib ├── notebooks │ ├── analise_grafica.ipynb │ ├── analise_modelos.ipynb │ ├── informacoes_dataset.ipynb │ └── teste_modelo.ipynb ├── src │ ├── avaliador.py │ ├── main.py │ ├── otimizacao.py │ └── preprocessador.py └── README.md

### Descrição das Pastas

- **data/**: Contém os arquivos de dados utilizados no projeto, incluindo o dataset original `heart.csv` e o dataset transformado `dados_transformados.csv`.
  
- **modelo/**: Contém o modelo treinado e otimizado salvo no arquivo `modelo_completo.joblib`.

- **notebooks/**: Contém os notebooks Jupyter utilizados para análise de dados, exploração de modelos, e teste do modelo final.
  
- **src/**: Contém os scripts Python que implementam as principais funcionalidades do projeto, como pré-processamento, otimização e avaliação do modelo.

## Dataset

O dataset original contém 918 entradas e 12 colunas, representando diversos fatores relacionados à saúde cardíaca dos pacientes. As colunas incluem:

- **Age**: Idade do paciente
- **Sex**: Gênero do paciente
- **ChestPainType**: Tipo de dor no peito
- **RestingBP**: Pressão arterial em repouso
- **Cholesterol**: Nível de colesterol
- **FastingBS**: Glicemia de jejum
- **RestingECG**: Resultados do eletrocardiograma em repouso
- **MaxHR**: Frequência cardíaca máxima atingida
- **ExerciseAngina**: Angina induzida por exercício
- **Oldpeak**: Depressão do segmento ST
- **ST_Slope**: Inclinação do segmento ST
- **HeartDisease**: Diagnóstico de doença cardíaca (variável alvo)

## Modelos de Machine Learning

Vários modelos foram testados, utilizando diferentes escaladores (Scaler). O modelo LogisticRegression combinado com o MinMaxScaler foi o que apresentou o melhor desempenho. Esse modelo passou por um processo de otimização utilizando GridSearchCV para selecionar os melhores hiperparâmetros.

## Scripts

´avaliador.py´
Contém a função funcao_avaliadora que treina o modelo e avalia seu desempenho nos dados de teste, utilizando métricas como classification_report, roc_auc_score, matthews_corrcoef, e confusion_matrix.

´otimizacao.py´
Implementa funções para criar um pipeline de machine learning (criar_pipeline), configurar a busca por hiperparâmetros (configurar_grid_search), e otimizar o modelo utilizando validação cruzada e GridSearchCV (funcao_otimizadora).

´preprocessador.py´
Inclui funções para realizar o pré-processamento dos dados, como remoção de colunas, codificação de rótulos e codificação one-hot. O script salva o dataset processado como dados_transformados.csv.

´main.py´
Script principal que integra o pré-processamento e a otimização do modelo. Permite configurar os parâmetros e executar todo o fluxo de trabalho de treinamento e avaliação do modelo.
