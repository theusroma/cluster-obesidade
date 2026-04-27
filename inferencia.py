import pandas as pd
import pickle

def prever_novo_paciente(dados_novo_paciente):
    try:
        normalizador = pickle.load(open('scaler_obesidade.pkl', 'rb'))
        modelo = pickle.load(open('modelo_obesidade.pkl', 'rb'))
        colunas_treino = pickle.load(open('colunas_treino.pkl', 'rb'))
    except FileNotFoundError:
        return "Erro: Arquivos não encontrados!"

    # se for dicionario, vira dataframe
    if isinstance(dados_novo_paciente, dict):
        df_novo = pd.DataFrame([dados_novo_paciente])
    else:
        df_novo = dados_novo_paciente


    # se faltar alguma coluna numerica, preenche com 0
    for col in colunas_treino:
        if col not in df_novo.columns:
            df_novo[col] = 0
            
    # filtra e ordena todas as coolunas pra ficarem iguais ao do treino
    df_novo = df_novo[colunas_treino]

    dados_norm = normalizador.transform(df_novo)
    cluster_predito = modelo.predict(dados_norm)

    return cluster_predito[0]