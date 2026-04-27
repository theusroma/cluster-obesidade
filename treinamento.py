import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle

def carregar_dados(path):
    return pd.read_csv(path, sep=',')

def preparar_dados(dados):

    # se a coluna existir remove, cluster deve ser "nao-supervisionado"    
    if 'NObeyesdad' in dados.columns:
        dados_features = dados.drop('NObeyesdad', axis=1)
    else:
        dados_features = dados.copy()
        
    dados_num = dados_features.select_dtypes(include=['float64', 'int64'])
    return dados_num


# normalizacao com minMax
def normalizar_dados(dados_num):
    scaler = MinMaxScaler()
    normalizador = scaler.fit(dados_num)
    
    pickle.dump(normalizador, open('scaler_obesidade.pkl', 'wb'))
    pickle.dump(list(dados_num.columns), open('colunas_treino.pkl', 'wb'))
    
    dados_norm = normalizador.transform(dados_num)
    return dados_norm, dados_num

def treinar_kmeans(numero_clusters_otimo, dados_norm):
    modelo_cluster = KMeans(n_clusters=numero_clusters_otimo, random_state=42, n_init=10).fit(dados_norm)
    return modelo_cluster

def salvar_modelo(modelo_cluster):
    pickle.dump(modelo_cluster, open('modelo_obesidade.pkl', 'wb'))

# faz a mesia das colunas numericas pra descrever o perfil do grupo
def descrever_segmentos(dados_originais, dados_norm, modelo):
    clusters_previstos = modelo.predict(dados_norm)
    df_analise = dados_originais.copy()
    df_analise['Cluster_Alocado'] = clusters_previstos
    
    df_numeric = df_analise.select_dtypes(include=['float64', 'int64']).copy()
    df_numeric['Cluster_Alocado'] = clusters_previstos
    
    descricao = df_numeric.groupby('Cluster_Alocado').mean()
    descricao['Qtd_Pacientes'] = df_analise['Cluster_Alocado'].value_counts()
    
    return descricao