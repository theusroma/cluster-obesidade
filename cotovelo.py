# 907543
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math

# cria de 1 a 15 grupos pra medir a distorcao de cada
def calcular_distorcoes(dados_norm, max_clusters=15):
    distorcoes = []
    K = range(1, max_clusters + 1) 

    for i in K:
        modelo = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_norm)
        
        # media das distancias
        distorcoes.append(
            sum(np.min(cdist(dados_norm, modelo.cluster_centers_, 'euclidean'), axis=1)) / dados_norm.shape[0]
        )   
    
    return distorcoes, list(K)

def calcular_numero_clusters(distorcoes, K):
    x0, y0 = K[0], distorcoes[0]
    xn, yn = K[-1], distorcoes[-1]
    distancias = []

    # formula do elbow (formula da reta)
    for i in range(len(distorcoes)):
        x = K[i]
        y = distorcoes[i]
        numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
        denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
        distancias.append(numerador/denominador)

    numero_clusters_otimo = K[distancias.index(np.max(distancias))]  
    
    return numero_clusters_otimo