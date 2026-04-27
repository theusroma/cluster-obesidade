import pandas as pd
import treinamento
import inferencia
import cotovelo


        # AGE 
        #FAF frequencia de atividades
        # WEIGHT 
        # FCVC  frequencia de verdurasa
        # NCP refeicoes principais
        # CH2O agua/hidratacao
        # FAF frequencia de exerccios
        # TUE tempo no celular



def main():
    print("TREINAMENTO")
    caminho_csv = 'ObesityDataSet_raw_and_data_sinthetic.csv' 
    
    try:
        dados_pacientes = treinamento.carregar_dados(caminho_csv)
        print(f"Dados carregados: {dados_pacientes.shape[0]} pacientes encontrados.")
    except FileNotFoundError:
        print(f"Erro: O arquivo {caminho_csv} não foi encontrado na pasta.")
        return

    # normaliza e faz a preparacao
    dados_num = treinamento.preparar_dados(dados_pacientes)
    dados_norm, dados_num_salvos = treinamento.normalizar_dados(dados_num)

    # acha o k usando o cotovelo
    distorcoes, K = cotovelo.calcular_distorcoes(dados_norm, max_clusters=15)
    k_otimo = cotovelo.calcular_numero_clusters(distorcoes, K)
    print(f"O elbow definiu o número ideal de clusters como: {k_otimo}")

    # treina e salva
    modelo = treinamento.treinar_kmeans(k_otimo, dados_norm)
    treinamento.salvar_modelo(modelo)
    print("Modelo treinado e arquivos .pkl salvos com sucesso!")

    # descreve os segmentos
    print("\nDescrição de Segmentos")
    descricao_clusters = treinamento.descrever_segmentos(dados_pacientes, dados_norm, modelo)
    print(descricao_clusters[['Age', 'Weight', 'FAF', 'Qtd_Pacientes']])

    
    print("\nINFERÊNCIA")
    paciente_desconhecido = {
        'Age': 23.0, 
        'Height': 1.75, 
        'Weight': 95.0, 
        'FCVC': 2.0, # frequencia de verdurasa
        'NCP': 3.0,  # refeicoes principais
        'CH2O': 2.0, # agua/hidratacao
        'FAF': 0.0,  # frequencia de exerccios
        'TUE': 2.0   # tempo no celular
    }
    
    print(f"Avaliando novo paciente...")
    cluster_do_paciente = inferencia.prever_novo_paciente(paciente_desconhecido)
    
    print(f"O paciente pertence ao Cluster {cluster_do_paciente}!")
    print("(Verifique a tabela da Fase 2)")

if __name__ == "__main__":
    main()