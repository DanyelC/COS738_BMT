import logging
import configparser
from datetime import datetime
import math
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_conf(file):
    """
    Funcão para capturar as configuracoes para a indexador.
    :param str file: nome do arquivo com as configuracoes
    return list files_to_read: lista de strings contendo os caminhos dos arquivos a serem lidos
    return str file_to_write: string contendo o caminho do arquivo a ser escrito
    """
    logging.basicConfig(filename='../logs/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Iniciando operacao de captura das configuracões.')

    try:
    # Ler arquivo de configuracão
        logging.info('Iniciando leitura do arquivo de configuracao.')
        cfg_parser = configparser.RawConfigParser()
        cfg_parser.readfp(open(file))
        
        files_to_read = cfg_parser.options('CONF')
        files_to_read = cfg_parser.get('CONF', 'LEIA')
        file_to_write = cfg_parser.get('CONF', 'ESCREVA')

        logging.info('Arquivo de configuracao lido com sucesso.')
        return files_to_read, file_to_write
    except Exception as e:
        logging.error(f'Ocorreu o seguinte erro ao ler o arquivo de configuracao: {str(e)}')



def get_doc_matrix(tokens_file):
    logging.basicConfig(filename='../logs/index.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Iniciando a criação da matriz de documentos.")

    inverted_list = pd.read_csv(f"../result/{tokens_file}", sep=';', converters={"Aparicao": pd.eval})

    matrix = pd.DataFrame(inverted_list["Token"])
    matrix.set_index(["Token"], inplace=True)

    for token, docs in inverted_list.itertuples(index=False):
        for doc in docs:
            if str(doc) in matrix.columns:
                matrix.at[token, str(doc)] += 1
            else:
                zeros = pd.DataFrame(np.zeros((matrix.shape[0], 1)), index=inverted_list["Token"], columns=[str(doc)])
                matrix = pd.concat([matrix, zeros], axis=1)
                matrix.at[token, str(doc)] = 1

    logging.info(f"A matriz possui {len(matrix.index)} tokens e {len(matrix.columns)} documentos.")

    return matrix


def get_tf(token, document, matrix):
    return int(matrix.loc[token, str(document)])


def get_tfn(token, document, matrix):
    tf = get_tf(token, document, matrix)
    biggest_tf = matrix.loc[:, str(document)].max()
    return tf / biggest_tf



def get_model(matrix, type_tf):
    logging.basicConfig(filename='../logs/index.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Iniciando a geração do modelo com tf.")
    if type_tf != "tf":
        logging.info("Utilizando normalização.")
    
    start_time = datetime.now()
    weights = matrix.copy()

    for token in tqdm(weights.index):
        idf = math.log10(len(weights.columns) / weights.loc[token].astype(bool).sum())

        for document in weights.columns:
            tf = get_tfn(token, document, matrix) if type_tf != "tf" else get_tf(token, document, matrix)
            wij = tf * idf
            weights.loc[token, str(document)] = wij

    time_taken = datetime.now() - start_time
    logging.info(f"Geração do modelo concluída. Tempo decorrido: {time_taken}.")

    return weights


def save_model(path, i_list, type_tf):
    """
    Funcão para salvar o modelo em csv
    :param str i_list: lista invertida gerada
    :param str file: se tf ou n
    """
    logging.basicConfig(filename='../logs/index.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    
    matrix = get_doc_matrix(i_list)
    matrix.to_csv("../result/matrix.csv", sep=";")

    model = get_model(matrix, type_tf)
    model.to_csv(path, sep=";")
    logging.info("Model saved.")