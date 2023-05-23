import logging
import configparser
import pandas as pd
import numpy as np
from datetime import datetime


def get_conf(file):
    """
    Funcão para capturar as configuracoes para a indexador.
    :param str file: nome do arquivo com as configuracoes
    return str model: caminho para o arquivo do modelo 
    return srt queries_file: caminho para o arquivo das consultas 
    return str results: string contendo o caminho do arquivo a ser escrito
    return bool stemmer: flag de stemmer
    """
    logging.basicConfig(filename='../logs/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Iniciando operacao de captura das configuracões.')

    try:
    # Ler arquivo de configuracão
        logging.info('Iniciando leitura do arquivo de configuracao.')
        cfg_parser = configparser.RawConfigParser()
        cfg_parser.readfp(open(file))
        try:
            if cfg_parser.options('STEMMER'):
                stemmer = True
        except:
            stemmer = False
        model = cfg_parser.get('CONF', 'MODELO')
        queries_file = cfg_parser.get('CONF', 'CONSULTAS')
        results = cfg_parser.get('CONF', 'RESULTADOS')

        logging.info('Arquivo de configuracao lido com sucesso.')
        return model, queries_file, results, stemmer
    
    except Exception as e:
        logging.error(f'Ocorreu o seguinte erro ao ler o arquivo de configuracao: {str(e)}')

def get_results(output_file, ranking):
    logging.basicConfig(filename='../logs/busca.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Gerando o arquivo de resultados. Arquivo em formato CSV com 'QueryNumber; [posição no ranking, número do documento, valor de sim_cos]'.")

    with open(output_file, 'w') as results:
        results.write("QueryNumber;DocInfos\n")
        for query in ranking.columns:
            query_number = query.replace('Q', '')
            sorted_ranking = ranking[query].sort_values(ascending=False)
            position_ranking = 1
            for doc_number, cos in sorted_ranking.items():
                if cos == 0:
                    break
                doc_infos = [position_ranking, doc_number, cos]
                position_ranking += 1
                results.write(f"{query_number};{doc_infos}\n")
    
    logging.info("Arquivo de resultados criado.")



def insert_queries(model, queries):
    logging.basicConfig(filename='../logs/busca.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Inserindo consultas no modelo.")
    start_time = datetime.now()


    for qnumber, qtext in queries.itertuples():
        zeros = pd.DataFrame(0, index=model.index, columns=[f"Q{qnumber}"])
        model = pd.concat([model, zeros], axis=1)

        for word in qtext:
            if word not in model.index:
                zeros = pd.DataFrame(0, index=[word], columns=model.columns)
                model = pd.concat([model, zeros], axis=0)

            model.at[word, f"Q{qnumber}"] += 1

    time_taken = datetime.now() - start_time
    logging.info(f"Consultas inseridas no modelo. Tempo decorrido: {time_taken}")

    return model



def get_ranking(model, queries):
    logging.basicConfig(filename='../logs/busca.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Rankeando consultas...")

    model = insert_queries(model, queries)
    ranking = pd.DataFrame(index=model.columns[:-99])

    for query in queries.index:
        q = f"Q{query}"
        ranking[q] = 0.0
        if str(query) in model.index:
            q_vector = model.loc[str(query)].values
            q_norm = np.linalg.norm(q_vector)
            qxd_norm = np.linalg.norm(model.iloc[:, :-99].values, axis=0)
            dot_product = np.dot(q_vector, model.iloc[:, :-99].values)
            ranking[q] = dot_product / (q_norm * qxd_norm)

    return ranking


