import sys
import logging
import configparser
import xml.etree.ElementTree as ET
import os
from datetime import datetime


def get_conf(file='../config/PC.cfg'):
    """
    Funcão para capturar as configuracoes iniciais.
    :param str file: nome do arquivo com as configuracoes
    return strs file_to_read, query, expected_result: strings contendo o caminho do arquivo a ser lido, qual consulta deve ser feita e os resultados esperados
    """
    logging.basicConfig(filename='../logs/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Iniciando operacao de captura das configuracões.')

    file_to_read = ""
    query = ""
    expected_result = ""

    try:
        # Ler arquivo de configuracão
        logging.info('Iniciando leitura do arquivo de configuracao.')
        cfg_parser = configparser.RawConfigParser()
        cfg_parser.readfp(open(file))
        file_to_read = cfg_parser.get('CONF', 'LEIA')
        query = cfg_parser.get('CONF', 'CONSULTAS')
        expected_result = cfg_parser.get('CONF', 'ESPERADOS')
        logging.info('Arquivo de configuracao lido com sucesso.')
    except Exception as e:
        logging.error(f'Ocorreu o seguinte erro ao ler o arquivo de configuracao: {str(e)}')

    return file_to_read, query, expected_result



def et(filepath, tag=None, txt=False):
    """
    Funcão para buscar elementos de uma tag específica.
    A funcão usa o método 'findall' que é case sensitive.
    :param str filepath: Caminho completo do XML que será lido.
    :param str tag: Tag para busca de elementos.
    :param bool txt: Saída sem tags de xml.
    :return list elements: Lista contendo todos os elementos encontrados.
    """
    logging.basicConfig(filename='../logs/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f'Iniciando operacao para buscar elementos no arquivo "{filepath}".')

    filepath = "../data/" + filepath
    tree = None
    root = None

    try:
        # Lendo o arquivo XML
        logging.info('Iniciando leitura do arquivo XML.')
        tree = ET.parse(filepath)
        root = tree.getroot()
        logging.info('Arquivo XML lido com sucesso.')

        # # Buscando os elementos da tag
        # elements_xml = root.findall(f".//{tag}")

        # if txt:
        #     elements = [element.text for element in elements_xml]
        # else:
        #     elements = [ET.tostring(element).decode("utf-8") for element in elements_xml]
    except Exception as e:
        logging.error(f'Ocorreu um erro ao ler o arquivo XML: {str(e)}')

    return root




def get_queries(xml, queries_file_path):
    """
    Funcao para realizar e salvar as consultas feitas em queries_file_path (csv)
    :param xml.etree.ElementTree.Element xml: arquivo xml a ser lido
    :param str queries_file_path: caminho do arquivo para salvar as consultas
    """

    logging.basicConfig(filename='../logs/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f'Iniciando operacao para realizar e salvar as consultas no arquivo "{queries_file_path}".')

    queries_file_path = "../result/" + queries_file_path
    
    queries_read = 0
    times = []

    try:
        with open(queries_file_path, 'w') as queries_file:
            queries_file.write("QueryNumber;QueryText\n")
            
            start_time = datetime.now()

            for query in xml:
                queries_read += 1
                query_number = ""
                query_text = ""
                for element in query:
                    if element.tag == "QueryNumber":
                        query_number = int(element.text)
                    elif element.tag == "QueryText":
                        query_text = element.text.upper()
                        query_text = query_text.replace('\n', '')
                        query_text = query_text.replace(';', '')

                queries_file.write(f"{query_number};{query_text}\n")

                time_taken = datetime.now() - start_time
                times.append(time_taken.total_seconds())

                start_time = datetime.now()

        logging.info(f'Consultas salvas com sucesso no arquivo "{queries_file_path}".')

        logging.info(f'Quantidade de dados lida: {queries_read}')
        logging.info(f'Tempo medio de processamento por consulta: {sum(times) / len(times):.2f} segundos')

    except Exception as e:
        logging.error(f'Ocorreu um erro ao realizar e salvar as consultas: {str(e)}')




def get_expected(xml, expected_result_file_path):
    """
    Funcao para obter e salvar os resultados esperados no arquivo expected_result_file_path (csv)
    :param xml.etree.ElementTree.Element xml: arquivo xml a ser lido
    :param str expected_result_file_path: caminho do arquivo para salvar os resultados esperados
    """

    logging.basicConfig(filename='../logs/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f'Iniciando operacao para obter e salvar os resultados esperados no arquivo "{expected_result_file_path}".')

    expected_result_file_path = "../result/" + expected_result_file_path

    try:
        with open(expected_result_file_path, 'w') as expected_file:
            expected_file.write("QueryNumber;DocNumber;DocVotes\n")

            for query in xml:
                query_number = ""
                for element in query:
                    if element.tag == "QueryNumber":
                        query_number = int(element.text)
                    elif element.tag == "Records":
                        for item in element:
                            doc_number = int(item.text)
                            score = item.attrib['score'].replace('0', '')
                            doc_votes = len(score)
                            expected_file.write(f"{query_number};{doc_number};{doc_votes}\n")

        logging.info(f'Resultados esperados salvos com sucesso no arquivo "{expected_result_file_path}".')

    except Exception as e:
        logging.error(f'Ocorreu um erro ao obter e salvar os resultados esperados: {str(e)}')

        
