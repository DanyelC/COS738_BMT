import logging
import configparser
import numpy as np
from datetime import datetime,timedelta
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from xml.etree import ElementTree as ET 

def get_conf(file='../config/GLI.cfg'):
    """
    Funcão para capturar as configuracoes para a lista invertida.
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
        try:
            if cfg_parser.options('STEMMER'):
                stemmer = True
        except:
            stemmer = False
        files_to_read = cfg_parser.options('CONF')
        files_to_read = [cfg_parser.get('CONF', instruction) for instruction in files_to_read if instruction.startswith('LEIA'.lower())]
        file_to_write = cfg_parser.get('CONF', 'ESCREVA')

        logging.info('Arquivo de configuracao lido com sucesso.')
        return files_to_read, file_to_write, stemmer
    
    except Exception as e:
        logging.error(f'Ocorreu o seguinte erro ao ler o arquivo de configuracao: {str(e)}')



def frequency(text, record_num, stemmer):
    tokens = wordpunct_tokenize(text)
    stop_en = set(stopwords.words("english"))
    stemmer = PorterStemmer() if stemmer else None
    frequency_dict = {}

    for word in tokens:
        word = word.lower()
        if word in stop_en or not word.isalpha() or len(word) < 3:
            continue
        if stemmer:
            word = stemmer.stem(word)
        word = word.upper()
        if word in frequency_dict:
            frequency_dict[word].append(record_num)
        else:
            frequency_dict[word] = [record_num]

    return frequency_dict


def get_inverted_list(read_files, stemmer):
    logging.basicConfig(filename='../logs/gli.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Iniciando criação da lista invertida")
    inverted_list = {}
    total_files = len(read_files)
    times = []

    for file in read_files:
        start_time = datetime.now()
        xml_file = ET.parse(f"../data/{file}")
        xml_root = xml_file.getroot()

        file_records = {}

        for record in xml_root:
            record_num = int(record.find("RECORDNUM").text)
            text = record.find("ABSTRACT").text.upper() if record.find("ABSTRACT") is not None else ""
            file_records[record_num] = text

        for record_num, text in file_records.items():
            record_dict = frequency(text, record_num, stemmer)
            for token, record_list in record_dict.items():
                inverted_list.setdefault(token, []).extend(record_list)

        time_taken = datetime.now() - start_time
        times.append(time_taken)

    mean_time = sum(times, timedelta(seconds=0)) / len(times)
    logging.info(f"{total_files} arquivos processados em média {mean_time.total_seconds()}s.")
    logging.info("Finalizando criação da lista invertida")

    return inverted_list


def get_all_files(read_files, path, stemmer):
    """
    Função para criar o arquivo de escrita.
    :param read_files: nome dos arquivos a serem lidos
    :param path: caminho do arquivo de escrita
    """
    logging.basicConfig(filename='../logs/gli.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Iniciando a criação do arquivo da lista invertida.")

    inverted_list = get_inverted_list(read_files, stemmer)
    tokens = inverted_list.keys()

    with open(path, 'w') as w_file:
        w_file.write("Token;Aparicao\n")
        for token in tokens:
            w_file.write(f"{token};{inverted_list[token]}\n")

    logging.info("Finalizada a criação do arquivo da lista invertida.")


