#imports
import query_proc
import logging
from datetime import datetime
import list_generator
import index
import finder
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



if __name__ == '__main__':
    normalized = input("O tf é normalizado [ y / n ]? ")
    if normalized.lower() == "y":
        type_tf = "tfn"
    else:
        type_tf = "tf"
        
    logging.basicConfig(filename='../logs/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f'Iniciando execucao do programa em {datetime.now()}".')

    file_to_read, query, expected_result = query_proc.get_conf()
    #print(file_to_read, query, expected_result)
    xml = query_proc.et(file_to_read)
    
    query_proc.get_queries(xml, query)
    query_proc.get_expected(xml, expected_result)

    read_files, write_file, stemmer = list_generator.get_conf("../config/GLI.cfg")

    list_generator.get_all_files(read_files, write_file, stemmer)

    tokens, model = index.get_conf("../config/INDEX.cfg")
    # Gerando modelo através da matriz termo documento que foi construída com a lista invertida
    index.save_model(model, tokens, type_tf)

    model_file, queries_file, results_file, stemmer = finder.get_conf("../config/BUSCA.cfg")
    # Lê o modelo na memória
    model = pd.read_csv(model_file, sep=";")
    model.set_index(["Token"], inplace=True)

    # Lê as consultas na memória
    queries = pd.read_csv(queries_file, sep=";")
    queries.set_index(["QueryNumber"], inplace=True)

    for number, text in queries.itertuples():
        tokens = wordpunct_tokenize(text)
        stop_en = stopwords.words("english")

        processed_text = []
        for word in tokens:
            if word.lower() in stop_en:
                continue
            elif not word.isalpha():
                continue
            elif len(word) < 3:
                continue
            if stemmer:
                stemmer = PorterStemmer()
                word_stemmed = stemmer.stem(word)
                processed_text.append(word_stemmed.upper())
            else:
                processed_text.append(word.upper())
        
        queries.at[number, "QueryText"] = processed_text
    
    ranking = finder.get_ranking(model, queries)
    
    finder.get_results(results_file, ranking)

    logging.info(f'Fim da execucao do programa em {datetime.now()}".')