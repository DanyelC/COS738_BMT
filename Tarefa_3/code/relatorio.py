import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

st.title("Recuperação de Informação - T3")

st.write("Este trabalho tem como objetivo implementar um sistema de recuperação de informação, utilizando o modelo vetorial, para a base de dados da Tarefa 2. O sistema deve receber uma consulta e retornar os documentos mais relevantes para a consulta. Para isso, o sistema deve utilizar o modelo vetorial para representar os documentos e a consulta, e o cosseno do ângulo entre os vetores de consulta e documento para determinar a relevância dos documentos para a consulta. O sistema deve retornar os 10 documentos mais relevantes para a consulta, ordenados por ordem de relevância.")

salvar_imagens = st.checkbox("Salvar imagens")

st._transparent_write("___")

#verificar se os datasets estão carregados e, senao, salvar no state do streamlit
if "expected" not in st.session_state:
    st.session_state.expected = pd.read_csv("../result/expected_results.csv", sep=";")

if "results" not in st.session_state:
    st.session_state.results = pd.read_csv("../result/results_stemmer.csv", sep=";", converters={"DocInfos": pd.eval})

if "results_no_stemmer" not in st.session_state:
    st.session_state.results_no_stemmer = pd.read_csv("../result/results_nostemmer.csv", sep=";", converters={"DocInfos": pd.eval})


def get_expected_docs(expected_file_df):
    query_docs = {}

    for query_number, group in expected_file_df.groupby("QueryNumber"):
        docs = list(group["DocNumber"])
        query_docs[query_number] = docs

    return query_docs


def get_results_docs(results_file_df):
    results_docs = {}

    for query_number, group in results_file_df.groupby("QueryNumber"):
        docs = [int(single_result[1]) for single_result in group["DocInfos"]]
        results_docs[query_number] = docs

    return results_docs


def precision(list_correct, list_return):
    total_return = len(list_return)
    total_correct = len(set(list_correct) & set(list_return))
    
    return (total_correct / total_return) * 100


def recall(doc_number, list_docs):
    if doc_number in list_docs:
        return 100 / len(list_docs)
    return 0.0


def pr_curve(expected_docs, results_docs, query_number):
    try:
        correct_docs = expected_docs[query_number]
        returned_docs = results_docs[query_number]
    except KeyError:
        return pd.DataFrame([], columns=["DocNumber", "Recall", "Precision"])
    plot_table = pd.DataFrame([], columns=["DocNumber", "Recall", "Precision"])
    r = 0.0

    for index, doc_number in enumerate(returned_docs):
        recall_value = recall(doc_number, correct_docs)
        if recall_value == 0.0:
            continue

        r += recall_value
        p = precision(correct_docs, returned_docs[:index + 1])
        plot_table.loc[index] = [int(doc_number), r, p]

    return plot_table


def create_eleven_points_table(expected_docs, results_docs):
    table = pd.DataFrame({"Recall": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    query_numbers = list(expected_docs.keys())

    for query_number in query_numbers:
        if query_number not in results_docs:
            continue

        eleven_points = []
        curve = pr_curve(expected_docs, results_docs, query_number)

        for r in range(0, 101, 10):
            precision_value = curve.loc[curve["Recall"] > r]["Precision"].max()
            eleven_points.append(precision_value)

        table[f"pQ{query_number}"] = eleven_points

    table.fillna(0, inplace=True)

    return table


def add_mean_precision(eleven_points_df):
    pi = []
    copy_11 = eleven_points_df.set_index("Recall")

    for i in range(11):
        precision_mean = copy_11.iloc[i].mean()
        pi.append(precision_mean)

    eleven_points_df["pi"] = pi

    return eleven_points_df


def calculate_recall(expected_docs, results_docs, query_number):
    try:
        relevant_docs = expected_docs[query_number]
        retrieved_docs = results_docs[query_number]
    except:
        return 0.0

    relevant_retrieved = len(set(relevant_docs) & set(retrieved_docs))
    relevant_total = len(relevant_docs)
    recall = relevant_retrieved / relevant_total if relevant_total > 0 else 0.0

    return recall


def calculate_precision(expected_docs, results_docs, query_number):
    try:
        relevant_docs = expected_docs[query_number]
        retrieved_docs = results_docs[query_number]
    except:
        return 0.0

    relevant_retrieved = len(set(relevant_docs) & set(retrieved_docs))
    retrieved_total = len(retrieved_docs)
    precision = relevant_retrieved / retrieved_total if retrieved_total > 0 else 0.0

    return precision


def calculate_f1_score(precision, recall):
    try:
        return (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return 0.0


def calculate_f1_table(expected_docs, results_docs):
    f1_table = pd.DataFrame([], columns=["F1", "QueryNumber"])
    query_numbers = list(expected_docs.keys())

    for index, query_number in enumerate(query_numbers):
        recall = calculate_recall(expected_docs, results_docs, query_number)
        precision = calculate_precision(expected_docs, results_docs, query_number)
        f1 = calculate_f1_score(precision, recall)
        f1_table.loc[index] = [f1, query_number]

    return f1_table


def calculate_avg_f1_score(f1_table):
    return f1_table["F1"].sum() / len(f1_table["F1"])


def calculate_precision_at_k(k, expected_docs, results_docs, query_number):
    try:
        correct_list = expected_docs[query_number]
        returned_list = results_docs[query_number]
    except KeyError:
        return 0.0

    return precision(correct_list, returned_list[:k])


def calculate_p_at_k_table(k, expected_docs, results_docs):
    pk_table = pd.DataFrame([], columns=[f"P@{k}", "QueryNumber"])
    query_numbers = list(expected_docs.keys())

    for index, query_number in enumerate(query_numbers):
        pk = calculate_precision_at_k(k, expected_docs, results_docs, query_number)
        pk_table.loc[index] = [pk, query_number]

    return pk_table


def calculate_avg_p_at_k(k, pk_table):
    return pk_table[f"P@{k}"].sum() / len(pk_table[f"P@{k}"])


def calculate_r_precision(expected_docs, results_docs, query_number):
    try:
        relevant_docs = expected_docs[query_number]
        retrieved_docs = results_docs[query_number]
    except KeyError:
        return 0.0

    relevant_total = len(relevant_docs)
    total = relevant_total
    retrieved_total = len(retrieved_docs)
    if relevant_total > retrieved_total:
        total = retrieved_total

    correct_retrieved = 0
    for i in range(total):
        if retrieved_docs[i] in relevant_docs:
            correct_retrieved += 1

    return correct_retrieved / relevant_total * 100


def calculate_r_precision_table(expected_docs, results_docs):
    r_precision_table = pd.DataFrame([], columns=[f"R-Precision", "QueryNumber"])
    query_numbers = list(expected_docs.keys())

    for index, query_number in enumerate(query_numbers):
        rp = calculate_r_precision(expected_docs, results_docs, query_number)
        r_precision_table.loc[index] = [rp, query_number]

    return r_precision_table


def calculate_mvp(expected_docs, results_docs, query_number):
    try:
        relevant_docs = expected_docs[query_number]
        retrieved_docs = results_docs[query_number]
    except KeyError:
        return 0.0

    precisions = []
    for index, doc_number in enumerate(retrieved_docs):
        if doc_number in relevant_docs:
            p = precision(relevant_docs, retrieved_docs[:index+1])
            precisions.append(p)

    while len(precisions) < len(relevant_docs):
        precisions.append(0.0)

    total = sum(precisions)
    mvp = total / len(precisions)
    
    return mvp


def calculate_mvp_table(expected_docs, results_docs):
    query_numbers = list(expected_docs.keys())
    mvp_table = pd.DataFrame(columns=["MVP", "QueryNumber"])

    for query_number in query_numbers:
        mvp = calculate_mvp(expected_docs, results_docs, query_number)
        mvp_table = mvp_table.append({"MVP": mvp, "QueryNumber": query_number}, ignore_index=True)
    
    return mvp_table


def calculate_map(mvp_table):
    return mvp_table["MVP"].sum() / len(mvp_table["MVP"])


def calculate_rr(expected_docs, results_docs, query_number, k=10):
    try:
        relevant_docs = expected_docs[query_number]
        retrieved_docs = results_docs[query_number][:k]
    except KeyError:
        return 0.0

    for position, doc_number in enumerate(retrieved_docs):
        if doc_number in relevant_docs:
            return 1 / (position + 1) * 100
    
    return 0.0

def calculate_rr_table(expected_docs, results_docs):
    query_numbers = list(expected_docs.keys())
    rr_table = pd.DataFrame(columns=["RR", "QueryNumber"])

    for query_number in query_numbers:
        rr = calculate_rr(expected_docs, results_docs, query_number)
        rr_table = rr_table.append({"RR": rr, "QueryNumber": query_number}, ignore_index=True)
    
    return rr_table


def calculate_mrr(rr_table):
    return rr_table["RR"].sum() / len(rr_table["RR"])


def get_gain(query_number, doc_number):
    expected_query = st.session_state.expected.loc[st.session_state.expected["QueryNumber"] == query_number]
    expected_doc = expected_query.loc[st.session_state.expected["DocNumber"] == doc_number]
    gain = expected_doc["DocVotes"]
    
    if gain.empty:
        return 0
    else:
        return gain.iloc[0]

def calculate_dcg_table(expected_docs, results_docs):
    query_numbers = list(expected_docs.keys())
    
    dcg_table = pd.DataFrame([], columns=["QueryNumber"])
    
    for i in range(1, 11):
        dcg_table[i] = 0
    
    for index, query_number in enumerate(query_numbers):
        dcg = calculate_dcg(results_docs, query_number)
        dcg_table.loc[index] = [query_number, *dcg]
    
    return dcg_table

def calculate_dcg(results_docs, query_number):
    k = 10
    
    try:
        retrieved_docs = results_docs[query_number][:k]
    except KeyError:
        return [0] * k
    
    cg_vector = [get_gain(query_number, doc_number) for doc_number in retrieved_docs]
    dcg = [cg_vector[0]]
    
    for i in range(1, k):
        try:
            dcg_i = dcg[i-1] + (cg_vector[i] / math.log2(i + 1))
        except IndexError:
            dcg_i = dcg[i-1]
        dcg.append(dcg_i)
    
    return dcg

def add_mean_dcg(dcg_table):
    mean_dcg_table = dcg_table.copy()
    means = dcg_table.iloc[:, 1:].mean()
    
    mean_dcg_table.loc[len(mean_dcg_table)] = [0] + means.tolist()
    
    return mean_dcg_table


def get_ndcg(results_docs, query_number):
    k = 10
    
    try:
        retrieved_docs = results_docs[query_number][:k]
    except KeyError:
        return [0] * k
    
    cg_vector = [get_gain(query_number, doc_number) for doc_number in retrieved_docs]
    cg_vector.sort(reverse=True)
    
    best_dcg = [cg_vector[0]]
    for i in range(1, k):
        try:
            dcg_i = best_dcg[i-1] + (cg_vector[i] / math.log2(i + 1))
        except IndexError:
            dcg_i = best_dcg[i-1]
        best_dcg.append(dcg_i)
    
    dcg = calculate_dcg(results_docs, query_number)
    
    ndcg = []
    for i in range(k):
        if best_dcg[i] != 0:
            ndcg_i = dcg[i] / best_dcg[i]
        else:
            ndcg_i = 0
        ndcg.append(ndcg_i)
    
    return ndcg

def calculate_ndcg_table(expected_docs, results_docs):
    query_numbers = list(expected_docs.keys())
    
    ndcg_table = pd.DataFrame([], columns=["QueryNumber"])
    
    for i in range(1, 11):
        ndcg_table[i] = 0
    
    for index, query_number in enumerate(query_numbers):
        ndcg = get_ndcg(results_docs, query_number)
        ndcg_table.loc[index] = [query_number, *ndcg]
    
    return ndcg_table


def add_mean_ndcg(ndcg_table):
    mean_ndcg_table = ndcg_table.copy()
    means = ndcg_table.iloc[:, 1:].mean()
    
    mean_ndcg_table.loc[len(mean_ndcg_table)] = [0] + means.tolist()
    
    return mean_ndcg_table



ex_docs = get_expected_docs(st.session_state.expected)
res_docs = get_results_docs(st.session_state.results)

#st.text("Recall e Precision para resultados com stemming")
with st.expander("Recall e Precision para resultados com stemming"):
    st.table(pr_curve(ex_docs, res_docs, 90))




success = False
while not success:
    query_number_random = random.choice(list(ex_docs.keys()))
    if query_number_random in res_docs and query_number_random in ex_docs:
        success = True


fig = plt.figure(figsize=(6, 6))
plt.plot(pr_curve(ex_docs, res_docs, query_number_random)["Recall"], pr_curve(ex_docs, res_docs, query_number_random)["Precision"], linewidth=1)
plt.axis([0, 100, 0, 120])
plt.grid(axis="y")
plt.grid(axis="x")
plt.title(f"Precison X Recall para uma query escolhida de forma aleatória ({query_number_random})")
plt.xlabel("Recall (%)")
plt.ylabel("Precision (%)")
plt.yticks([0, 20, 40, 60, 80, 100])

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.pyplot(fig)


r = calculate_recall(ex_docs, res_docs, query_number_random)
p = calculate_precision(ex_docs, res_docs, query_number_random)

columns = st.columns(2)

with columns[0]:
    st.subheader("Recall")
    st.write(f"{r:.4f}")

with columns[1]:
    st.subheader("Precision")
    st.write(f"{p:.4f}")

st._transparent_write("___")

st.subheader("Recall 11 pontos")

eleven_points_df = create_eleven_points_table(ex_docs, res_docs)
with st.expander("11 pontos para resultados com stemming"):
    st.write(eleven_points_df)


eleven_points_df = add_mean_precision(eleven_points_df)
eleven_points_df.to_csv("../data/11pontos-stemmer.csv")
#st.write(eleven_points_df)


fig2 = plt.figure()

plt.plot(eleven_points_df["Recall"], eleven_points_df["pi"], linewidth=1)

plt.axis([0, 100, 0, 120])
plt.grid(axis="y")
plt.grid(axis="x")

plt.title("Recall de 11 pontos")
plt.xlabel("Recall (%)")
plt.ylabel("Precision (%)")
plt.yticks([0, 20, 40, 60, 80, 100])

if salvar_imagens:
    plt.savefig("../data/11pontos-stemmer.png")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.pyplot(fig2)


st._transparent_write("___")

st.subheader("F1 Score para resultados com stemming")


f1 = calculate_f1_table(ex_docs, res_docs)
#st.write(f1)
f1.to_csv("../data/f1-stemmer-3.csv")

fig3 = plt.figure()

plt.scatter(f1["QueryNumber"], f1["F1"])

plt.axis([0, 100, 0, 0.5])
plt.grid(axis="y", linestyle = '--', linewidth=0.5)

plt.title("F1 Score for all queries")
plt.xlabel("Query Number")
plt.ylabel("F1 Score")


avg_f1 = calculate_avg_f1_score(f1)

fig4 = plt.figure()

plt.scatter(f1["QueryNumber"], f1["F1"])
plt.axhline(avg_f1, color="red", label="Average F1 Score")

plt.axis([0, 100, 0, 0.5])
plt.grid(axis="y")
plt.grid(axis="x")

plt.title("F1 Score for all queries")
plt.xlabel("Query Number")
plt.ylabel("F1 Score")
plt.yticks([0, avg_f1, 0.2, 0.3, 0.4, 0.5])
plt.legend()

if salvar_imagens:
    plt.savefig("../data/f1-stemmer-3.png")
#st.pyplot(fig4) 

columns_2 = st.columns(2)

with columns_2[0]:
    st.subheader("Recall")
    st.pyplot(fig3)

with columns_2[1]:
    st.subheader("Precision")
    st.pyplot(fig4)


st._transparent_write("___")

st.subheader("Precision@5 e Precision@10 para resultados com stemming")

with st.expander("Visualizar tabela com Precision@5"):
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        st.table(calculate_p_at_k_table(5, ex_docs, res_docs))

p5 = calculate_p_at_k_table(5, ex_docs, res_docs)
p10 = calculate_p_at_k_table(10, ex_docs, res_docs)

p5.to_csv("../data/precision@5-stemmer-5.csv")
p10.to_csv("../data/precision@10-stemmer-7.csv")



fig5 =  plt.figure()

plt.scatter(p5["QueryNumber"], p5["P@5"])
plt.axhline(calculate_avg_p_at_k(5, p5), color="red", label="Average Precision@5")

plt.axis([0, 100, 0, 120])
plt.grid(axis="y")
plt.grid(axis="x")

plt.title("Precision@5 for all queries")
plt.xlabel("Query Number")
plt.ylabel("Precision (%)")
plt.yticks([0, 20, 40, calculate_avg_p_at_k(5, p5), 60, 80, 100])
plt.legend()

if salvar_imagens:
    plt.savefig("../data/precision@5-stemmer-5.png")
#plt.show()
#st.pyplot(fig5)

fig6 = plt.figure()

plt.scatter(p10["QueryNumber"], p10["P@10"])
plt.axhline(calculate_avg_p_at_k(10, p10), color = 'red', label="Average Precision@10")

plt.axis([0, 100, 0, 120])
plt.grid(axis="y")
plt.grid(axis="x")

plt.title("Precision@10 para todas as consultas")
plt.xlabel("Número da Consulta")
plt.ylabel("Precisão(%)")
plt.yticks([0, 20, 40, calculate_avg_p_at_k(10, p10), 60, 80, 100])
plt.legend()

if salvar_imagens:
    plt.savefig("../data/precision@10-stemmer-7.png")

#st.pyplot(fig6)

columns_2 = st.columns(2)

with columns_2[0]:
    st.subheader("Precisãon@5")
    st.pyplot(fig5)

with columns_2[1]:
    st.subheader("Precisãon@10")
    st.pyplot(fig6)


st._transparent_write("___")

st.subheader("R-Precision para resultados com stemming e sem stemming")
r_precision_stemmer = calculate_r_precision_table(ex_docs, res_docs)
#st.write(r_precision_stemmer.head())
results_nostemmer = pd.read_csv("../result/results_nostemmer.csv", sep=";", converters={"DocInfos": pd.eval})
nostemmer_res_docs = get_results_docs(results_nostemmer)
r_precision_nostemmer = calculate_r_precision_table(ex_docs, nostemmer_res_docs)
#st.write(r_precision_nostemmer.head())

columns_2 = st.columns(2)

with columns_2[0]:
    st.subheader("Stemming")
    st.write(r_precision_stemmer.head())

with columns_2[1]:
    st.subheader("Sem Stemming")
    st.write(r_precision_nostemmer.head())



# Crie uma figura e um conjunto de eixos
fig7, ax = plt.subplots()

# Defina os valores x (números de consulta)
x = r_precision_stemmer["QueryNumber"]

# Calcule a diferença entre os valores de R-Precision com e sem stemming
diff_r_precision = r_precision_stemmer["R-Precision"] - r_precision_nostemmer["R-Precision"]

# Plote a diferença de R-Precision como um gráfico de linha
ax.plot(x, diff_r_precision, marker="o")
ax.grid(axis="y")
ax.grid(axis="x")

# Adicione rótulos e títulos
ax.set_xlabel("Número da Consulta")
ax.set_ylabel("Diferença de R-Precision")
ax.set_title("Diferença de R-Precision entre Stemming e Sem Stemming")

if salvar_imagens: 
    plt.savefig("../data/r-precision-comparativo-9.png")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.pyplot(fig7)

st._transparent_write("___")

st.subheader("MAP para resultados obtidos")

#===================================================================================================

mvp_table = calculate_mvp_table(ex_docs, res_docs)
map_value = calculate_map(mvp_table)

fig8, ax = plt.subplots()

ax.scatter(mvp_table["QueryNumber"], mvp_table["MVP"])
ax.axhline(map_value, color = 'red', label="MAP (Mean Average Precision)")

ax.set_xlim([0, 100])
ax.set_ylim([0, 70])
ax.grid(axis="y", linewidth=0.5)

ax.set_title("MVP para todas as consultas")
ax.set_xlabel("Número da Consulta")
ax.set_ylabel("Precisão (%)")
ax.set_yticks([0, 20, map_value, 40, 60])
ax.legend()

if salvar_imagens:
    plt.savefig("../data/map-stemmer-11.png")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.pyplot(fig8)


st._transparent_write("___")

st.subheader("MRR para resultados obtidos")

rr_table = calculate_rr_table(ex_docs, res_docs)
mrr_value = calculate_mrr(rr_table)

fig9, ax = plt.subplots()

ax.scatter(rr_table["QueryNumber"], rr_table["RR"])
ax.axhline(mrr_value, color = 'red', label="MRR (Mean Reciprocal Rank)")

ax.set_xlim([0, 100])
ax.set_ylim([0, 120])
ax.grid(axis="y", linestyle='--', linewidth=0.5)

ax.set_title("RR para todas as consultas")
ax.set_xlabel("Número da Consulta")
ax.set_ylabel("Precisão (%)")
ax.set_yticks([0, 20, 40, 60, mrr_value, 100])
ax.legend()

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.pyplot(fig9)



if salvar_imagens:
    plt.savefig("../data/mrr-stemmer-12.png")

st._transparent_write("___")

st.subheader("Vetor DCG médio")

dcg_table = calculate_dcg_table(ex_docs, res_docs)
mean_dcg_table = add_mean_dcg(dcg_table)

fig10 = plt.figure(figsize=(8, 8))
plt.plot(mean_dcg_table.T[99][1:], marker="o")
plt.axis([0, 11, 0, 10])
plt.grid(axis="y")
plt.grid(axis="x")
plt.title("Média do DCG para cada posição em todas as consultas")
plt.xlabel("DCG(i)")
plt.ylabel("Média do DCG")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.yticks([0, 2, 4, 6, 8, 10])

columns = st.columns([1, 4, 1])
with columns[1]:
    st.pyplot(fig10)


st._transparent_write("___")

st.subheader("NDCG para resultados obtidos")

ndcg_table = calculate_ndcg_table(ex_docs, res_docs)
mean_ndcg_table = add_mean_ndcg(ndcg_table)

fig11 = plt.figure()

plt.plot(mean_ndcg_table.T[99][1:],marker="o")

plt.axis([0, 11, 0, 1.0])
plt.grid(axis="y")
plt.grid(axis="x")

plt.title("Média do NDCG para cada posição em todas as consultas")
plt.xlabel("NDCG(i)")
plt.ylabel("Média do NDCG")
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

if salvar_imagens:
    plt.savefig("../data/ndcg-stemmer-16.png")

columns = st.columns([1, 4, 1])
with columns[1]:
    st.pyplot(fig11)

st._transparent_write("___")

st.write("""Os resultados obtidos com o uso do Stemmer foram melhores do que os resultados obtidos sem o uso do Stemmer. Isso pode ser observado nos gráficos de R-Precision, MAP e MRR. O gráfico de R-Precision mostra que o uso do Stemmer aumentou a precisão para todas as consultas. O gráfico de MAP mostra que o uso do Stemmer aumentou a precisão média para todas as consultas. O gráfico de MRR mostra que o uso do Stemmer aumentou o valor médio do Reciprocal Rank para todas as consultas. O gráfico de DCG mostra que o uso do Stemmer aumentou o valor médio do DCG para todas as consultas.
        Entretanto, ambos os modelos não tiveram uma precisão satisfatória.""")