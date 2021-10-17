# https://medium.com/datamindedbe/ml-pipelines-in-azure-machine-learning-studio-the-right-way-26b1f79db3f8
# https://towardsdatascience.com/azure-machine-learning-service-run-python-script-experiment-1a9b2fc1b550
# https://docs.microsoft.com/sv-se/azure/machine-learning/tutorial-pipeline-batch-scoring-classification


import pandas as pd
import numpy as np

#from sklearn.feature_extraction import stop_words
import string
from tqdm.autonotebook import tqdm


from rank_bm25 import BM25Okapi
import torch

#import joblib
from azureml.core import Run
run = Run.get_context()

#from sentence_transformers import SentenceTransformer, CrossEncoder, util
#Users/cla.Min.Liu/NLP/IR_RR/pipeline/src/close_defects_small_preprocessed_kblab.csv
embeddings_filepath = 'close_defects_small_preprocessed_kblab.csv' 

df = pd.read_csv(embeddings_filepath, encoding = 'ISO 8859-1', sep = ";")
df.drop_duplicates(inplace=True)

passages = df['Skade_text'].values.tolist()


# We lower case our text and remove stop-words from indexing
def bm25_tokenizer(text):
  tokenized_doc = []
  for token in text.lower().split():
    token = token.strip(string.punctuation)

    #if len(token) > 0 and token not in stop_words.ENGLISH_STOP_WORDS:
    tokenized_doc.append(token)
  return tokenized_doc

tokenized_corpus = []

for passage in tqdm(passages):
  tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)

# Lexical Search
#This function will use lexical search all texts in passages that answer the query
def model_lexical_search(query, top_n_res):
  #BM25 search (lexical search)
  tokenized_corpus = []
  for passage in tqdm(passages):
    tokenized_corpus.append(bm25_tokenizer(passage))

  bm25 = BM25Okapi(tokenized_corpus)
  bm25_scores = bm25.get_scores(bm25_tokenizer(query))
  top_n_res = int(top_n_res)
  top_n = np.argpartition(bm25_scores, -top_n_res)[-top_n_res:]
  bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
  bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    
  bm25_output = pd.DataFrame([])

  print("Top-" + str(top_n_res) + "lexical search (BM25) hits")
  for hit in bm25_hits[0:top_n_res]:
    a = round(hit['score'], 2) 
    b = passages[hit['corpus_id']] 
    c = df.iloc[hit['corpus_id']]['Littera']
    d = df.iloc[hit['corpus_id']]['Fordon']
    e =  df.iloc[hit['corpus_id']]['Åtgärdsdatum']
    f = df.iloc[hit['corpus_id']]['Åtgärdskod']
    g = df.iloc[hit['corpus_id']]['Åtgärdsbeskrivning']
    h = df.iloc[hit['corpus_id']]['Skadedatum']
        
        
    bm25_output_temp = pd.DataFrame([(a, b, c, d, e, f, g, h)], 
                                    columns = ['Likhetsgrad', 'LikadanaSkador', 'Littera', 'Fordon',
                                                'Åtgärdsdatum', 'Åtgärdskod', 'Åtgärder', 
                                                'Skadedatum'])
    bm25_output = bm25_output.append(bm25_output_temp)
        
  bm25_output = bm25_output.reset_index(drop=True)

  return bm25_output

# reading in query from a file
with open("query_input.txt", "r") as text_file:
    query = text_file.readlines()

query = ''.join(query)
run.log("Input_query", query)

#query = "TOA AVSTÄNGD STOP I HANDFAT"
top_number_output = 10
run.log("Number of top output", str(top_number_output))

output = model_lexical_search(query, top_n_res=top_number_output)
run.log_list("top Lexical search result - Littera", output['Littera'])
run.log_list("top Lexical search result - Fordon", output['Fordon'])
run.log_list("top Lexical search result - LikadanaSkador", output['LikadanaSkador'])
run.log_list("top Lexical search result - Åtgärder", output['Åtgärder'])
run.log_list("top Lexical search result - Skadedatum", str(output['Skadedatum']))
# run.log_table("Output", output)  # not working
#joblib.dump(output, "lexicalSearch.pkl")  # not working 

