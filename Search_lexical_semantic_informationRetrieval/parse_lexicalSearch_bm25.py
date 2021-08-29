import pandas as pd
from tqdm.autonotebook import tqdm
import string
import argparse
import numpy as np
import json
from rank_bm25 import BM25Okapi

parser = argparse.ArgumentParser(description='Lexical Search with bm25')
parser.add_argument('--query', type=str, required=True, help='Input text')  # help = description of the argumenent
parser.add_argument('--top_output', type=int, required=True, help='Top n')  # help = description of the argumenent
args = parser.parse_args()


embeddings_filepath = "C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/bert-nlp/data/close_defects_small_preprocessed_kblab.csv"

def load_data():
    df = pd.read_csv(embeddings_filepath, encoding = 'latin-1', sep = ";")
    return df

# Will only run once if already cached
df = load_data()
passages = df['Skade_text'].values.tolist()

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
        c_1 = df.iloc[hit['corpus_id']]['Littera']
        c_2 = df.iloc[hit['corpus_id']]['Fordon']
        c = df.iloc[hit['corpus_id']]['Åtgärdskod']
        d = df.iloc[hit['corpus_id']]['Åtgärdsbeskrivning']
        d_1 = df.iloc[hit['corpus_id']]['Skadenummer']
        d_2 = df.iloc[hit['corpus_id']]['Skadedatum']
        e = df.iloc[hit['corpus_id']]['Skadekategori']
        
        bm25_output_temp = pd.DataFrame([(a, b, c_1, c_2, c, d, d_1, d_2, e)], 
                                        columns = ['Likhetsgrad', 'LikadanaSkador', 'Littera', 'Fordon',
                                                    'Atgardskod', 'Atgarder', 'Skadenummer', 'Skadedatum',
                                                    'Skadekategori'])
        bm25_output = bm25_output.append(bm25_output_temp)
        
    bm25_output = bm25_output.reset_index(drop=True)
    
    return bm25_output

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
if __name__ == '__main__':
  bm25_output = model_lexical_search(args.query, args.top_output)
  output_json = bm25_output.to_json(orient='table')  # index GOOD split
  parsed=json.loads(output_json)
  print(json.dumps(parsed, indent=2))
