'''
This script can be run from a terminal, e.g Python prompt terminal
# to run this script in a Python prompt terminal:
# python "C:\Users\LIUM3478\OneDrive Corp\OneDrive - Atkins Ltd\Work_Atkins\textanalys\Test\azure_ml_pipeline_test\publish_run_irrr\information_retrieval_re_rank.py" 
# --query "toalet belysningsfel" --top_out 5
# The files in this script are stored in the same folder
'''

import pandas as pd
import numpy as np

import argparse
import json

#from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util

parser = argparse.ArgumentParser(description='Information Retrieval and Re-Rank')
parser.add_argument('--query', type=str, required=True, help='Input text')  # help = description of the argumenent
parser.add_argument('--top_output', type=int, required=True, help='Top n')  # help = description of the argumenent
args = parser.parse_args()


#from sentence_transformers import SentenceTransformer, CrossEncoder, util
#Users/cla.Min.Liu/NLP/IR_RR/pipeline/src/close_defects_small_preprocessed_kblab.csv
embeddings_filepath = 'close_defects_small_preprocessed_kblab.csv' 

df = pd.read_csv(embeddings_filepath, encoding = 'ISO 8859-1', sep = ";")
df.drop_duplicates(inplace=True)

passages = df['Skade_text'].values.tolist()



#This function will search all texts in passages that answer the query: ##### Sematic Search #####
def info_retrieval_rerank(query, top_n_res,
                         bi_encoder_name='Gabriel/kb-finetune-atkins', 
                         cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-12-v2', top_k_biencoder=100):
    
    bi_encoder = SentenceTransformer(bi_encoder_name, device='cuda')  # , device='cuda'
    top_k = int(top_k_biencoder)     #Number of passages we want to retrieve with the bi-encoder
    
    #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    cross_encoder = CrossEncoder(cross_encoder_name, device='cuda')

    #Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
    corpus_embeddings = bi_encoder.encode(passages, device='cuda', batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    
    
    ##### Sematic Search #####
    #Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, device='cuda', convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    #Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    #Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]
    #Output of top-10 hits
    print("Top" + " Number of Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    
    cross_encoder_output = pd.DataFrame([])

    for hit in hits[0:top_n_res]:
        a = round(hit['score'], 2) 
        b = df.iloc[hit['corpus_id']]['Littera']
        c = df.iloc[hit['corpus_id']]['Fordon']
        d_1 = df.iloc[hit['corpus_id']]['Skadenummer']
        d = df.iloc[hit['corpus_id']]['Skaderubrik']
        e = df.iloc[hit['corpus_id']]['Skadebeskrivning']
        f = df.iloc[hit['corpus_id']]['Åtgärdskod']
        g = df.iloc[hit['corpus_id']]['Åtgärdsbeskrivning']
        h = df.iloc[hit['corpus_id']]['Skadedatum']
        
        cross_encoder_output_temp = pd.DataFrame([(a, b, c, d_1, d, e, f, g, h)], 
                                        columns = ['Likhetsgrad', 'Littera', 'Fordon', 'Skadenummer',
                                                   'Skaderubrik', 'Skadebeskrivning',
                                                    'Åtgärdskod', 'Åtgärder', 
                                                    'Skadedatum'])
        cross_encoder_output = cross_encoder_output.append(cross_encoder_output_temp)
        
    cross_encoder_output = cross_encoder_output.reset_index(drop=True)
    
    return cross_encoder_output


if __name__ == '__main__':
  output = info_retrieval_rerank(args.query, args.top_output)
  output_json = output.to_json(orient='index', force_ascii=False)  # index GOOD split
  parsed=json.loads(output_json)
  print(json.dumps(parsed, indent=4, sort_keys=True, ensure_ascii=False))