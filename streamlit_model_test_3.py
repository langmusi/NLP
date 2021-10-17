# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:02:29 2021

@author: minli
"""

"""
This Streamlit app is for us as developer to quickly know which models give more reasonable answer outputs, and
a demo for information searching.
"""
import base64
import streamlit as st

import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from rank_bm25 import BM25Okapi
#from sklearn.feature_extraction import stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np


st.title("Information Searching in A Given Defects Context Demo")

embeddings_filepath = "C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/bert-nlp/data/close_defects_small_preprocessed_kblab.csv"

# caching data. Only run once if the data has not been loaded
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv(embeddings_filepath, encoding = 'ISO 8859-1', sep = ";")  # ISO 8859-1
    return df

# Will only run once if already cached
df = load_data()

# filling nan
df[['Skadebeskrivning', 'Skaderubrik', 'Åtgärdsbeskrivning']] = df[['Skadebeskrivning','Skaderubrik', 'Åtgärdsbeskrivning']].fillna(value='')
df['Skade_text'] = df['Skaderubrik'] + ' ' + df['Skadebeskrivning']
# droping nan
df = df[df['Skade_text'].notna()]
# removing multiple blanks
df['Skade_text'] = df['Skade_text'].replace('\s+', ' ', regex=True)
# removing trailing space of column in pandas
df['Skade_text'] = df['Skade_text'].str.rstrip()
passages = df['Skade_text'].values.tolist()

st.write('The whole datasets:', df.shape)
#train_owner = df['Fordonsägare'].drop_duplicates().values.tolist()
#train_owner_all = ['ALL'] + train_owner

#train_owner_option = st.sidebar.selectbox("Train Owner Selection:", train_owner_all)

train_owner = df['Fordonsägare'].unique()
train_owner= np.insert(train_owner, 0, 'All', axis=0)
train_owner_selected = st.sidebar.multiselect("Train Owner Selection:", train_owner)
# Mask to filter dataframe
mask_countries = df['Fordonsägare'].isin(train_owner_selected)

if "All" in train_owner_selected:
    df = df.copy()
else: 
    df = df[mask_countries] 

passages = df['Skade_text'].values.tolist()

st.write("The first rows in the dataset", df.head())
st.write("The size of the selected dataset", df.shape)

top_k = 100     #Number of passages we want to retrieve with the bi-encoder
top_number_output = st.sidebar.slider('Choose a value to show top number of outputs', 5, 20)

query = st.text_input("Please write your question/text here")


st.sidebar.header('Settings')


bi_encoder_option = st.sidebar.selectbox(
    'Bi Encoder Mdoel list',
    ('paraphrase-multilingual-mpnet-base-v2', # model.max_seq_length
      'Gabriel/paraphrase-swe-mpnet-base-v2',
      'Gabriel/paraphrase-multi-mpnet-base-atkins',  # good 
      'KBLab/sentence-bert-swedish-cased',           # good
      'sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned'))



cross_encoder_option = st.sidebar.selectbox(
    'Cross Encoder Mdoel list',
    ("cross-encoder/ms-marco-TinyBERT-L-6", 
      "cross-encoder/quora-roberta-large", 
      "cross-encoder/ms-marco-MiniLM-L-12-v2",
      "cross-encoder/stsb-roberta-large"))


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

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 07:46:09 2021

@author: minli
"""

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

    
    bm25_output = pd.DataFrame([], 
                                columns = ['Likhetsgrad', 'LikadanaSkador', 
                                            'Åtgärdskod', 'Åtgärder', 
                                                    'Skadekategori'])

    print("Top-" + str(top_n_res) + "lexical search (BM25) hits")
    for hit in bm25_hits[0:top_n_res]:
        a = round(hit['score'], 2) 
        b = passages[hit['corpus_id']] 
        c = df.iloc[hit['corpus_id']]['Åtgärdskod']
        d = df.iloc[hit['corpus_id']]['Åtgärdsbeskrivning']
        e = df.iloc[hit['corpus_id']]['Skadekategori']
        
        bm25_output_temp = pd.DataFrame([(a, b, c, d, e)], 
                                        columns = ['Likhetsgrad', 'LikadanaSkador', 
                                                    'Åtgärdskod', 'Åtgärder', 
                                                    'Skadekategori'])
        bm25_output = bm25_output.append(bm25_output_temp)
        
    bm25_output = bm25_output.reset_index(drop=True)

    return bm25_output


# =============================================================================
# ###############
#This function will search all texts in passages that answer the query: ##### Sematic Search #####
def model_semantic_search(query, bi_encoder_name, top_k_biencoder, top_n_res):
    
    bi_encoder = SentenceTransformer(bi_encoder_name)
    top_k = int(top_k_biencoder)     #Number of passages we want to retrieve with the bi-encoder

    #Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
    corpus_embeddings = bi_encoder.encode(passages,  batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    
    
    ##### Sematic Search #####
    #Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    #question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    
    #Output of top-10 hits
    print("Top-" + str(top_n_res) + "Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    
    bi_encoder_output = pd.DataFrame([], 
                                     columns = ['Likhetsgrad', 'LikadanaSkador', 
                                                'Åtgärdskod', 'Åtgärder', 
                                                'Skadekategori'])
    for hit in hits[0:top_n_res]:
        a = round(hit['score'], 2) 
        b = passages[hit['corpus_id']] 
        c = df.iloc[hit['corpus_id']]['Åtgärdskod']
        d = df.iloc[hit['corpus_id']]['Åtgärdsbeskrivning']
        e = df.iloc[hit['corpus_id']]['Skadekategori']
        
        bi_encoder_output_temp = pd.DataFrame([(a, b, c, d, e)], 
                                        columns = ['Likhetsgrad', 'LikadanaSkador', 
                                                    'Åtgärdskod', 'Åtgärder', 
                                                    'Skadekategori'])
        bi_encoder_output = bi_encoder_output.append(bi_encoder_output_temp)
        
    bi_encoder_output = bi_encoder_output.reset_index(drop=True)
    
    return bi_encoder_output


# ##############################
#This function will search all texts in passages that answer the query
def model_semantic_search_rerank(query, bi_encoder_name, cross_encoder_name, top_k_biencoder, top_n_res):
    
    bi_encoder = SentenceTransformer(bi_encoder_name)
    top_k = int(top_k_biencoder)     #Number of passages we want to retrieve with the bi-encoder

    #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    cross_encoder = CrossEncoder(cross_encoder_name)
    
    #Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
    corpus_embeddings = bi_encoder.encode(passages,  batch_size=128, convert_to_tensor=True, show_progress_bar=True)
    
    
    ##### Sematic Search #####
    #Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    #question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    #Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    #Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]


    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    
    
    print("Top-" + str(top_n_res) + "Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    cross_encoder_output = pd.DataFrame([], 
                                        columns = ['Likhetsgrad', 'LikadanaSkador', 
                                                    'Åtgärdskod', 'Åtgärder', 
                                                    'Skadekategori'])
    for hit in hits[0:top_n_res]:
        a = round(hit['cross-score'], 2) 
        b = passages[hit['corpus_id']] 
        c = df.iloc[hit['corpus_id']]['Åtgärdskod']
        d = df.iloc[hit['corpus_id']]['Åtgärdsbeskrivning']
        e = df.iloc[hit['corpus_id']]['Skadekategori']
        
        cross_encoder_output_temp = pd.DataFrame([(a, b, c, d, e)], 
                                        columns = ['Likhetsgrad', 'LikadanaSkador', 
                                                    'Åtgärdskod', 'Åtgärder', 
                                                    'Skadekategori'])
        cross_encoder_output = cross_encoder_output.append(cross_encoder_output_temp)
    
    cross_encoder_output = cross_encoder_output.reset_index(drop=True)
                
    return cross_encoder_output



# downloading the dataframe
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False, sep = ';')

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode("latin1")).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'



#st.write('You selected:', cross_encoder_option)


search_type_option = ("Lexical Search", "Semantic Search", "Semantic Search and Re-ranking")
search_type = st.sidebar.selectbox("Searching Options:" , search_type_option)


if search_type == "Lexical Search":
    st.write(model_lexical_search(query, top_n_res=top_number_output))
    btn_download = st.button("Click to Donwload DataFrame as CSV")
    if btn_download:
        tmp_download_link = download_link(model_lexical_search(query, top_n_res=top_number_output), 
                                      'YOUR_DF.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
elif search_type == "Semantic Search":
    st.write(model_semantic_search(query, 
                              bi_encoder_name = bi_encoder_option, 
                              top_k_biencoder=top_k, top_n_res=top_number_output))
elif search_type == "Semantic Search and Re-ranking":
    st.write(model_semantic_search_rerank(query, 
                      bi_encoder_name = bi_encoder_option, 
                      cross_encoder_name = cross_encoder_option,  
                      top_k_biencoder=top_k, top_n_res=top_number_output))
    btn_download = st.button("Click to Donwload DataFrame as CSV")
    if btn_download:
        tmp_download_link = download_link(model_semantic_search_rerank(query, 
                      bi_encoder_name = bi_encoder_option, 
                      cross_encoder_name = cross_encoder_option,  
                      top_k_biencoder=top_k, top_n_res=top_number_output), 
                                      'YOUR_DF.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    


# # st.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)

