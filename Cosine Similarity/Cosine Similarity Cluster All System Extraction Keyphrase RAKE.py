#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
path = "C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/nlp"
os.chdir(path)

import pandas as pd
import numpy as np

import fasttext

import Import_and_clean_data as ic
import WCS_Clustering_MainFunction as cmf
import SlimWeightedVecData as swd
import OutputWarranty as ow
import WarrantyCluster as wc


# installing RAKE Rapid Automatic Keyword Extraction
# !pip install python-rake==1.4.4
import RAKE
import operator


# In[ ]:


#!pip install seaborn


# In[2]:


# showing multiple outputs in one cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


df_path = "C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/otu sap wcs 2020 12/system_text_df.csv"
stopword_path = "C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/nlp/danska-stopwords.csv"
fasttext_link = "cc.da.300.bin"


# In[4]:


# here Text-variable is created for analysis
df = ic.ImportAndCleanCSV(df_path, datenc = "ISO 8859-1", text_multi_var = False, text_var = "Beskrivelse")
df.head()
len(df)


# ## Obtaining data for clustering
# 
# The function DataForClustering is applied to the subset of the data (in the following example, the subset of the data is the one which system = 1.   
# 
# The output of the function returns weighted sentence vectors, and index created for each row.

# In[ ]:


df_nlp = swd.DataForClustering(dat = df, group = True, 
                                      sort_var_1 = 'System', sort_var_2 = 'Meddelelsesdato',
                                      textcol = 'Text',
                                      stopwordpath = stopword_path, stopenc = "ISO 8859-1",
                                      fasttext_link = fasttext_link)


# In[ ]:


pd.set_option('display.max_columns', None)  
df_nlp.head()
df_nlp['Text'][0]  # the first row


# In[ ]:


df_nlp[df_nlp['System'] == 9].Index


# In[ ]:


df_nlp.to_csv('C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/otu sap wcs 2020 12/processed_df_allSystem.csv', 
              sep=';', encoding='ISO 8859-1')


# ## Cosine Similarity

# In[ ]:


sim_cluster_output = cmf.WCSClustering(df_nlp, fasttext_link,
                                       group = True, groupvar = 'System',
                                       unique_cluster = True, cos_similarity = 0.8,
                                       numdefects = 1)
sim_cluster_output


# In[ ]:


#sim_cluster_output.to_csv('C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/otu sap wcs 2020 12/nlp/output/cosine_sim_cluster_output_1.csv',
#                         sep = ";", header = True, encoding = "ISO 8859-1")

sim_cluster_output.to_csv('C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/otu sap wcs 2020 12/nlp/output/cosine_sim_cluster_allSystem.csv',
                         decimal = ",", header = True, index = False)


# In[ ]:


## Test for RAKE


# In[5]:


sim_cluster_output = pd.read_csv('C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/otu sap wcs 2020 12/nlp/output/cosine_sim_cluster_allSystem.csv',
                         decimal = ",")
sim_cluster_output.shape
sim_cluster_output


# In[99]:


#sim_cluster_output[sim_cluster_output.Cluster == 30293]
cluster_list = (407, 353)
sim_cluster_output[sim_cluster_output.Cluster.isin (cluster_list)]
#sim_cluster_output[sim_cluster_output.Cluster == 353]


# In[107]:


sim_cluster_output[sim_cluster_output.Cluster == 353].WeightVec 
sim_cluster_output[sim_cluster_output.Cluster == 407].WeightVec


# In[8]:


# drop duplicates in Text by group
unique_text_per_group = sim_cluster_output.drop_duplicates(["Cluster", "Text"])
unique_text_per_group.shape
unique_text_per_group[unique_text_per_group.Cluster == 30293]['Text']


# In[85]:


# combining all the texts from each rows together in order to present texts for groups
cluster_text = unique_text_per_group.groupby('Cluster')['Text'].apply(' '.join).reset_index()
cluster_text[11:15]


# In[87]:


import re
#t = "m43 rådjur påkört"
def remove_stringAndnumber(document):
    # Remove characters + numbers
    document = re.sub("(\D)([0-9]{2,4})", '', str(document))  #^[a-e]{1,4}
    
    # Remove number
    document = re.sub(r'[0-9]+', '', document)
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing repeated words
    document = re.sub(r'((\b\w+\b {1,2}\w+\b)+).+\1', r'\1', document, flags = re.I) 

    return document

cluster_text['Text_1'] = [remove_stringAndnumber(t) for t in cluster_text['Text']]
cluster_text.shape
cluster_text


# In[89]:


cluster_text[11:20]


# In[80]:


import RAKE
import operator

def AssessKeyphrase(stopwords, textvec = cluster_text['Text_1']):
    rake_object = RAKE.Rake(stopwords)
    keyphrase = []
    for elem in textvec:
        counts = sort_tuple(rake_object.run(elem, minCharacters = 3, maxWords = 8, minFrequency = 1))
        keyphrase.append(counts)
    return keyphrase

def sort_tuple(tup):
    # reverse = none (sorts in ascending order)
    # key is set to sort using second element of sublist 
    # lambda has been used
    tup.sort(key = lambda x: x[1])
    
    return tup

stopwordpath = stopword_path
stopwords = pd.read_csv(stopwordpath, encoding = "ISO 8859-1", sep = " ")
stopwords = stopwords['stoppord'].values.tolist()

cluster_text['Keyphrase'] = AssessKeyphrase(stopwords=stopwords)
cluster_text['Keyphrase']
#cluster_text['Keyphrase'] = keyphrase
#cluster_text['Keyphrase']
#cluster_text


# In[81]:


AssessKeyphrase(stopwords=stopwords)


# In[ ]:





# In[ ]:


#text = ['togsæt påkørt noget havari togsæt påkørt noget',
# 'skift justering rengøring frontlys skift justering rengøring frontlys',
# 'otätt tak läcker in stora mängder vatte otätt tak läcker in stora mängder vatte',
# 'ddi vindue førerrum',
# 'ddi vindue ddi vindue']
#text = [['Deep Learning is a subfield of AI. It is very useful.'], ["Role of women in agriculture"]]
rake_object = RAKE.Rake(stopwords)
keyphrase = []
for elem in text:
    counts = sort_tuple(rake_object.run(elem, minCharacters = 3, maxWords = 8, minFrequency = 1))
    keyphrase.append(counts)

cluster_text['Keyphrase'] = keyphrase
cluster_text['Keyphrase']


# In[ ]:


text = cluster_text['Text_1'].tolist()
text


# In[ ]:


sim_cluster_output_keyph = pd.merge(sim_cluster_output, cluster_text, on='Cluster', how='left')
sim_cluster_output_keyph.shape


# In[ ]:


sim_cluster_output_keyph.to_csv('C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/otu sap wcs 2020 12/nlp/output/cosine_sim_cluster_allSystem_keyphrase.csv',
                         decimal = ",", header = True, index = False)


# In[ ]:


from rake_nltk import Metric, Rake
rake = Rake()

text='Hello World'
#text = 'Deep Learning is a subfield of AI. It is very useful.'
#text = "Role of women in agriculture"
#text = "togsæt påkørt noget havari togsæt påkørt noget"

rake.get_ranked_phrases_with_scores()


# In[ ]:


from rake_nltk import Rake
from nltk.corpus import stopwords 
r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.Please note that "hello" is not included in the list of stopwords.

text='Hello World'
r.extract_keywords_from_text(text)
r.get_ranked_phrases()
r.get_ranked_phrases_with_scores()


# In[ ]:


# RAKE-NLTK
#!pip install rake-nltk

from rake_nltk import Metric, Rake
rake = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO, min_length=2, max_length=7)

#text = 'Deep Learning is a subfield of AI. It is very useful.'
text = "Role of women in agriculture"
#text = "togsæt påkørt noget havari togsæt påkørt noget"
keyphrase = []
for elem in text:
    counts = rake.extract_keywords_from_text(elem)
    keyphrase.append(counts)
keyphrase


# In[ ]:


#!pip install multi-rake
from multi_rake import Rake

#text_en = ('togsæt påkørt noget havari togsæt påkørt noget')

rake = Rake()

keyphrase = []
for elem in text:
    counts = rake.apply(elem)
    keyphrase.append(counts)
keyphrase[0:2]
text[0:2]


# In[ ]:


def CreateCorpus(textcol):
    corpus = textcol.tolist()
    return corpus


def TF_Idf_WeightMatrix(text):
    corpus = CreateCorpus(text)
    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()

    return response, words


# In[ ]:


text


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext
import numpy as np

def RunThisFirstIfYouWantToRunTestsInGlobalState():
    model = fasttext.load_model(fasttext_link)


def TF_Idf_WeightMatrix(text):
    #corpus = CreateCorpus(text)
    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform(text)
    words = vectorizer.get_feature_names()

    return words

text = [text]
keyphrase = []
for elem in text:
    counts = TF_Idf_WeightMatrix(elem)
    keyphrase.append(counts)
keyphrase

RunThisFirstIfYouWantToRunTestsInGlobalState()
#response, words = TF_Idf_WeightMatrix(cluster_text['Text'])
len(keyphrase)
len(words)
len(cluster_text['Text'])


# In[ ]:


#text = ['togsæt påkørt noget havari togsæt påkørt noget']
# text = cluster_text['Text'][0:5].tolist()
keyphrase = []
for elem in text:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    word = vectorizer.get_feature_names()
    keyphrase.append(word)
keyphrase


len(keyphrase)
len(cluster_text['Text'])


# In[ ]:


cluster_text['Keyph_tfidf'] = keyphrase
sim_cluster_output_keyph_tfidf = pd.merge(sim_cluster_output, cluster_text, on='Cluster', how='left')
sim_cluster_output_keyph_tfidf.shape
sim_cluster_output_keyph_tfidf.to_csv('C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/otu sap wcs 2020 12/nlp/output/cosine_sim_cluster_allSystem_keyphrase.csv',
                         decimal = ",", header = True, index = False)


# In[ ]:


# checking what the clusters are
sim_cluster_output.groupby('Cluster')['Cluster'].nunique()


# In[ ]:


# checking total number of rows in one cluster
pd.Series(sim_cluster_output.Cluster).value_counts()


# In[ ]:


# Output only last defect per cluster
sim_cluster_output_last = sim_cluster_output.loc[sim_cluster_output.groupby('Cluster').Meddelelsesdato.idxmax()]
sim_cluster_output_last.sort_values(by=['Counts'], ascending=False, inplace=True)
len(sim_cluster_output_last)
sim_cluster_output_last


# In[ ]:


# listing the elements in one cluster
text_sim_id_dic = dict(zip(similarity_output.Index, similarity_output.Text))
#text_sim_id_dic
keys = similarity_output.SimInd[7135]
for key in keys:
    text_sim_id_dic.get(key)
#text_sim_id_dic[(111, 116)]

