import re
import nltk
import spacy
import medspacy
import en_core_web_sm
import multiprocessing

import pandas as pd 
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
nltk.download('punkt')
from pandas import DataFrame
from gensim.models import Word2Vec
from spacy.tokens import doc
from spacy.language import Language

@Language.component("token_info")
def custom_component(document:doc)-> doc:
    """Print the number of tokens in each pipeline

    Args:
        document (spaCy document): sequence of tokens

    Returns:
        spaCy document: sequence of tokens
    """    
    print(f"This document contains {len(document)} tokens")
    return document

def preprocess_corpus(x:str)->str:
    """Replace names and extra whitespace

    Args:
        x (str): clinical notes

    Returns:
        str: cleaned clinical notes
    """    
    x = re.sub(r"(Mr.\s[A-Z][a-z]+)", "man", x) # replace Mr. with name using "man"
    x = re.sub(r"(Mr.\s[A-Z]*)", "man", x) # replace Mr. ABC using "man"
    x = re.sub(r"(Mrs.\s[A-Z][a-z]+)", "woman", x) # replace Mrs. with name using "woman"
    x = re.sub(r"(Mrs.\s[A-Z]*)", "woman", x) # replace Mrs. ABC name using "woman"
    x = re.sub(r"(Ms.\s[A-Z][a-z]+)", "woman", x) # replace Ms. with name using "woman"
    x = re.sub(r"(Ms.\s[A-Z]*)", "woman", x) # replace Ms. ABC name using "woman"
    x = re.sub(r"(Dr.\s[A-Z][a-z]+)", "doctor", x) # replace Dr. with name using "doctor"
    x = re.sub(r"(Dr.\s[A-Z]*)", "doctor", x) # replace Dr. ABC using "doctor"
    x = re.sub(r"(:\s*,)", " ", x) # remove artifacts
    x = re.sub(r"_+", " ", x) # remove artifacts
    x = re.sub(r".,", ". ", x) # remove artifacts
    x = re.sub(' +', ' ', x) # remove extra whitespace
    return x.lower()

def wv_cosine_similarity(model, values:DataFrame)-> float:
    """Compute the cosine similarity using word2vec model

    Args:
        model (Word2Vec Object): Word2Vec Model
        values (DataFrame): evaluation data

    Returns:
        float: cosine similarity metric
    """    
    try: 
        return model.wv.similarity(values['Term1'].lower(), values['Term2'].lower())
    # highlight the words which were not previously fed into w2v
    except KeyError as e: 
        print(e)
        return None

def spacy_cosine_similarity(model, values:DataFrame)->float:
    """Compute the cosine similarity using spacy model

    Args:
        model (Spacy Object): Spacy Model
        values (DataFrame): evaluation data

    Returns:
        float: cosine similarity metric
    """    
    term1, term2 = model(values['Term1'].lower()), model(values['Term2'].lower())
    return term1.similarity(term2)

def main(data_path:str, eval_path:str)-> None:
    # load the csv files
    df = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # preprocess clinical notes and create training corpus
    df['clean_data'] = df['data'].apply(lambda x: preprocess_corpus(x))
    df['tokenize_text'] = df['clean_data'].apply(lambda x: word_tokenize(x))
    training_corpus = df['tokenize_text'].values.tolist()
    training_corpus = [[y.lower() for y in x] for x in training_corpus]

    # instantiate the number of cores
    cores =  multiprocessing.cpu_count()
    
    # instantiate the word2vec model 
    model = Word2Vec(sentences=training_corpus,
                     min_count=1, 
                     window=2,
                     alpha=0.01, 
                     min_alpha=0.0005,
                     negative=20,
                     workers=cores-1)
    
    # instantiate spacy and medspacy pipelines with only tokenizers
    basic_spacy = spacy.load('en_core_web_sm', exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    # basic_spacy.add_pipe('token_info', name='print_tokens', last=True)

    medical_spacy = medspacy.load('en_core_web_sm', medspacy_disable=['medspacy_pyrush', 'medspacy_target_matcher', 'medspacy_context'])
    # medical_spacy.add_pipe('token_info', name='print_tokens', last=True)
    # note: the medical spacy tokenizer is not visible within the pipeline,
    # refer to for more information: https://github.com/medspacy/medspacy/blob/master/notebooks/01-Introduction.ipynb
    
    # compute the cosine similaritites on the evaluation data
    df_eval['model_eval'] = df_eval.apply(lambda x: wv_cosine_similarity(model, x), axis=1)
    df_eval['spacy_eval'] = df_eval.apply(lambda x: spacy_cosine_similarity(basic_spacy, x), axis=1)
    df_eval['medspacy_eval'] = df_eval.apply(lambda x: spacy_cosine_similarity(medical_spacy, x), axis=1)
    
    # drop the NA rows, these are unseen data within w2v and therefore do not have an embedding
    # data is unseen due to whitespace/punctuation
    df_eval = df_eval[df_eval['model_eval'].notna()]

    # create plot 
    index = df_eval.index.tolist()
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(index, df_eval['model_eval'].values.tolist(), label='Word2Vec')
    plt.plot(index, df_eval['spacy_eval'].values.tolist(), label='Spacy')
    plt.plot(index, df_eval['medspacy_eval'].values.tolist(), label='MedSpacy')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Word Pair Index')
    plt.title('Embedding Comparison')
    plt.grid()
    plt.legend()
    plt.savefig('plot/embeddings_comparison.jpeg')
    return


if __name__ == "__main__":
    data_path, eval_path = 'data/augmented_training.csv', 'data/MedicalConcepts.csv'
    main(data_path, eval_path)