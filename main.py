import re
import nltk
import multiprocessing
# import en_core_web_sm
# import spacy
# import medspacy
import pandas as pd 

from nltk.tokenize import word_tokenize
nltk.download('punkt')
from gensim.models import Word2Vec
# from spacy.tokens import doc
# from spacy.language import Language

# @Language.component("token_info")
# def custom_component(document:doc)-> doc:
#     """Print the number of tokens in each pipeline

#     Args:
#         document (spaCy document): sequence of tokens

#     Returns:
#         spaCy document: sequence of tokens
#     """    
#     print(f"This document contains {len(document)} tokens")
#     return document

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

def compute_cosine_similarity(model, values):
    try: 
        return model.wv.similarity(values['Term1'], values['Term2'])
    except KeyError as e: 
        print(e)
        return None

def main(clinical_notes_path:str, medical_concept_path:str)-> None:
    
    # load clincial notes and medical concepts
    df = pd.read_csv(clinical_notes_path)
    df_eval = pd.read_csv(medical_concept_path)
    
    #### generate text for df_eval and add in the training corpus
    
    # preprocess clinical notes and create training corpus
    df['clean_notes'] = df['notes'].apply(lambda x: preprocess_corpus(x))
    df['tokenize_text'] = df['clean_notes'].apply(lambda x: word_tokenize(x))
    training_corpus = df['tokenize_text'].values.tolist()
    
    # instantiate the number of cores
    cores =  multiprocessing.cpu_count()
    
    # instantiate the word2vec model 
    model = Word2Vec(sentences=training_corpus,
                     min_count=5, 
                     window=2,
                    #  size=300,
                     alpha=0.01, 
                     min_alpha=0.0005,
                     negative=20,
                     workers=cores-1)
    
    # # instantiate spacy and medspacy pipelines with only tokenizers
    # basic_spacy = spacy.load('en_core_web_sm', exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    # basic_spacy.add_pipe('token_info', name='print_tokens', last=True)

    # medical_spacy = medspacy.load('en_core_web_sm', medspacy_disable=['medspacy_pyrush', 'medspacy_target_matcher', 'medspacy_context'])
    # medical_spacy.add_pipe('token_info', name='print_tokens', last=True)
    # # note: the medical spacy tokenizer is not visible within the pipeline,
    # # refer to for more information: https://github.com/medspacy/medspacy/blob/master/notebooks/01-Introduction.ipynb
    
    
    df_eval['model_eval'] = df_eval.apply(lambda x: compute_cosine_similarity(model, x), axis=1)
    print(df_eval["model_eval"])
    # print(df_eval.columns)
    return


if __name__ == "__main__":
    clinical_notes_path, medical_concept_path = 'data/ClinNotes.csv', 'data/MedicalConcepts.csv'
    main(clinical_notes_path, medical_concept_path)