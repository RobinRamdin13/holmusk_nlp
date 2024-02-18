import re
import en_core_web_sm
import spacy
import medspacy
import pandas as pd 

from spacy.tokens import doc
from spacy.language import Language


# from spacy.tokenizer import Tokenizer
# from spacy.lang.en import English
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

def tokenize_text(text:list):
    # combine all strings into one
    text = ' '.join(text)

    # load basic spacy pipeline, exclude all other contents of pipeline
    basic_spacy = spacy.load('en_core_web_sm', exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    basic_spacy.add_pipe('token_info', name='print_tokens', last=True)
    # check to ensure only tok2vec is present within pipe, 
    # print(basic_spacy.pipe_names) 

    
    # load medspacy pipeline, exclude all other contents of pipeline
    medical_spacy = medspacy.load(medspacy_disable=['medspacy_pyrush', 'medspacy_target_matcher', 'medspacy_context'])
    medical_spacy.add_pipe('token_info', name='print_tokens', last=True)
    # note: the medical spacy tokenizer is not visible within the pipeline,
    # refer to for more information: https://github.com/medspacy/medspacy/blob/master/notebooks/01-Introduction.ipynb
    # print(medical_spacy.pipe_names)

    # increase text length limit to override limit set by spacy
    basic_spacy.max_length = 2300000
    medical_spacy.max_length = 2300000

    # load text into tokenizers
    basic_token = basic_spacy(text)
    medical_token = medical_spacy(text)

    basic_token_list = [t.text for t in basic_token]
    print(basic_token_list)
    return

def main(data_path:str)-> None:
    
    # load clincial notes
    df = pd.read_csv(data_path)
    
    # preprocess clinical notes
    df['clean_notes'] = df['notes'].apply(lambda x: preprocess_corpus(x))
    
    # tokenize the clean_notes text
    tokenize_text(df['clean_notes'].values.tolist())

    # # check token count
    # token_list = [i.text for i in document]
    # print(token_list)
    
    return


if __name__ == "__main__":
    data_path = 'data/ClinNotes.csv'
    main(data_path)