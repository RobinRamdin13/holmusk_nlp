import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def freq_counter(train_text:str, eval_list:list)-> dict: 
    """Function to count the frequency of words from eval_list in the train_text

    Args:
        train_text (str): training text
        eval_list (list): list of words to be used in evaluation

    Raises:
        ValueError: check if dictionay key already exists

    Returns:
        dict: evaluation words with their frequency as key-value pairs
    """    
    counter = {}
    str_list = train_text.split() # split text on whitespace
    eval_list = set(eval_list) # ensure there is no duplicated words
    # iterate through eval_list
    for item in eval_list:
        if item in counter:
            raise ValueError('dict key already exists')
        else:
            counter[item] = str_list.count(item)
    return counter

def plot_freq_count(freqdict:dict)->None:
    """Generate the frequency plot of all words. Plot is too dense for viewing,
    alternate visualization should be use such as pie charts

    Args:
        freqdict (dict): words and their respective counts as key-value pairs
    """    
    freqdict = list(freqdict.items())
    word, count = zip(*freqdict)
    plt.figure(figsize=(30, 6), dpi=80)
    plt.plot(word, count)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Word Count in Training Data')
    plt.xticks(fontsize=8, rotation=90)
    plt.savefig('plot/freq_count.jpeg')
    return

def plot_pie(freqdict:dict, low_freqlist:list, count_threshold:int)->None:
    """Generate the pie chart at different thresholds

    Args:
        freqdict (dict): words and their respective counts as key-value pairs
        low_freqlist (list): list of words below count threshold
        count_threshold (int): arbitrary threshold
    """    
    plot_data = np.array([len(freqdict)-len(low_freqlist), len(low_freqlist)])
    labels = [f"% Above Threshold", f"% Below Threshold"]
    plt.pie(plot_data, autopct='%1.1f%%')
    plt.title(f"Word Distribution for Threshold: {count_threshold}")
    plt.legend(labels=labels)
    plt.savefig(f'plot/pieplot_threshold{count_threshold}.jpeg')
    return

def main(clinical_notes_path:str, medical_concept_path:str)-> None: 
    # define count threshold 
    count_threshold = 5

    # load clincial notes and medical concepts
    train = pd.read_csv(clinical_notes_path)
    eval = pd.read_csv(medical_concept_path)

    # extract text and list and generate frequency count
    train_text = " ".join(train['notes'].values.tolist()).lower() # extract the clinical notes from train
    eval_list = eval["Term1"].values.tolist() + eval['Term2'].values.tolist() # extract the medical concetps from eval
    eval_list = [x.lower() for x in eval_list]
    freqdict = freq_counter(train_text, eval_list)
    plot_freq_count(freqdict)

    # plot words below certain treshold 
    low_freqlist = [(k,v) for k,v in freqdict.items() if v <=count_threshold]
    plot_pie(freqdict, low_freqlist, count_threshold)
    return

if __name__ == "__main__":
    clinical_notes_path, medical_concept_path = 'data/ClinNotes.csv', 'data/MedicalConcepts.csv'
    main(clinical_notes_path, medical_concept_path)