import pandas as pd
from pandas import DataFrame

def augment(term1:str, term2:str)-> str:
    """Create new text based on evluation data pairs

    Args:
        term1 (str): term1
        term2 (str): term2

    Returns:
        str: new text for data augmentation
    """    
    return f"clinician annotation identified {term1.lower()} and {term2.lower()} are closely related to each other"

def merge_str(category:str, notes:str)-> str:
    """Merge the two strings into one

    Args:
        category (str): category
        notes (str): notes

    Returns:
        str: concatenated string
    """    
    return category+notes

def main(clinical_notes_path:str, medical_concept_path:str)-> None:  
    # load clincial notes and medical concepts
    train = pd.read_csv(clinical_notes_path)
    eval = pd.read_csv(medical_concept_path)

    # extract text and list and generate frequency count
    train['merge'] = train.apply(lambda x: merge_str(x.category, x.notes), axis=1) # merge the category with the notes
    train_list = train['merge'].values.tolist() # extract the merge text
    eval['augment'] = eval.apply(lambda x: augment(x.Term1, x.Term2), axis=1) # generate new text
    augment_list = eval['augment'].values.tolist() # combine augmented text into 1 string

    # merge train_text and augment_text
    final_list = train_list + augment_list
    df = pd.DataFrame(final_list, columns=['data'])
    df.to_csv('data/augmented_training.csv', index=False)

    final_text = " ".join(final_list)
    with open('data/augmented.txt', 'w') as outfile: 
        outfile.write(final_text)
    return 

if __name__ == "__main__":
    clinical_notes_path, medical_concept_path = 'data/ClinNotes.csv', 'data/MedicalConcepts.csv'
    main(clinical_notes_path, medical_concept_path)