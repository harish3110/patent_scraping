import pandas as pd
import nltk
import spacy
import glob
from tqdm import tqdm

nltk.download('averaged_perceptron_tagger')

path = '../data/scraped/'
files = glob.glob(f'{path}*.csv')
nlp = spacy.load("en_core_web_sm")

def is_prpn_nltk(x):
    '''
    Helper function to detect proper nouns from pos tags in text
    '''
    for word, pos in x: 
        if pos == 'NNP':
            return 1
    return 0

def build_nltk_pos_tags(df):
    # converting title to str type
    df['title'] = df['title'].astype(str)
    # splitting title into tokens
    df['tokens'] = df['title'].apply(lambda x: x.split())
    # getting nltk pos tags from tokens
    df['nltk_pos'] = df['tokens'].apply(lambda x: nltk.pos_tag(x))
    return df

def spacy_pos(x):
    doc = nlp(x)
    return [token.pos_ for token in doc]

def spacy_label(x):
    doc = nlp(x)
    return [token.label_ for token in doc.ents]

def build_spacy_pos_tag(df):
    # converting title to str type
    df['title'] = df['title'].astype(str)
    df['spacy_pos'] = df['title'].apply(spacy_pos)
    df['spacy_label'] = df['title'].apply(spacy_label)
    df['is_prpn_spacy'] = df['spacy_pos'].apply(lambda x: 1 if 'PROPN' in x else 0) # checking if tokens are proper nouns
    df['is_person_spacy'] = df['spacy_label'].apply(lambda x: 1 if 'PERSON' in x else (1 if 'ORG' in x else 0)) # checking if tokens are person or organization entities
    return df

save_path = '../data/preprocessed/'

for file in tqdm(files):
    name = file.split('/')[-1]
    df = pd.read_csv(file)
    # df = build_nltk_pos_tags(df)
    df = build_spacy_pos_tag(df)
    df.to_csv(f'{save_path}{name}', index=False)


