import pandas as pd
import glob
import os
from spellchecker import SpellChecker
from ast import literal_eval

path = '../data/preprocessed/'
files = glob.glob(f'{path}*.csv')
spell = SpellChecker()

for file in files:
    name = file.split('/')[-1]
    df = pd.read_csv(f'{path}{name}', converters={'tokens': literal_eval, 'nltk_pos': literal_eval, 'spacy_pos': literal_eval,'spacy_label': literal_eval,})
    df['is_person_spacy'] = df['spacy_label'].apply(lambda x: 1 if 'PERSON' in x else 0)
    df['spacy_pos_len'] = df['spacy_pos'].apply(len)
    df_inventor = df[df['is_person_spacy'] == 1]
    df_non_inventor = df[df['is_person_spacy'] == 0]
    
    df_prpn = df_non_inventor[(df_non_inventor['is_prpn_spacy'] == 1) & (df_non_inventor['spacy_pos_len'] <= 2)]
    df_non_prpn_1 = df_non_inventor[(df_non_inventor['is_prpn_spacy'] == 1) & (df_non_inventor['spacy_pos_len'] > 2)]
    df_non_prpn_2 = df_non_inventor[df_non_inventor['is_prpn_spacy'] == 0]
    df_non_prpn = pd.concat([df_non_prpn_1, df_non_prpn_2], ignore_index=True)
    
    non_terminating_pos = ['ADP', 'CONJ', 'CCONJ', 'DET', 'ADP']
    df_non_prpn['bad_endings'] = df['spacy_pos'].apply(lambda x: 1 if x[-1] in non_terminating_pos else 0)
    df_bad_endings = df_non_prpn[df_non_prpn['bad_endings'] == 1]
    df_non_bad_endings = df_non_prpn[df_non_prpn['bad_endings'] == 0]

    df_single_pos = df_non_bad_endings[df_non_bad_endings['spacy_pos_len'] == 1]
    df_non_single_pos = df_non_bad_endings[df_non_bad_endings['spacy_pos_len'] != 1]

    df_non_single_pos['bad_starts'] = df_non_single_pos['spacy_pos'].apply(lambda x: 1 if x[0] in non_terminating_pos else 0)
    df_bad_starts = df_non_single_pos[df_non_single_pos['bad_starts'] == 1]
    df_non_bad_starts = df_non_single_pos[df_non_single_pos['bad_starts'] == 0]

    df_non_bad_starts['title'] = df_non_bad_starts['title'].apply(lambda x: x.replace('-', ' '))
    df_non_bad_starts['misspelt_words'] = df_non_bad_starts['title'].apply(lambda x: spell.unknown(x.split()))
    df_non_bad_starts['misspelt'] = df_non_bad_starts['misspelt_words'].apply(lambda x: 1 if len(x) > 0 else 0)
    df_spelling_errors = df_non_bad_starts[df_non_bad_starts['misspelt'] != 0]
    df_non_spelling_errors = df_non_bad_starts[df_non_bad_starts['misspelt'] == 0]

    if len(df) == (len(df_inventor) + len(df_prpn) + len(df_bad_endings) + len(df_single_pos) + len(df_bad_starts) + len(df_spelling_errors) + len(df_non_spelling_errors)):
        name = name.split('.')[0] # removing the .csv in the name
        save_path = '../data/cleaned/'
        os.mkdir(f'{save_path}{name}')

        df_inventor.to_csv(f'../data/cleaned/{name}/{name}_inventor.csv', index=False)
        df_prpn.to_csv(f'../data/cleaned/{name}/{name}_proper_nouns.csv', index=False)
        df_bad_endings.to_csv(f'../data/cleaned/{name}/{name}_incomplete.csv', index=False)
        df_single_pos.to_csv(f'../data/cleaned/{name}/{name}_single_pos.csv', index=False)
        df_bad_starts.to_csv(f'../data/cleaned/{name}/{name}_bad_starts.csv', index=False)
        df_spelling_errors.to_csv(f'../data/cleaned/{file}_non_bad_starts.csv', index=False)
        df_non_spelling_errors.to_csv(f'../data/cleaned/{file}_non_bad_starts.csv', index=False)
