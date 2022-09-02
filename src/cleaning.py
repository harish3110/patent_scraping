import pandas as pd
import glob
import os
from ast import literal_eval

path = '../data/preprocessed/'
files = glob.glob(f'{path}*.csv')

for file in files:
    name = file.split('/')[-1]
    df = pd.read_csv(f'{path}{name}', converters={'tokens': literal_eval, 'nltk_pos': literal_eval, 'spacy_pos': literal_eval,'spacy_label': literal_eval,})
    df['is_person_spacy'] = df['spacy_label'].apply(lambda x: 1 if 'PERSON' in x else 0)
    df['spacy_pos_len'] = df['spacy_pos'].apply(len)
    df_inventor = df[df['is_person_spacy'] == 1]
    df_non_inventor = df[df['is_person_spacy'] == 0]
    df_prpn = df_non_inventor[df_non_inventor['is_prpn_spacy'] == 1]
    df_non_prpn = df_non_inventor[df_non_inventor['is_prpn_spacy'] == 0]
    
    non_terminating_pos = ['ADP', 'CONJ', 'CCONJ', 'DET', 'ADP']
    df_non_prpn['bad_endings'] = df['spacy_pos'].apply(lambda x: 1 if x[-1] in non_terminating_pos else 0)
    df_bad_endings = df_non_prpn[df_non_prpn['bad_endings'] == 1]
    df_non_bad_endings = df_non_prpn[df_non_prpn['bad_endings'] == 0]

    df_single_pos = df_non_bad_endings[df_non_bad_endings['spacy_pos_len'] == 1]
    df_non_single_pos = df_non_bad_endings[df_non_bad_endings['spacy_pos_len'] != 1]

    df_non_single_pos['bad_starts'] = df_non_single_pos['spacy_pos'].apply(lambda x: 1 if x[0] in non_terminating_pos else 0)
    df_bad_starts = df_non_single_pos[df_non_single_pos['bad_starts'] == 1]
    df_non_bad_starts = df_non_single_pos[df_non_single_pos['bad_starts'] == 0]
    
    save_path = '../data/cleaned/'
    os.mkdir(f'{save_path}{name}')

    df_inventor.to_csv(f'../data/cleaned/{name}/{name}_inventor.csv', index=False)
    df_prpn.to_csv(f'../data/cleaned/{name}/{name}_proper_nouns.csv', index=False)
    df_bad_endings.to_csv(f'../data/cleaned/{name}/{name}_incomplete.csv', index=False)
    df_single_pos.to_csv(f'../data/cleaned/{name}/{name}_single_pos.csv', index=False)
    df_bad_starts.to_csv(f'../data/cleaned/{name}/{name}_bad_starts.csv', index=False)
    df_non_bad_starts.to_csv(f'../data/cleaned/{name}/{name}_non_bad_starts.csv', index=False)
