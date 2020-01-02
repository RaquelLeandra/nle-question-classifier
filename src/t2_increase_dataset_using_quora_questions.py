import os

import pandas as pd
import seaborn as sns

from src.data_analysis import clean_word

sns.set(style="darkgrid")

data_analysis_path = '../results/data_analysis/'
data_path = '../data'


def check_med_words_in_question(question):
    medical_words_df = pd.read_csv(os.path.join(data_path, 'medical_words.txt'), names=['words'], )
    med_words_list = medical_words_df['words'].values.tolist()
    try:
        for word in question.split():
            if clean_word(word) in med_words_list:
                return True
    except:
        print(question)
        return False
    return False


def filter_using_medical_words(df):
    q1_bool = df['question1'].apply(check_med_words_in_question)
    q2_bool = df['question2'].apply(check_med_words_in_question)

    df['medical_questions'] = q1_bool | q2_bool
    print('Number of found medical questions:', df['medical_questions'].values.sum())
    df.to_csv(os.path.join(data_path, 'quora_with_med_questions.csv'))


if __name__ == '__main__':
    quora_duplicate_questions = pd.read_csv(os.path.join(data_path, 'quora_duplicate_questions.tsv'), sep='\t')
    quora_duplicate_questions = quora_duplicate_questions.set_index('id')
    print('Original shape:', quora_duplicate_questions.shape)
    # filter_using_medical_words(quora_duplicate_questions)
    quora_with_med = pd.read_csv(os.path.join(data_path, 'quora_duplicate_questions.tsv'), sep='\t')
