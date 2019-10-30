import os

import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")
import numpy as np

from matplotlib import pyplot as plt

question_types = ['summary', 'list', 'yesno', 'factoid']


def get_first_element(phrase):
    return phrase.split()[0].lower()


def get_last_element(phrase):
    return phrase.split()[-1].lower()


def plot_most_common_words(dataframe, column_hue, target_column, min_num_samples):
    cross_tab = pd.crosstab(dataframe[column_hue], dataframe[target_column])
    valid_samples_cross_tab = cross_tab.loc[:, cross_tab.sum() >= min_num_samples]

    num_words = len(valid_samples_cross_tab.columns)
    x_size = num_words // 3
    y_size = 4
    plt.figure(figsize=(x_size, y_size))

    stacked = valid_samples_cross_tab.stack().reset_index().rename(columns={0: 'value', 1: target_column})
    sns.barplot(x=stacked[target_column], y=stacked['value'], hue=stacked[column_hue])
    plt.xticks(rotation=60)
    plt.show()


def report_data_analysis(path):
    # Basic analysis
    questions = pd.read_excel(path)
    print('Dataset size:', questions.shape)
    print('Columns:', questions.columns)
    print('Example:\n', questions.head())
    sns.countplot(x='Type', data=questions)
    plt.show()

    # Potential features
    # Question length
    questions['question_size'] = questions['Question'].apply(len)
    print(questions['question_size'].describe())
    for question_type in question_types:
        print(question_type, '\n', questions.loc[questions['Type'] == question_type, 'question_size'].describe())

    # First word
    questions['first_word'] = questions['Question'].apply(get_first_element)
    plot_most_common_words(questions, 'Type', target_column='first_word', min_num_samples=3)

    # Last word
    questions['last_word'] = questions['Question'].apply(get_last_element)
    plot_most_common_words(questions, 'Type', target_column='last_word', min_num_samples=3)

    # Most common words for each class
    # words_by_class = {}
    # for question_type in question_types:
    #     words_by_class[question_type] = questions.loc[questions['Type'] == question_type, 'Question'].str.lower().split()
    # print(words_by_class)


if __name__ == '__main__':
    local_data_path = '../data/'
    questions_path = os.path.join(local_data_path, 'Questions.xlsx')
    report_data_analysis(questions_path)
