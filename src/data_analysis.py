import os
import re
from collections import Counter

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

sns.set(style="darkgrid")

from matplotlib import pyplot as plt

question_types = ['summary', 'list', 'yesno', 'factoid']
data_analysis_path = '../results/data_analysis/'


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
    plt.savefig(data_analysis_path + 'barplot_' + target_column, bbox_inches='tight', dpi=200)
    # plt.show()


def plot_countplot(dataframe, target_column):
    number_of_classes = len(Counter(dataframe[target_column]))
    x_size = number_of_classes + 1
    y_size = 4
    plt.figure(figsize=(x_size, y_size))
    sns.countplot(x=target_column, data=dataframe)
    plt.savefig(data_analysis_path + 'barplot_' + target_column, bbox_inches='tight', dpi=200)
    # plt.show()


def clean_word(word):
    word = word.lower()
    word = re.sub(r'[^\w]', '', word)
    return word


def report_data_analysis(questions):
    # Basic analysis
    print('Dataset size:', questions.shape)
    print('Columns:', questions.columns)
    print('Example:\n', questions.head())

    plot_countplot(questions, 'Type')

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
    words_by_class = {}
    for question_type in question_types:
        word_list = [clean_word(w) for s in
                     questions.loc[questions['Type'] == question_type, 'Question'].values for w in s.split()]
        words_by_class[question_type] = word_list
    print(words_by_class.keys())


def most_common_words(questions, max_num_words=20):
    words_by_class = {}
    all_the_words = [clean_word(w) for s in
                     questions['Question'].values for w in s.split()]
    word_set = set()

    most_common_words_from_all_types = [word for word, word_count in Counter(all_the_words).most_common(max_num_words)]
    print('Most common words in the whole dataset:\n', most_common_words_from_all_types)
    df_most_common = pd.DataFrame(0, index=most_common_words_from_all_types, columns=question_types)

    for question_type in question_types:
        word_list = [clean_word(w) for s in
                     questions.loc[questions['Type'] == question_type, 'Question'].values for w in s.split() if
                     clean_word(w) in most_common_words_from_all_types]
        for word in word_list:
            if word not in set(stopwords.words('english')):
                word_set.add(word)
        count = Counter(word_list)
        words = list(count.keys())
        values = list(count.values())

        df_most_common.loc[words, question_type] = values
        print('{}:'.format(question_type, len(Counter(word_list))), '\n', Counter(word_list).most_common(max_num_words),
              '\n')
        words_by_class[question_type] = word_list

    print(df_most_common)
    if max_num_words ==20:
        stacked = df_most_common.stack().reset_index().rename(columns={0: 'value'})

        x_size = max_num_words // 2
        y_size = 4
        plt.figure(figsize=[x_size, y_size])
        barplot = sns.barplot(x=stacked['level_0'], y=stacked['value'], hue=stacked['level_1'])
        barplot.axes.get_legend().set_title('Types')
        plt.xlabel('Words')
        plt.xticks(rotation=60)
        plt.savefig(data_analysis_path + 'barplot_' + 'most_common_words_by_type', bbox_inches='tight', dpi=200)

    print('Common words:', len(word_set))

    with open('../data/medical_words.txt', 'w') as f:
        for item in word_set:
            f.write("%s\n" % item)


def most_common_bigrams(questions, max_num_words=20):
    words_by_class = {}
    all_the_words = [clean_word(w) for s in
                     questions['Question'].values for w in s.split()]
    most_common_words_from_all_types = [word for word, word_count in
                                        Counter(zip(all_the_words, all_the_words[1:])).most_common(max_num_words)]
    print('Most common bigrams in the whole dataset:\n', most_common_words_from_all_types)
    df_most_common = pd.DataFrame(0, index=most_common_words_from_all_types, columns=question_types)
    for question_type in question_types:
        word_list = [clean_word(w) for s in
                     questions.loc[questions['Type'] == question_type, 'Question'].values for w in s.split()]

        count = Counter(zip(word_list, word_list[1:]))
        words = list(count.keys())
        values = list(count.values())
        for w, v in zip(words, values):
            try:
                df_most_common.loc[w, question_type] = v
            except:
                pass

        print('{}:'.format(question_type, len(Counter(word_list))), '\n', count.most_common(max_num_words),
              '\n')
        words_by_class[question_type] = word_list

    print(df_most_common)

    stacked = df_most_common.stack().reset_index().rename(columns={0: 'value'})

    x_size = max_num_words // 2
    y_size = 4
    plt.figure(figsize=[x_size, y_size])
    barplot = sns.barplot(x=stacked['level_0'], y=stacked['value'], hue=stacked['level_1'])
    barplot.axes.get_legend().set_title('Types')
    plt.xlabel('Words')
    plt.xticks(rotation=90)
    plt.savefig(data_analysis_path + 'barplot_' + 'most_common_bigrams_by_type', bbox_inches='tight', dpi=200)


if __name__ == '__main__':
    questions = pd.read_csv('../data/All_Questions.csv', index_col=0)
    print(questions.columns)
    y = questions['Type'].values
    X = questions['Question'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    train_df = pd.DataFrame(X_train, columns=['Question'])
    train_df['Type'] = y_train

    test_df = pd.DataFrame(X_test, columns=['Question'])
    test_df['Type'] = y_test

    print(X_train.shape, X_test.shape)
    most_common_words(train_df,max_num_words=500)
    # most_common_bigrams(train_df)
