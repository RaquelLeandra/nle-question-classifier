import os
import re
from collections import Counter
from datetime import timedelta
from time import time

import numpy as np
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from src.data_analysis import clean_word
from src.t1_type_of_question_classifier import train_model, extract_tfidf
from sklearn.svm import LinearSVC

question_types = ['summary', 'list', 'yesno', 'factoid']

nlp = spacy.load("en_core_web_sm")


def get_first_element(phrase):
    return phrase.split()[0].lower()


def get_last_element(phrase):
    return phrase.split()[-1].lower()


def get_question_mark(word):
    return list(word)[-1] == '?'


def remove_question_mark(word):
    if (get_question_mark(word)):
        return word[:-1]
    else:
        return word


def is_plural(phrase):
    question = nlp(phrase)
    tags = [token.tag_ for token in question]
    return 'NNS' in tags or 'VBP' in tags


def get_length_phrase(phrase):
    return len(phrase.split())


def clean_word(word):
    word = word.lower()
    word = re.sub(r'[^\w]', '', word)
    return word


def check_word_exists_in(phrase, important_word):
    return important_word in phrase.split()


def run_all_experiments(results_path, features_train, y_train, features_test, y_test, preprocessing=False):
    voting_estimators = [
        ('Random Forest', RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced')),
        ('LinearSVC', LinearSVC(class_weight='balanced')),
        ('LogisticRegression', LogisticRegression(class_weight='balanced'))
    ]
    models = [
        RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced'),
        LinearSVC(class_weight='balanced'),
        MultinomialNB(),
        LogisticRegression(class_weight='balanced'),
        VotingClassifier(estimators=voting_estimators),
        MLPClassifier(hidden_layer_sizes=100)
    ]
    models_names = ['Random Forest', 'LinearSVC', 'Multinomial', 'LogisticRegression', 'Voting Classifier',
                    'MLP default', ]

    accuracies = []
    with open(os.path.join(results_path,
                           'features_{}{}.txt'.format(dataset_name, '_preprocessing' if preprocessing else '')),
              'w') as f:
        for model, model_name in zip(models, models_names):
            print(model_name)
            initial_time = time()
            f.write('{}\n'.format(model_name))
            acc, classification_report = train_model(model, features_train, features_test, y_train, y_test)
            f.write('Classification Report:{}\n'.format(classification_report))
            print('Time training :{} '.format(timedelta(seconds=time() - initial_time)))
            f.write('Time training: {}\n '.format(timedelta(seconds=time() - initial_time)))
            f.write('\n' + '-' * 80 + '\n\n')
            accuracies.append(acc)


def split_small_dataset(df):
    x_train, x_test, y_train, y_test = train_test_split(df['Question'], df['Type'],
                                                        test_size=0.2, random_state=0)
    x_train = pd.DataFrame(x_train, columns=['Question'])
    x_train['Type'] = y_train
    x_test = pd.DataFrame(x_test, columns=['Question'])
    x_test['Type'] = y_test
    return x_train, x_test


def split_quora(df):
    df = df.dropna()

    x_original = df.loc[df['original_dataset'].values, 'Question']
    y_original = df.loc[df['original_dataset'].values, 'Type']

    x_quora = df.loc[~df['original_dataset'].values, 'Question']
    y_quora = df.loc[~df['original_dataset'].values, 'Type']

    # I only want the original data in the test
    x_train_original, x_test, y_train_original, y_test = train_test_split(x_original, y_original, test_size=0.2,
                                                                          random_state=0)

    # I append all the quora data to the train
    x_train = np.append(x_train_original.values, x_quora.values)
    y_train = np.append(y_train_original, y_quora)

    x_train = pd.DataFrame(x_train, columns=['Question'])
    x_train['Type'] = y_train
    x_test = pd.DataFrame(x_test, columns=['Question'])
    x_test['Type'] = y_test

    return x_train, x_test


def extract_features(questions_train, questions_test):
    features = pd.DataFrame()
    features_train = pd.DataFrame()
    features_test = pd.DataFrame()

    # Last word
    features_train['last_word'] = questions_train['Question'].apply(get_last_element)
    features_test['last_word'] = questions_test['Question'].apply(get_last_element)

    # Sentence number
    features_train['is_plural'] = questions_train['Question'].apply(is_plural)
    features_test['is_plural'] = questions_test['Question'].apply(is_plural)

    # Has question mark?
    features_train['has_question_mark'] = features_train['last_word'].apply(get_question_mark) * 1
    features_test['has_question_mark'] = features_test['last_word'].apply(get_question_mark) * 1

    # Get average length of the dataset
    average_len_train = questions_train['Question'].apply(get_length_phrase).mean()

    # classify phrase into short, long
    features_train['long_phrase'] = questions_train['Question'].apply(get_length_phrase) > average_len_train * 1
    features_test['long_phrase'] = questions_test['Question'].apply(get_length_phrase) > average_len_train * 1

    features_train = features_train.drop(['last_word'], axis=1)
    features_test = features_test.drop(['last_word'], axis=1)

    X_train = features_train.values
    y_train = questions_train['Type'].values
    X_test = features_test.values
    y_test = questions_test['Type'].values
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # dataset_name = 'Questions'
    # questions = pd.read_excel('../data/{}.xlsx'.format(dataset_name))
    dataset_name = 'whole_quora'
    questions = pd.read_csv('../data/{}.csv'.format(dataset_name), index_col=0)
    preprocessing = False

    # Shuffle questions dataset
    if dataset_name == 'Questions':
        questions_train, questions_test = split_small_dataset(questions)
    else:
        questions_train, questions_test = split_quora(questions)

    features_train, y_train, features_test, y_test = extract_features(questions_train, questions_test)

    tf_train, tf_test = extract_tfidf(questions_train['Question'].values, questions_test['Question'].values,
                                      preprocessing=preprocessing)

    X_train = np.concatenate([features_train, tf_train], axis=1)
    X_test = np.concatenate([features_test, tf_test], axis=1)

    results_path = '../results/concat_experiments/'
    os.makedirs(results_path, exist_ok=True)
    run_all_experiments(results_path, X_train, y_train, X_test, y_test, preprocessing=preprocessing)
