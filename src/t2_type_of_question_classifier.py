import os
from datetime import timedelta
from time import time

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from src.t1_type_of_question_classifier import extract_features, train_model

sns.set(style="darkgrid")

results_path = '../results/t2_experiments/'


def split_and_extract_features(df, preprocessing):
    x_original = df.loc[df['original_dataset'].values, 'Question']
    y_original = df.loc[df['original_dataset'].values, 'category_id']

    x_quora = df.loc[~df['original_dataset'].values, 'Question']
    y_quora = df.loc[~df['original_dataset'].values, 'category_id']

    # I only want the original data in the test
    x_train_original, x_test, y_train_original, y_test = train_test_split(x_original, y_original, test_size=0.2,
                                                                          random_state=0)

    # I append all the quora data to the train
    x_train = np.append(x_train_original.values, x_quora.values)
    y_train = np.append(y_train_original, y_quora)

    features_train, features_test = extract_features(x_train, x_test, preprocessing)

    return features_train, features_test, y_train, y_test


def load_or_extract(dataset_name, df, preprocessing):
    try:
        features_train = np.load(
            '../data/{}{}_features_train'.format(dataset_name, '_preprocessing' if preprocessing else ''))
        y_train = np.load('../data/{}_{}_y_train'.format(dataset_name, '_preprocessing' if preprocessing else ''))
        features_test = np.load(
            '../data/{}{}_features_test'.format(dataset_name, '_preprocessing' if preprocessing else ''))
        y_test = np.load('../data/{}{}_y_test'.format(dataset_name, '_preprocessing' if preprocessing else ''))
    except:
        features_train, features_test, y_train, y_test = split_and_extract_features(df, preprocessing)
        np.save('../data/{}{}_features_train'.format(dataset_name, '_preprocessing' if preprocessing else ''),
                features_train)
        np.save('../data/{}{}_features_test'.format(dataset_name, '_preprocessing' if preprocessing else ''),
                features_test)
        np.save('../data/{}{}_y_train'.format(dataset_name, '_preprocessing' if preprocessing else ''), y_train)
        np.save('../data/{}{}_y_test'.format(dataset_name, '_preprocessing' if preprocessing else ''), y_test)
    return features_train, features_test, y_train, y_test


if __name__ == '__main__':
    dataset_name = 'whole_quora'
    df = pd.read_csv('../data/{}.csv'.format(dataset_name), index_col=0)
    df = df.dropna()
    voting_estimators = [
        ('Random Forest', RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced')),
        ('LinearSVC', LinearSVC(class_weight='balanced')),
        ('LogisticRegression', LogisticRegression(class_weight='balanced'))
    ]

    models = [
        RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced'),
        LinearSVC(class_weight='balanced'),
        MultinomialNB(),
        LogisticRegression(random_state=0, class_weight='balanced'),
        VotingClassifier(estimators=voting_estimators),
        MLPClassifier(hidden_layer_sizes=100)
    ]
    models_names = [
        'Random Forest',
        'LinearSVC',
        'Multinomial',
        'LogisticRegression',
        'Voting Classifier',
        'MLP default']
    preprocessing = True

    accuracies = []
    with open(os.path.join(results_path, '{}{}.txt'.format(dataset_name, '_preprocessing' if preprocessing else '')),
              'a') as f:
        initial_time = time()
        features_train, features_test, y_train, y_test = load_or_extract(dataset_name, df, preprocessing)
        f.write('Time Preprocessing: {}\n '.format(timedelta(seconds=time() - initial_time)))
        for model, model_name in zip(models, models_names):
            print(model_name)
            initial_time = time()
            f.write('{}\n'.format(model_name))
            acc, classification_report = train_model(model, features_train, features_test, y_train, y_test)
            f.write('Classification Report:{}\n'.format(classification_report))
            accuracies.append(acc)
            print('Time training {}: '.format(model_name), timedelta(seconds=time() - initial_time))
            f.write('Time training: {}\n '.format(timedelta(seconds=time() - initial_time)))
            f.write('\n' + '-' * 80 + '\n\n')

    # sns.barplot(x=models_names, y=accuracies)
    # plt.ylabel('Accuracies')
    # plt.ylim(0, 1)
    # plt.savefig(results_path + 'barplot_quora_'.format('preprocessing' if preprocessing else ''), bbox_inches='tight',
    #             dpi=200)
