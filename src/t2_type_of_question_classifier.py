import os
from datetime import timedelta
from time import time

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.t1_type_of_question_classifier import extract_features, train_model

sns.set(style="darkgrid")

data_analysis_path = '../results/quora_dataset_experiments/'
os.makedirs(data_analysis_path, exist_ok=True)


def split_and_extract_features(df):
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

    features_train, features_test = extract_features(x_train, x_test)

    return features_train, features_test, y_train, y_test


if __name__ == '__main__':
    df = pd.read_csv('../data/All_Questions.csv', index_col=0)
    features_train, features_test, y_train, y_test = split_and_extract_features(df)
    models = [
        RandomForestClassifier(n_estimators=200, n_jobs=-1),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    models_names = ['RF', 'LinearSVC', 'Multinomial', 'LogisticRegression']

    accuracies = []

    for model, model_name in zip(models, models_names):
        print(model_name)
        initial_time = time()
        acc = train_model(model, features_train, features_test, y_train, y_test)
        accuracies.append(acc)
        print('Time training {}: '.format(model_name), timedelta(seconds=time() - initial_time))

    sns.barplot(x=models_names, y=accuracies)
    plt.ylabel('Accuracies')
    plt.ylim(0, 1)
    plt.savefig(data_analysis_path + 'barplot_quora_' + 'baseline_models_with_stop_words', bbox_inches='tight', dpi=200)
