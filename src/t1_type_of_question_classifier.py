import os
import re
from datetime import timedelta
from time import time

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import spacy

nlp = spacy.load("en_ner_bionlp13cg_md")
lemmatizer = spacy.load("en_core_web_sm")

sns.set(style="darkgrid")

data_analysis_path = '../results/data_analysis/'


def train_model(model, X_train, X_test, y_train, y_test, target_names=['summary', 'list', 'yesno', 'factoid']):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cr = metrics.classification_report(y_test, y_pred,
                                       target_names=target_names)
    print(cr)
    acc = model.score(X_test, y_test)
    return acc, cr


def perform_NER(s):
    doc = nlp(s)
    # displacy.serve(doc, style="ent")
    # processed_question = " ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc])
    newString = s
    for e in reversed(doc.ents):  # reversed to not modify the offsets of other entities when substituting
        start = e.start_char
        end = start + len(e.text)
        newString = newString[:start] + e.label_ + newString[end:]

    return newString


def remove_symbols(question):
    without_symbols = re.sub('[^a-zA-Z-_]+', ' ', question)
    return without_symbols + ' ?' if '?' in question else without_symbols


def replace_syndrome(s):
    try:
        s = s.replace('syndromes', 'syndrome')
        if 'syndrome' in s:
            splitted = s.split()
            syndrome_index = splitted.index('syndrome')
            splitted[syndrome_index - 1] = 'syndrome'
            s = ' '.join([w for w in splitted]).replace('syndrome syndrome', 'SYNDROME')
    except:
        pass
    return s

def lemmatize(s):
    s = lemmatizer(s)
    lemmas = [token.lemma_ for token in s]
    s = ' '.join([w for w in lemmas])
    return s

def preprocessor(question):
    processed_question = perform_NER(question)
    processed_question = remove_symbols(processed_question)
    processed_question = replace_syndrome(processed_question)
    # processed_question = lemmatize(processed_question)
    # print('\nOriginal:', question, '\nProcessed:', processed_question, '\n')
    return processed_question


def extract_tfidf(X_train, X_test, preprocessing=False):
    if preprocessing:
        tfidf = TfidfVectorizer(sublinear_tf=True,
                                preprocessor=preprocessor,
                                max_features=900,
                                min_df=5, norm='l2',
                                encoding='latin-1',
                                ngram_range=(1, 2), )
    else:
        tfidf = TfidfVectorizer(sublinear_tf=True,
                                max_features=900,
                                min_df=5, norm='l2',
                                encoding='latin-1',
                                ngram_range=(1, 2))
    tfidf.fit(X_train)
    features_train = tfidf.transform(X_train).toarray()
    features_test = tfidf.transform(X_test).toarray()
    return features_train, features_test


if __name__ == '__main__':
    df = pd.read_excel('../data/Questions.xlsx')
    print(df.shape)
    type_to_category = {'summary': 0, 'list': 1, 'yesno': 2, 'factoid': 3}

    apply_type_to_category = lambda t: type_to_category[t]

    df['category_id'] = df['Type'].apply(apply_type_to_category)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(df['Question'], df['category_id'],
                                                                                     df.index,
                                                                                     test_size=0.2, random_state=0)
    preprocessing = True
    features_train, features_test = extract_tfidf(X_train, X_test, preprocessing=preprocessing)
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
    results_path = '../results/t1_experiments/'
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, 'baseline{}.txt'.format('_preprocessing' if preprocessing else '')), 'w') as f:
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

    sns.barplot(x=models_names, y=accuracies)
    plt.ylabel('Accuracies')
    plt.ylim(0, 1)
    plt.savefig(results_path + 'barplot_'.format('preprocessing' if preprocessing else ''), bbox_inches='tight', dpi=200)
