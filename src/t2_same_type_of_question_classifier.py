import seaborn as sns
import numpy as np
import pandas as pd
from datetime import timedelta
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sns.set(style="darkgrid")

data_analysis_path = '../results/data_analysis/'


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def extract_categories(df):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), )
    # stop_words='english')
    df['category_id'] = df['Type'].factorize()[0]
    category_id_df = df[['Type', 'category_id']].drop_duplicates().sort_values('category_id')
    labels = df.category_id
    return labels, category_id_df


def extract_features_all(df_qt, df_quora):
    tfidf = TfidfVectorizer(sublinear_tf=True, max_features=900, min_df=5, norm='l2', encoding='latin-1',
                            ngram_range=(1, 2), )
    all_questions = np.concatenate((df_qt.Question.values.astype('U'), df_quora.question1.values.astype('U'),
                                    df_quora.question2.values.astype('U')), axis=0)
    features = tfidf.fit_transform(all_questions).toarray()
    return features


def split_features(features, num_qt, num_quora):
    features_qt = features[0:num_qt][:]
    features_1 = features[num_qt:num_qt + num_quora][:]
    features_2 = features[num_qt + num_quora:num_qt + 2 * num_quora][:]
    return features_qt, features_1, features_2


if __name__ == '__main__':
    initial_time = time()
    df = pd.read_excel('../data/Questions.xlsx')
    quora_df = pd.read_csv('../data/quora_with_med_questions.csv')

    quora_df = quora_df.loc[quora_df['medical_questions'].tolist(), :]
    # quora_df = quora_df.iloc[0:10000, :]
    print(df.shape)
    print(quora_df.shape)

    labels, category_id_df = extract_categories(df)
    features = extract_features_all(df, quora_df)
    features_qt, features_1, features_2 = split_features(features, df.shape[0], quora_df.shape[0])
    # We use all the original dataset for training

    print('Time extracting features: ', timedelta(seconds=time() - initial_time))
    X_train = features_qt
    X_test_1 = features_1
    X_test_2 = features_2
    y_train = labels
    initial_time = time()
    types = ['summary', 'list', 'yesno', 'factoid']
    model = LinearSVC()
    model_name = 'LinearSVC'
    print(model_name)
    trained_model = train_model(model, X_train, y_train)
    print('Time training: ', timedelta(seconds=time() - initial_time))

    initial_time = time()
    y_pred_1 = trained_model.predict(X_test_1)
    y_pred_2 = trained_model.predict(X_test_2)

    print('Time predicting: ', timedelta(seconds=time() - initial_time))

    initial_time = time()

    to_save = y_pred_1 == y_pred_2

    df_1 = pd.DataFrame(columns=['Question', 'Type', 'category_id'])
    df_1['Question'] = quora_df.loc[to_save, 'question1']
    df_1['Type'] = [types[value] for value in y_pred_1[to_save]]
    df_1['category_id'] = y_pred_1[to_save]

    df_2 = pd.DataFrame(columns=['Question', 'Type', 'category_id'])
    df_2['Question'] = quora_df.loc[to_save, 'question2']
    df_2['Type'] = [types[value] for value in y_pred_1[to_save]]
    df_2['category_id'] = y_pred_1[to_save]

    df['original_dataset'] = True
    df_1['original_dataset'] = False
    df_2['original_dataset'] = False

    all_questions = pd.concat([df, df_1, df_2], axis=0)
    print(df.shape, df_1.shape, df_2.shape, all_questions.shape, all_questions.isnull().sum().sum())
    print(all_questions)

    print('Time saving questions into dataset: ', timedelta(seconds=time() - initial_time))

    all_questions.to_csv('../data/All_Questions.csv')
