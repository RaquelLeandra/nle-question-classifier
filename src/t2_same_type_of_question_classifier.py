import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

data_analysis_path = '../results/data_analysis/'


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def extract_categories(df):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),)
                            # stop_words='english')
    df['category_id'] = df['Type'].factorize()[0]
    category_id_df = df[['Type', 'category_id']].drop_duplicates().sort_values('category_id')
    labels = df.category_id
    return labels, category_id_df

def extract_features_all(df_qt,df_quora):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), )
    # stop_words='english')

    all_questions=np.concatenate((df_qt.Question.values.astype('U'),df_quora.question1.values.astype('U'),df_quora.question2.values.astype('U')),axis=0)
    features = tfidf.fit_transform(all_questions).toarray()
    return features

def split_features(features,num_qt,num_quora):
    features_qt=features[0:num_qt][:]
    features_1 = features[num_qt:num_qt+num_quora][:]
    features_2 = features[num_qt+num_quora:num_qt+2*num_quora][:]
    return features_qt,features_1, features_2


if __name__ == '__main__':
    df = pd.read_excel('../data/Questions.xlsx')
    quora_df = pd.read_csv('../data/quora_with_med_questions.csv')
    quora_df=quora_df.iloc[0:10000, :]
    print(df.shape)
    print(quora_df.shape)


    labels, category_id_df = extract_categories(df)
    features = extract_features_all(df,quora_df)
    features_qt,features_1,features_2 = split_features(features, df.shape[0], quora_df.shape[0])
    #We use all the original dataset for training
    X_train = features_qt
    X_test_1 = features_1
    X_test_2 = features_2
    y_train = labels

    models = [
        RandomForestClassifier(n_estimators=200),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    models_names = ['RF', 'LinearSVC', 'Multinomial', 'LogisticRegression']


    types=['summary','list','yesno','factoid']
    model=LinearSVC()
    model_name='LinearSVC'
    print(model_name)
    trained_model = train_model(model, X_train, y_train)
    y_pred_1 = trained_model.predict(X_test_1)
    y_pred_2 = trained_model.predict(X_test_2)
    all_questions = df
    for i in range(y_pred_1.shape[0]):
        if y_pred_1[i]==y_pred_2[i]:
            question1=pd.DataFrame([[quora_df.iloc[i,3],types[y_pred_1[i]],y_pred_1[i]]],columns=['Question','Type','category_id'])
            question2 = pd.DataFrame([[quora_df.iloc[i, 4], types[y_pred_1[i]], y_pred_1[i]]], columns=['Question', 'Type', 'category_id'])
            all_questions=all_questions.append(question1)
            all_questions=all_questions.append(question2)

    all_questions.to_csv('All_Questions.csv')
