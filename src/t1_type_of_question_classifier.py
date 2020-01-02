import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sns.set(style="darkgrid")

data_analysis_path = '../results/data_analysis/'


def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(metrics.classification_report(y_test, y_pred,
                                        target_names=df['Type'].unique()))
    acc = model.score(X_test, y_test)
    return acc


def extract_features(X_train, X_test):
    tfidf = TfidfVectorizer(sublinear_tf=True, max_features=900, min_df=5, norm='l2', encoding='latin-1',
                            ngram_range=(1, 2), )
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

    print(df)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(df['Question'], df['category_id'],
                                                                                     df.index,
                                                                                     test_size=0.2, random_state=0)
    features_train, features_test = extract_features(X_train, X_test)

    models = [
        RandomForestClassifier(n_estimators=200),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    models_names = ['RF', 'LinearSVC', 'Multinomial', 'LogisticRegression']

    accuracies = []

    for model, model_name in zip(models, models_names):
        print(model_name)
        acc = train_model(model, features_train, features_test, y_train, y_test)
        accuracies.append(acc)

    sns.barplot(x=models_names, y=accuracies)
    plt.ylabel('Accuracies')
    plt.ylim(0, 1)
    plt.savefig(data_analysis_path + 'barplot_' + 'baseline_models_with_stop_words', bbox_inches='tight', dpi=200)
