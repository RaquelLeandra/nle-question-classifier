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

data_analysis_path = '../results/data_analysis/'


def train_model(model, X_train, X_test, y_train, y_test, category_id_df):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    print(metrics.classification_report(y_test, y_pred,
                                        target_names=df['Type'].unique()))
    acc = model.score(X_test, y_test)
    return acc


def extract_features(df):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),)
                            # stop_words='english')
    df['category_id'] = df['Type'].factorize()[0]

    category_id_df = df[['Type', 'category_id']].drop_duplicates().sort_values('category_id')
    features = tfidf.fit_transform(df.Question).toarray()
    labels = df.category_id
    return features, labels, category_id_df


if __name__ == '__main__':
    df = pd.read_excel('../data/Questions.xlsx')

    features, labels, category_id_df = extract_features(df)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                     test_size=0.2,random_state=0)

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
        acc = train_model(model, X_train, X_test, y_train, y_test, category_id_df)
        accuracies.append(acc)

    sns.barplot(x=models_names, y=accuracies)
    plt.ylabel('Accuracies')
    plt.ylim(0,1)
    plt.savefig(data_analysis_path + 'barplot_' + 'baseline_models_with_stop_words', bbox_inches='tight', dpi=200)
