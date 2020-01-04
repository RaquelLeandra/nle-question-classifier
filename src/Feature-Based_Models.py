import re
from collections import Counter
import pandas as pd
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from src.data_analysis import clean_word
#from src.t1_type_of_question_classifier import train_model
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
question_types = ['summary', 'list', 'yesno', 'factoid']

def get_first_element(phrase):
    return phrase.split()[0].lower()

def get_last_element(phrase):
    return phrase.split()[-1].lower()

def get_question_mark(word):
    return list(word)[-1] == '?'

def remove_question_mark(word):
    if(get_question_mark(word)):
        return word[:-1]
    else:
        return word
def get_length_phrase(phrase):
    return len(phrase.split())

def clean_word(word):
    word = word.lower()
    word = re.sub(r'[^\w]', '', word)
    return word

def important_words(questions, max_num_words=100):
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
    total_occurrences = df_most_common['summary']+df_most_common['list']+df_most_common['yesno']+df_most_common['factoid']
    df_important_words = df_most_common[(df_most_common['summary'] >= 0.45*total_occurrences) |
                                        (df_most_common['list'] >= 0.45*total_occurrences) |
                                        (df_most_common['yesno'] >= 0.45*total_occurrences) |
                                        (df_most_common['factoid'] >= 0.45 * total_occurrences) |
                                        (df_most_common['summary'] <= 0.05 * total_occurrences) |
                                        (df_most_common['list'] <= 0.05 * total_occurrences) |
                                        (df_most_common['yesno'] <= 0.05 * total_occurrences) |
                                        (df_most_common['factoid'] <= 0.05 * total_occurrences)]
    print(df_important_words)
    return df_important_words.index.values

def check_word_exists_in(phrase,important_word):
        return important_word in phrase.split()

def train_model(model, X_train, X_test, y_train, y_test, target_names=['summary', 'list', 'yesno', 'factoid']):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #cr = metrics.classification_report(y_test, y_pred,
     #                                  target_names=target_names)
    #print(cr)
    acc = model.score(X_test, y_test)
    return acc

if __name__ == '__main__':
    questions = pd.read_excel('../data/Questions.xlsx')
    features=pd.DataFrame()
    features_train=pd.DataFrame()
    features_test = pd.DataFrame()
    #Shuffle questions dataset
    questions=questions.sample(frac=1)
    questions_train = questions.iloc[:round(0.8 * questions.shape[0]),:]
    questions_test = questions.iloc[round(0.8 * questions.shape[0]):, :]
    # First word
    features_train['first_word'] = questions_train['Question'].apply(get_first_element)
    features_test['first_word'] = questions_test['Question'].apply(get_first_element)

    # Last word
    features_train['last_word'] = questions_train['Question'].apply(get_last_element)
    features_test['last_word'] = questions_test['Question'].apply(get_last_element)

    # Has question mark?
    features_train['has_question_mark'] = features_train['last_word'].apply(get_question_mark)*1
    features_test['has_question_mark'] = features_test['last_word'].apply(get_question_mark)*1

    # Last word without question mark
    features_train['last_word'] = features_train['last_word'].apply(remove_question_mark)
    features_test['last_word'] = features_test['last_word'].apply(remove_question_mark)

    # Get average length of the dataset
    average_len_train = questions_train['Question'].apply(get_length_phrase).mean()

    #classify phrase into short, long
    features_train['long_phrase'] = questions_train['Question'].apply(get_length_phrase)>average_len_train*1
    features_test['long_phrase'] = questions_test['Question'].apply(get_length_phrase) > average_len_train*1

    #iImportant words
    important_words=important_words(questions_train)
    for important_word in important_words:
        features_train[important_word] = questions_train['Question'].apply(check_word_exists_in, important_word=important_word)*1
        features_test[important_word] = questions_test['Question'].apply(check_word_exists_in, important_word=important_word)*1

    #Train first and last words to one hot
    one_hot_train = pd.get_dummies(features_train['first_word'],prefix='first_word').join(pd.get_dummies(features_train['last_word'],prefix='last_word'))
    features_train = features_train.drop(['first_word','last_word'], axis=1)
    features_train.join(one_hot_train)
    #Test first and last words to one hot
    one_hot_test = pd.get_dummies(features_test['first_word'],prefix='first_word').join(pd.get_dummies(features_test['last_word'],prefix='last_word'))
    features_test = features_test.drop(['first_word','last_word'], axis=1)
    features_test.join(one_hot_test)

    X_train = features_train.values
    y_train = questions_train['Type'].values
    X_test = features_test.values
    y_test = questions_test['Type'].values

    print(X_train)
    print(y_train)
    model = LinearSVC()
    acc = train_model(model, X_train, X_test, y_train, y_test)
    print(acc)
