import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

LE = LabelEncoder()
TFIDF = TfidfVectorizer()

def data_reading():
    data_url = "https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    RAW_DATA = pd.read_csv(data_url,sep="\t", header=None, on_bad_lines="skip")
    RAW_DATA.columns = ["user", "type", "name", "sentence"]
    return RAW_DATA

def train_test_valid():
    RAW_DATA = data_reading()
    Train = RAW_DATA[RAW_DATA["type"] == "training"]
    Valid = RAW_DATA[RAW_DATA["type"] == "validation"]
    Test = RAW_DATA[RAW_DATA["type"] == "testing"]

    TFIDF.fit(RAW_DATA.sentence)
    LE.fit(RAW_DATA.name)
    X_train = TFIDF.transform(Train.sentence)
    X_test = TFIDF.transform(Test.sentence)
    y_train = LE.transform(Train.name)
    y_test=LE.transform(Test.name)

    return X_train, X_test, y_train, y_test

def fit_prediction_models():
    X_train, X_test, y_train, y_test = train_test_valid()    

    clf = MLPClassifier(random_state=1).fit(X_train, y_train)
    clf_preds = clf.predict(X_test)
    f1, r_score, p_score = get_metrics(y_test, clf_preds)
    
    return clf, f1, r_score, p_score
    # X_train, X_test, y_train, y_test = train_test_valid()    

    # clf = RandomForestClassifier(random_state=1).fit(X_train, y_train)
    # clf_preds = clf.predict(X_test)
    # f1, r_score, p_score = get_metrics(y_test, clf_preds)
    
    # return clf, f1, r_score, p_score

def get_metrics(y_actual, y_pred):
    f1 = f1_score(y_actual,y_pred, average = 'macro', zero_division = 1)
    r_score = recall_score(y_actual,y_pred, average = 'macro', zero_division = 1)
    p_score = precision_score(y_actual,y_pred, average = 'macro', zero_division = 1)

    return f1, r_score, p_score


def get_model(clf_path):
    if not os.path.exists(clf_path):
        models_fldr = os.path.join(os.getcwd(), "model")
        if not os.path.exists(models_fldr):
            os.mkdir(models_fldr)

        clf, f1, r_score, p_score = fit_prediction_models()
        with open(clf_path, 'wb') as file_f:
            pickle.dump(clf, file_f)

        
        
        return clf, f1, r_score, p_score 
    else:
        with open(clf_path, 'rb') as file_f:
            clf = pickle.load(file_f)
            _, X_test, _, y_test = train_test_valid()   

            f1, r_score, p_score = get_metrics(y_test, clf.predict(X_test))

            return clf, f1, r_score, p_score
