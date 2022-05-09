import os
import argparse
import predictor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

LE = LabelEncoder()
TFIDF = TfidfVectorizer()

def predict_name(args):
    clf_path = os.path.join(os.getcwd(), "model", "clf.pickle")
    clf, f1, r_score, p_score = predictor.get_model(clf_path)
    RAW_DATA = predictor.data_reading()
    TFIDF.fit(RAW_DATA.sentence)
    LE.fit(RAW_DATA.name)
    sentence = args.sentence
    df = pd.Series(sentence)
    sent = TFIDF.transform(df)
    pre = clf.predict(sent)
    predicted_name = LE.classes_[pre[0]]
    print(f"\nThe f1 score is {f1}")
    print(f"\nThe recall score is {r_score}")
    print(f"\nThe precision score is {p_score}")
    print(f"\nThe sentence that you gave is -- {sentence} -- and the name predicted is -- {predicted_name} -- \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence',required = True, type = str,help='Redacted sentence is to be given.')
    args = parser.parse_args()
    predict_name(args)