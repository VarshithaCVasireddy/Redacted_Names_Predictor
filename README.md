# cs5293sp22-project3
## Author: Varshitha Choudary Vasireddy

## Description of the project:
Whenever sensitive information is shared with the public, the data must go through a redaction process. That is, all sensitive names, places, and other sensitive information must be hidden. Documents such as police reports, court transcripts, and hospital records all containing sensitive information. Redacting this information is often expensive and time consuming.
<br/>
For project 3, I created an Unredactor. The unredactor will take redacted documents where name of a person is redacted and it returns the most likely candidates to fill in the redacted location. I also displayed the f1 score, recall score and precision score.

## Structure of project
2 python files are present in this project, project3.py and predictor.py. <br/>
In predictor.py all the functionality is done. <br/>
In project3.py, predictor.py file is called. 

### predictor.py
Below Packages imported are in order to run this program code and to use their functionalities

~~~
import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
~~~

So as part of this project I used sklearn and pandas external libraries. And from sklearn I imported functions. <br/>
- os is imported in order to get the directory path.
- pickle is used to build model persistency.
- pandas is used for creating dataframes and to read data.
- TfidfVectorizer, LabelEncoder are used to convert data into vectors.
- precision_score, recall_score, f1_score are imported to find f1, recall and precision score
- MLPClassifier is imported to use MLPClassifier model.
- RandomForestClassifier is imported to use RandomForestClassifier

### data_reading
In this function the latest data from the "unredactor.tsv" will be read. I am reading the url of the "unredactor.tsv" file into pandas dataframe and I separated the data with tab space and I skipped the bad lines of data. I named the columns for each dataframe column.
~~~
def data_reading():
    data_url = "https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    RAW_DATA = pd.read_csv(data_url,sep="\t", header=None, on_bad_lines="skip")
    RAW_DATA.columns = ["user", "type", "name", "sentence"]
    return RAW_DATA
~~~
I am returning the dataframe in this function. This dataframe consists of all the "unredactor.tsv" file data.

Referred: https://datascientyst.com/drop-bad-lines-with-read_csv-pandas/

### train_test_valid
In this function columns of dataframe are checked. "type" is the dataframe column that is checked. If the "type" is training then Train dataframe is created. If it is validation, then Valid dataframe is created. If it is testing then Test dataframe is created. I am firstly vectorizing the data. Firstly on all the sentence type of data i.e RAW_Data's sentence column TFIDF vectorization is done. And X_train is created by TFIDF transfrom vectorization of Train dataframe. X_test is created by TFIDF transfrom vectorization of Test dataframe. Label encoding is done on the to be predicted data i.e name, so label encoding fit and transformation is done on all name column data. To get y_train and y_test, label encoding transformation is done on name column data on train and test dataframes. I used MLFClassifier to predict the results. I fitted the model with X_train and y_train, i.e training data.
~~~
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
~~~
I am returning 4 sparse matrices. Where 2 are training matrices and 2 are testing matrices. <br/>
Referred: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html, https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

### get_metrics()
In this method I am determining f1 score, recall score and precision score. To get the score I imported "precision_score, recall_score, f1_score" from "sklearn.metrics". I used to functions to get the score.
~~~
def get_metrics(y_actual, y_pred):
    f1 = f1_score(y_actual,y_pred, average = 'macro')
    r_score = recall_score(y_actual,y_pred, average = 'macro')
    p_score = precision_score(y_actual,y_pred, average = 'macro')

    return f1, r_score, p_score
~~~
I am returning f1 score, recall score and precision score.<br/>
Referred: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

### fit_prediction_models
I am calling the train_test_valid() function and named the arguments that are returned as X_train, X_test, y_train, y_test. MLPClassifier model is created then. From get_metrics() method I got f1 score, recall score and precision score. To get_metrics() method I inputted y_test data and model prediction.
~~~
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
~~~
Sometimes MLPClassifier is failing so I am also attaching the code for RandomForestClassifier.<br/>
In this function I am returning MLPClassifier model, f1 score, recall score and precision score. <br/>
Referred: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html,  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

### get_model
For the persistence of the models that I used I wrote below code. This helps in models not be retrained always. Firstly I checked if the models are existing in the path, then I dummped the models. I used python built-in persistence model, namely pickle. Pickle dump and load functionality are used to do this. If models are not present then "dump" is used to create models. If models are already present then they are loaded.
~~~
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
~~~
model, f1 score, recall score and precision score are returned as part of this function. <br/>
Referred https://scikit-learn.org/stable/modules/model_persistence.html

### project3.py
Packages imported in this file are
~~~
import os
import argparse
import predictor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
~~~
- argparse is used to get command line arguments from the user
- LabelEncoder and TfidVectorizer are imported to vectorize data.
- predictor.py file is imported to get their methods.

### main
an argument is to be passed for sure, and it should be a sentence and should be given inside double quotes. I am sending the arg that I got to predict_name function. 
~~~
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence',required = True, type = str,help='Redacted sentence is to be given.')
    args = parser.parse_args()
    predict_name(args)
~~~
Referred: https://docs.python.org/3/library/argparse.html

### predict_name
model's path where it is used is given. And it is given as input to get_model() function of predictor.py file. Model, f1 score, recall score and precision score will be returned by that function and they are stored in variables. Whole data is taken and tfid vectorization is done on whole data's sentences and label encoding is done on whole data's name set. And the sentence that is inputted by the user is converted into arrya and to that array I performed tfidf operation. I am sending that sentence in the model and predicting the output. The output is in vectors form, so I used label encoder classes to get back the name that is predicted. I printed the f1 score, recall score, precision score and the sentence that is inputted and the name that is predicted.
~~~
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
~~~
Referred: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

## **Tests**
In the tests I tested all the 5 functions functionalities of predictor.py file.

### test_data_reading
In this test, it is checked if the data_reading() functionality is working properly. As seen above data_reading() function return's dataframe, so in this test it is checked if the returned data is dataframe.
~~~
def test_data_reading():
    actual_data = data_reading()
    assert isinstance(actual_data,pd.DataFrame) == True
~~~

### test_train_test_valid
In this test, it is checked if the test_train_test_valid() functionality is working properly. As seen above test_train_test_valid() function return's 4 sparse matrices, so in this test it is checked if the returned data shape is not (0,0) as it holds somedata in it.
~~~
def test_train_test_valid():
    X_train, X_test, y_train, y_test = train_test_valid()
    assert X_train.shape != (0,0)
    assert X_test.shape != (0,0)
    assert y_train.shape != (0,0)
    assert y_test.shape != (0,0)
~~~

### test_get_metrics
In this test, it is checked if the get_metrics() functionality is working properly. I gave two arrays as inputs to the function. We get f1 score, recall score, precision score as outputs of this function. I am checking if the scores are not equal to zero.
~~~
def test_get_metrics():
    f1, r_score, p_score = get_metrics([0, 1, 2, 0, 1, 2],[0, 2, 1, 0, 0, 1])
    assert f1 != 0
    assert r_score != 0
    assert p_score != 0
~~~

### test_get_model
In this test, it is checked if the get_model() functionality is working properly. I am sending the model's saved path as input to get_model. function return's model, f1, recall and precision score, so in this test it is checked if scores are not equal to zero and model is not None.
~~~
def test_get_model():
    clf_path = os.path.join(os.getcwd(), "model", "clf.pickle")
    actual_clf, actual_f1, actual_rscore, actual_pscore = get_model(clf_path)
    assert actual_clf is not None
    assert actual_f1 != 0
    assert actual_rscore != 0
    assert actual_pscore != 0
~~~


### test_fit_prediction_models
In this test, it is checked if the fit_prediction_models() functionality is working properly. As seen above fit_prediction_models() function return's model, f1, recall and precision score, so in this test it is checked if scores are not equal to zero and model is not None.
~~~
def test_fit_prediction_models():
    actual_clf, actual_f1, actual_rscore, actual_pscore = fit_prediction_models()
    assert actual_clf is not None
    assert actual_f1 != 0
    assert actual_rscore != 0
    assert actual_pscore != 0
~~~

## Assumptions/Bugs
- I am getting very less scores.
- I trained the model with the training data and tested the model with testing data.
- I didn't use the validation data.
- The predicted names are not accurate.
- A sentence has to be given for sure as an input parameter in order to get the name prediction as output.
- Sometimes the code is failing for MLPClassifier model so I also wrote code with RandomForestClassifier, if one is not working please comment it and uncomment the commented code to make it work

## **Steps to Run project1**

- **Step1**  
Clone the project directory using below command

~~~json
git clone https://github.com/VarshithaCVasireddy/cs5293sp22-project3
~~~
Navigate to directory that we cloned from git and run the below command to install dependencies
<br/>

-**Step2**
Run below command to install pipenv
~~~
pipenv install
~~~

- **Step3**  
To check the code run the below command 
~~~
pipenv run python project3.py --sentence "I'll admit that I was reluctant to see it because from what I knew of ██████████████ he was only able to do comedy"
~~~
- **Step4** 

Then run the below command to run and test the testcases. 

~~~
 pipenv run python -m pytest -v
~~~