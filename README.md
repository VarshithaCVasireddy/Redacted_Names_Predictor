# cs5293sp22-project3
## Author: Varshitha Choudary Vasireddy

## Description of the project:
Whenever sensitive information is shared with the public, the data must go through a redaction process. That is, all sensitive names, places, and other sensitive information must be hidden. Documents such as police reports, court transcripts, and hospital records all containing sensitive information. Redacting this information is often expensive and time consuming.
<br/>
For project 3, I created an Unredactor. The unredactor will take redacted documents where name of a person is redacted and it returns the most likely candidates to fill in the redacted location.

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
~~~

So as part of this project I used sklearn and pandas external libraries. And from sklearn I imported functions.

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
In this function columns of dataframe are checked. "type" is the dataframe column that is checked. If the "type" is training then Train dataframe is created. If it is validation, then Valid dataframe is created. If it is testing then Test dataframe is created.
~~~
def train_test_valid(RAW_DATA):
    Train = RAW_DATA[RAW_DATA["type"] == "training"]
    Valid = RAW_DATA[RAW_DATA["type"] == "validation"]
    Test = RAW_DATA[RAW_DATA["type"] == "testing"]

    return Train, Valid, Test
~~~
I am returning the 3 dataframes in this function. Dataframes consisting of Training, validation and testing are returned.

### fit_prediction_models
In this function I am firstly vectorizing the data that the model is to be trained and also determing the f1, recall and precision score. Firstly on all the sentence type of data i.e RAW_Data's sentence column TFIDF vectorization is done. And X_train is created by TFIDF transfrom vectorization of Train dataframe. X_test is created by TFIDF transfrom vectorization of Test dataframe. Label encoding is done on the to be predicted data i.e name, so label encoding fit and transformation is done on all name column data. To get y_train and y_test, label encoding transformation is done on name column data on train and test dataframes. I used MLFClassifier to predict the results. I fitted the model with X_train and y_train, i.e training data. F1, recall and precision score is checked between the y_test i.e testing data and model prediction data.
~~~
def fit_prediction_models(Train, Test):
    X = TFIDF.fit_transform(RAW_DATA.sentence)
    y = LE.fit_transform(RAW_DATA.name)
    X_train = TFIDF.transform(Train.sentence)
    X_test = TFIDF.transform(Test.sentence)
    y_train = LE.transform(Train.name)
    y_test=LE.transform(Test.name)
    clf = MLPClassifier(random_state=1).fit(X_train, y_train)
    clf_preds = clf.predict(X_test)
    f1 = f1_score(y_test,clf_preds, average = 'macro')
    r_score = recall_score(y_test,clf_preds, average = 'macro')
    p_score = precision_score(y_test,clf_preds, average = 'macro')
    return clf, f1, r_score, p_score
~~~
In this function I am returning MLPClassifier model, f1 score, recall score and precision score. <br/>
Referred: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html, https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html, https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

### get_model
For the persistence of the models that I used I wrote below code. This helps in models not be retrained always. Firstly I checked if the models are existing in the path, then I dummped the models. I used python built-in persistence model, namely pickle. Pickle dump and load functionality are used to do this. If models are not present then "dump" is used to create models. If models are already present then they are loaded.
~~~
def get_model(clf):
    if not os.path.exists(model['clf']):
        models_fldr = os.path.join(os.getcwd(), "model")
        if not os.path.exists(models_fldr):
            os.mkdir(models_fldr)

        randomforest = fit_prediction_models()
        with open(model['clf'], 'wb') as file_f:
            pickle.dump(clf, file_f)
        
        return randomforest
    else:
        _, _ = get_yummly_data()
        with open(model['clf'], 'rb') as file_f:
            return pickle.load(file_f)
~~~
Referred https://scikit-learn.org/stable/modules/model_persistence.html

### project3.py
Packages imported in this file are


### main



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
In this test, it is checked if the test_train_test_valid() functionality is working properly. As seen above test_train_test_valid() function return's 3 dataframes, so in this test it is checked if the returned data is dataframes.
~~~
def test_train_test_valid(RAW_DATA):
    actual_Train, actual_Valid, actual_Test = train_test_valid(RAW_DATA)
    assert isinstance(actual_Train,pd.DataFrame) == True
~~~

### test_fit_prediction_models
In this test, it is checked if the fit_prediction_models() functionality is working properly. As seen above fit_prediction_models() function return's model, f1, recall and precision score, so in this test it is checked if scores are not equal to zero.
~~~
def test_fit_prediction_models(Train, Test):
    actual_clf, actual_f1, actual_rscore, actual_pscore = fit_prediction_models(Train, Test)
    assert actual_f1 ! = 0
~~~

## Assumptions/Bugs
- I am getting very less scores.
- I trained the model with the training data and tested the model with testing data.
- I didn't use the validation data.

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
pipenv run python project3.py
~~~
- **Step4** 

Then run the below command to test the testcases. 

~~~
 pipenv run python -m pytest -v
~~~