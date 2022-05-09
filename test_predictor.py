from types import NoneType
import pytest
import os
from predictor import *

def test_data_reading():
    actual_data = data_reading()
    assert isinstance(actual_data,pd.DataFrame) == True

def test_train_test_valid():
    X_train, X_test, y_train, y_test = train_test_valid()
    assert X_train.shape != (0,0)
    assert X_test.shape != (0,0)
    assert y_train.shape != (0,0)
    assert y_test.shape != (0,0)

def test_get_metrics():
    f1, r_score, p_score = get_metrics([0, 1, 2, 0, 1, 2],[0, 2, 1, 0, 0, 1])
    assert f1 != 0
    assert r_score != 0
    assert p_score != 0


def test_get_model():
    clf_path = os.path.join(os.getcwd(), "model", "clf.pickle")
    actual_clf, actual_f1, actual_rscore, actual_pscore = get_model(clf_path)
    assert actual_clf is not None
    assert actual_f1 != 0
    assert actual_rscore != 0
    assert actual_pscore != 0

def test_fit_prediction_models():
    actual_clf, actual_f1, actual_rscore, actual_pscore = fit_prediction_models()
    assert actual_clf is not None
    assert actual_f1 != 0
    assert actual_rscore != 0
    assert actual_pscore != 0