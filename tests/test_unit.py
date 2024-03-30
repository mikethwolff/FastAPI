from unittest.mock import Mock, patch
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, inference, save_model, MODEL_FILENAME, ENCODER_FILENAME, LB_FILENAME
from ml.train_model import DATA_FILE, CAT_FEATURES
from ml.data import process_data

@patch('ml.model.dump')
def testSaveModel(mockDump):
    lr_model_mock = Mock()
    encoder_mock = Mock()
    lb_mock = Mock()
    save_model(lr_model_mock, encoder_mock, lb_mock)
    mockDump.assert_any_call(lr_model_mock, MODEL_FILENAME)
    mockDump.assert_any_call(encoder_mock, ENCODER_FILENAME)
    mockDump.assert_called_with(lb_mock, LB_FILENAME)

def testInference():
    model_mock = Mock()
    X_mock = Mock()
    pred = inference(model_mock, X_mock)
    assert pred is not None
    model_mock.predict.assert_called_with(X_mock)

def testInferenceReturnType():
    data = pd.read_csv(DATA_FILE)
    train, test = train_test_split(data, test_size=0.20, stratify=data['salary'])
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    X_test, _, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label='salary',
        training=False, encoder=encoder, lb=lb)
    lr_model = train_model(X_train,y_train)
    pred = inference(lr_model, X_test)
    assert isinstance(pred, ndarray)

def testTrainModelReturnType():
    data = pd.read_csv(DATA_FILE)
    train, _ = train_test_split(data, test_size=0.20, stratify=data['salary'])
    X_train, y_train, _, _ = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    lr_model = train_model(X_train,y_train)
    assert isinstance(lr_model, LogisticRegression)
