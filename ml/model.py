import logging
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from joblib import dump
import sys

if __package__ is None or __package__ == '':
    from data import process_data
else:
    from .data import process_data


LOG_FILE = './logs/census.log'
SLICE_OUTPUT = './artifacts/slice_output.txt'
MODEL_FILENAME = './artifacts/model.joblib'
ENCODER_FILENAME = './artifacts/encoder.joblib'
LB_FILENAME = './artifacts/lb.joblib'

# Define logging handlers
logFileHandler = logging.FileHandler(LOG_FILE)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logFileHandler,
        consoleHandler
    ]
)


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    logging.info("[ START: Training... ]")
    logRegres = LogisticRegression(max_iter=1000, random_state=42)
    logRegres.fit(X_train, y_train)
    logging.info("[ FINISH: Training successful ]")
    return logRegres


def calc_metrics(cat_features, model, y_test, y_pred, test_data, encoder, lb):
    """
    Calculates metrics of the model

    Inputs
    ------
    cat_features : list
        List of categorical feature names.
    model : sklearn.linear_model.LogisticRegression
        Trained model.
    y_test : np.array
        Labels.
    y_pred : np.array
        Predicted Labels.
    test_data : pd.DataFrame
        Test data.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted encoder for values of category features.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted label binarizer.
    Returns
    -------
    None
    """
    precision, recall, fbeta = _compute_model_metrics(y_test, y_pred)
    logging.info(f"[ METRICS: Precision: {precision}, Recall: {recall}, Fbeta: {fbeta} ]")

    metrics = []
    for cat in cat_features:
        for catVar in test_data[cat].unique():
            logging.debug(f"cat {cat}, catVar {catVar}")
            slice_df = test_data[test_data[cat] == catVar]
            X_slice, y_slice, _, _ = process_data(
                slice_df, categorical_features=cat_features,
                label='salary', training=False, encoder=encoder, lb=lb)
            y_slice_pred = model.predict(X_slice)
            precision, recall, fbeta = _compute_model_metrics(y_slice,
                                                              y_slice_pred)
            metrics.append(f"[ {cat}: {catVar} ] --- Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")
    with open(SLICE_OUTPUT, 'w') as file:
        file.write('\n'.join(metrics))
    logging.info(f"[ FILE: Metrics written to {SLICE_OUTPUT} ]")


def _compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logging.info(f"[ MODEL: Run inference with input: {X} ]")
    pred = model.predict(X)
    return pred


def save_model(model, encoder, lb):
    """ Save model/encoder/label binarizer to file.

    Inputs
    ------
    model : sklearn.linear_model.LogisticRegression
        Trained model.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted encoder for values of category features.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted label binarizer.
    Returns
    -------
    None
    """
    dump(model, MODEL_FILENAME)
    logging.info(f"[ FILE: Model saved to file {MODEL_FILENAME} ]")
    dump(encoder, ENCODER_FILENAME)
    logging.info(f"[ FILE: Encoder saved to {ENCODER_FILENAME} ]")
    dump(lb, LB_FILENAME)
    logging.info(f"[ FILE: Label binarizer saved to {LB_FILENAME} ]")
