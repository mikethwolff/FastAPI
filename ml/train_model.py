import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import sys

if __package__ is None or __package__ == '':
    from data import process_data
    from model import train_model, calc_metrics, save_model, LOG_FILE
else:
    from .data import process_data
    from .model import train_model, calc_metrics, save_model, LOG_FILE

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

CAT_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

DATA_FILE = './data/census_clean.csv'


# Train and save a model.
def train_and_save_model():
    # Fetch raw data
    data = pd.read_csv(DATA_FILE)

    # Split data
    # Optional enhancement, use K-fold cross validation
    # instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20,
                                   stratify=data['salary'])
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label='salary',
        training=False, encoder=encoder, lb=lb)

    # Train model
    model = train_model(X_train, y_train)

    # Save model
    y_pred = model.predict(X_test)
    calc_metrics(CAT_FEATURES, model, y_test, y_pred, test, encoder, lb)
    save_model(model, encoder, lb)


if __name__ == '__main__':
    train_and_save_model()
