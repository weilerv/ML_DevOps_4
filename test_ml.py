# ML unit test
from starter.ml.model import train_model, compute_model_metrics, inference, perf_slices
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split
import logging
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

logging.basicConfig(filename = "./testing_logging.log",
                    level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

data = pd.read_csv('data/cleaned_census.csv')

train, test = train_test_split(data, test_size=0.20)


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb,)




def test_process_data():
    try:
        assert X_train.shape[0] == len(y_train)
    except AssertionError as err:
        logging.info("Different number of columns in train and test set")
        raise err
    
def test_compute_model_metrics():
    trained_model = train_model( X_train, y_train)
    preds = inference(trained_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert 0 <= precision <= 1, 'Precision value is incorrect'
    assert 0 <= recall <= 1, 'Recall value is incorrect'
    assert 0 <= fbeta <= 1, 'Fbeta value is incorrect'

def test_inference():
    trained_model = train_model(X_train, y_train)
    preds = inference(trained_model, X_test)

    for pred in preds:
        assert pred == 0 or pred == 1, 'Model prediction incprrect.'
