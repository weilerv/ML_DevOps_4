# ML unit test
# Write at least 3 unit tests. Unit testing ML can be hard due to the stochasticity -- at least test if any ML functions return the expected type.

import pytest
import os
import logging
import pandas as pd
import pickle
from .ml.model import inference, compute_model_metrics

def test_import_data(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err
    
    try: 
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Dataset appears not to have rows or columns")
        raise err
    
def test_inference(train_dataset):
    X_train, y_train = train_dataset

    if os.path.isfile("./model/trained_model.pkl"):
        model = pickle.load(open("./model/trained_model.pkl", 'rb'))

        try:
            preds = inference(model,X_train)
        except Exception as err:
            logging.error("Error in inference")
            raise err

def test_compute_model_metrics(df):
    X_train, y_train = df
    if os.path.isfile("./model/trained_model.pkl"):
        model = pickle.load(open("./model/trained_model.pkl", 'rb'))
        preds = inference(model, X_train)

        try:
            precision, recall, dbeta = compute_model_metrics(y_train, preds)
        except Exception as err:
            logging.error("Error while computing performance metrics")
            raise err

        
