from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier

from .data import process_data

logging.basicConfig(filename='logging.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
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
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

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
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def perf_slices(df_test: pd.DataFrame, encoder, lb, model, cat_features, save_slice: bool = True):
    """Compute performance on slices for given categorical feature.

    Returns
    -----
    df with
        feature: str
        value: str
        n_samples: integer
        precision: float
        recall: float
        fbeta: float

    """ 
    cat_list = []
    class_list = []
    n_sample_list = []
    precision_list = []
    recall_list = []
    fbeta_list = []

    df_slice = pd.DataFrame(columns=['feature', 'class', 'n_samples', 'precision', 'recall', 'fbeta'])
    for cat in cat_features:
        for cls in df_test[cat].unique():
            df_temp = df_test[df_test[cat]==cls]
            #process test data
            X_test, y_test, _,_ = process_data(
                df_temp, categorical_features=cat_features, label="salary", 
                training=False, encoder=encoder, lb=lb
            )

            preds = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)

            cat_list.append(cat)
            class_list.append(cls)
            n_sample_list.append(len(y_test))
            precision_list.append(precision)
            recall_list.append(recall)
            fbeta_list.append(fbeta)


    df_slice = pd.DataFrame({'feature': cat_list, 'class': class_list, 'n_samples': n_sample_list, 'precision': precision_list, 'recall':recall_list, 'fbeta':fbeta_list})

    if save_slice:
        df_slice.to_csv('slice_output.txt', index=False, sep='\t')

    return df_slice

