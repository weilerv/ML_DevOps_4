from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier

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


def perf_slices(df, feature, y, preds):
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
    feature_list = []
    class_list = []
    n_sample_list = []
    precision_list = []
    recall_list = []
    fbeta_list = []

    feature_values = df[feature].unique().tolist()

    df_slice = pd.DataFrame(columns=['feature', 'class', 'n_samples', 'precision', 'recall', 'fbeta'])

    for cls in feature_values:
        feature_ind = df[feature] == cls
        y_feature = y[feature_ind]
        preds_feature = preds[feature_ind]
        precision, recall, fbeta = compute_model_metrics(y_feature, preds_feature)

        feature_list.append(feature)
        class_list.append(cls)
        n_sample_list.append(len(y_feature))
        precision_list.append(precision)
        recall_list.append(recall)
        fbeta_list.append(fbeta)


    df_slice = pd.DataFrame({'feature': feature_list, 'class': class_list, 'n_samples': n_sample_list, 'precision': precision_list, 'recall':recall_list, 'fbeta':fbeta_list})

    return df_slice

