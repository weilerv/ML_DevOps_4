# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle, os

# Add the necessary imports for the starter code.
import pandas as pd
import logging
from .ml.model import train_model, compute_model_metrics, inference, perf_slices
from .ml.data import process_data

logging.basicConfig(filename='logging.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')
# Add code to load in the data.
data = pd.read_csv('data/cleaned_census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True,
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# check if trained model exists
model_path = r'C:\Users\z0176083\OneDrive - ZF Friedrichshafen AG\Documents\Udacity\ML_DevOps\Section_4_deployment\nd0821-c3-starter-code-master\starter\model'
#if model exists, load the model
if os.path.isfile(os.path.join(model_path,'trained_model.pkl')):
    model = pickle.load(open(os.path.join(model_path,'trained_model.pkl'), 'rb'))
    encoder = pickle.load(open(os.path.join(model_path,'encoder.pkl'), 'rb'))
    lb = pickle.load(open(os.path.join(model_path,'labelizer.pkl'), 'rb'))
# Train and save a model.
else:
    model = train_model(X_train,y_train)
    pickle.dump(model, open(os.path.join(model_path,'trained_model.pkl'),'wb'))
    pickle.dump(encoder, open(os.path.join(model_path,'encoder.pkl'),'wb'))
    pickle.dump(lb, open(os.path.join(model_path,'labelizer.pkl'),'wb'))
    logging.info(f"Model saved to: {model_path}")


#evaluate trained model on test set
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(
    f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")

print(test)
#compute performance on categorical slices. save results to slice_output.txt
perf_slices(test, encoder, lb, model, cat_features, save_slice=True)
logging.info(f"Performance on slices done")
