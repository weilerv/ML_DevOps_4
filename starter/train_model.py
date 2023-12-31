# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle, os

# Add the necessary imports for the starter code.
import pandas as pd
import logging
from ml.model import train_model, compute_model_metrics, inference, perf_slices
from ml.data import process_data

logging.basicConfig(filename='logging.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')
# Add code to load in the data.
data = pd.read_csv(r'C:\Users\z0176083\OneDrive - ZF Friedrichshafen AG\Documents\Udacity\ML_DevOps\Section_4_deployment\nd0821-c3-starter-code-master\starter\data\cleaned_census.csv')

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
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

print(os.path.exists("../model/"))

# Train and save a model.
model_path = './model'
model = train_model(X_train,y_train)
pickle.dump(model, open("../model/trained_model.pkl","wb"))
pickle.dump(encoder, open("model/encoder.pkl","wb"))
pickle.dump(lb, open("model/labelizer.pkl","wb"))

logging.info(f"Model saved to: {model_path}")

#compute performance on categorical slices. save results to slice_output.txt
for feature in cat_features:
    perf_df = perf_slices(test, feature, y_test, inference(model, X_test))
    f = open('slice_output.txt', 'a')
    f.write(perf_df)
    f.close()
    logging.info(f"Performance on slice {feature}")
    logging.info(perf_df)