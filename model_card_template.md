# Model Card

Virag Weiler developed the model in December 2023

## Model Details

Random Forest Classifier with default parameters was used to predict peoples salary.

## Intended Use

The model should be used by anyone who would like to predict someone's salary based on their age, workclass, education etc

## Training Data

The census data was used for training from 1994, it can be found here: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data

20% of the original dataset was not used for training, only for evaluation

## Metrics
The classification model's performance was calculated with three measurements, the precision, recall and f1 score:
- precision: 1.000
- recall: 0.008 
- fbeta: 0.016

## Ethical Considerations
Dataset contains sensitive information so data leakage should not happen.

## Caveats and Recommendations
Its a 1994 dataset so the model should be retrained with current data if using nowadays. 