import pandas as pd

df = pd.read_csv(r'C:\Users\z0176083\OneDrive - ZF Friedrichshafen AG\Documents\Udacity\ML_DevOps\Section_4_deployment\nd0821-c3-starter-code-master\starter\data\census.csv')
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "salary"
]

for col in df.columns:
    if col != 'age':
        print(col)
        df.rename(columns={col: col[1:]}, inplace=True)
        col = col[1:]
        if col in cat_features:
            df[col] = df[col].map(str.strip)

df.to_csv("./data/cleaned_census.csv", index=False)
