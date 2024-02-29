import requests
import json

data =  {  'age':37,
                'workclass':"Private", 
                'fnlgt':284582,
                'education':"Masters",
                'education_num':14,
                'marital_status':"Married-civ-spouse",
                'occupation':"Exec-managerial",
                'relationship':"Wife",
                'race':"White",
                'sex':"Female",
                'capital_gain':0,
                'capital_loss':0,
                'hours_per_week':40,
                'native_country':"United-States"
        }

r = requests.post("https://ml-devops-4.onrender.com/prediction", data=json.dumps(data))

print(r.status_code)
print(r.json())