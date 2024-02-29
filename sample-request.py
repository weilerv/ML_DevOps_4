import requests
import json

data =  {  'age':37,
                'workclass':"Private", 
                'fnlgt':284582,
                'education':"Masters",
                'education-num':14,
                'marital-status':"Married-civ-spouse",
                'occupation':"Exec-managerial",
                'relationship':"Wife",
                'race':"White",
                'sex':"Female",
                'capital-gain':0,
                'capital-loss':0,
                'hours-per-week':40,
                'native-country':"United-States"
        }

r = requests.post("https://ml-devops-4.onrender.com/prediction", data=json.dumps(data))

print(r.status_code)
print(r.json())