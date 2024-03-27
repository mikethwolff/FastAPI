import requests
import json
from fastapi.responses import HTMLResponse


response = requests.get("https://census-salaries-d3e2956470bf.herokuapp.com/")
# Check the response status code
if response.status_code == 200:
    print("GET request successful.")
    print("Response:", response.text)
else:
    print("GET request failed with status code:", response.status_code)

# Test data
d = {     
    'age': 31,
    'workclass': 'Private',
    'fnlgt': 45781,
    'education': 'Masters',
    'education_num': 14,
    'marital_status': 'Never-married',
    'occupation': 'Prof-specialty',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Female',
    'capital_gain': 14084,
    'capital_loss': 0,
    'hours_per_week': 50,
    'native_country': 'United-States'
    }

response = requests.post("https://census-salaries-d3e2956470bf.herokuapp.com/predict", json=d)
if response.status_code == 200:
    print("POST request successful.")
    print(response.text)
else:
    print("POST request failed with status code:", response.status_code)