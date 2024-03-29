from fastapi.testclient import TestClient

if __package__ is None or __package__ == "":
    from main import app
else:
    from ..main import app

client = TestClient(app)

def test_welcome_msg():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == "<html><body><p>CensusAPI</p><p>Documentation can be found <a href='/docs'>here</a>.</p></body></html>"


def test_lower_50k():
    test = {"age": 39,
              "workclass": "State-gov",
              "fnlgt": 77516,
              "education": "Bachelors",
              "education-num": 13,
              "marital-status": "Never-married",
              "occupation": "Adm-clerical",
              "relationship": "Not-in-family",
              "race": "White",
              "sex": "Male",
              "capital-gain": 2174,
              "capital-loss": 0,
              "hours-per-week": 40,
              "native-country": "United-States"}
    resp = client.post("/predict", json=test)
    assert resp.status_code == 200
    assert resp.json() == {"predicted_salary": "<=50K"}


def test_greater_50k():
    test = {"age": 31,
              "workclass": "Private",
              "fnlgt": 45781,
              "education": "Masters",
              "education-num": 14,
              "marital-status": "Never-married",
              "occupation": "Prof-specialty",
              "relationship": "Not-in-family",
              "race": "White",
              "sex": "Female",
              "capital-gain": 14084,
              "capital-loss": 0,
              "hours-per-week": 50,
              "native-country": "United-States"}

    resp = client.post("/predict", json=test)
    assert resp.status_code == 200
    assert resp.json() == {"predicted_salary": ">50K"}
