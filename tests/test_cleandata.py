import pytest
import pandas as pd
from ml.clean_data import basic_cleaning

@pytest.fixture(scope="session")
def loadData():
    data = pd.read_csv("tests/testdata/census_test.csv")
    return data

def testLoadData(loadData):
    assert loadData.shape == (100, 15)

def testCleaning(loadData):
    df, cat_cols, num_cols = basic_cleaning(loadData, "tests/testdata/census_cleaned.csv", "salary", test=True)
    assert df.shape == (100, 15)
    assert df.isna().sum().sum() == 0
    assert len(cat_cols) == 9
    assert len(num_cols) == 6
