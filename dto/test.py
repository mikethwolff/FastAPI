from pydantic import BaseModel, Field


class Test(BaseModel):
    age: int
    workclass: str
    #fnlgt: int
    education: str
    #education_num: int = Field(..., alias='education_num')
    marital_status: str = Field(..., alias='marital_status')
    occupation: str
    relationship: str
    race: str
    sex: str
    #capital_gain: int = Field(..., alias='capital_gain')
    #capital_loss: int = Field(..., alias='capital_loss')
    hours_per_week: int = Field(..., alias='hours_per_week')
    native_country: str = Field(..., alias='native_country')

    class Config:
        schema_extra = {
            "example": {
                'age': 39,
                'workclass': 'State-gov',
                #'fnlgt': 77516,
                'education': 'Bachelors',
                #'education_num': 13,
                'marital_status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                #'capital_gain': 2174,
                #'capital_loss': 0,
                'hours_per_week': 40,
                'native_country': 'United-States'
            }
        }
