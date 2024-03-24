## ML Model Deployment with FastAPI and Heroku


# Project starter kit

Census project: Predict whether income exceeds $50K/yr based on census data. Also known as Adult dataset.

Data has been downloaded from the [Udacity nd0821-c3 project starter kit](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv)

The UC Irvine Machine Learning Repository is where you can find information on the [original dataset](https://archive.ics.uci.edu/dataset/20/census+income)

# Environment

- Create your conda environment: 

  $ conda create --name <your environment name> --file requirements.txt
  $ conda env create --file conda.yaml

- $ conda activate <your environment name>


# Usage

- Data cleaning:

  Data cleaning can be performed by using the Jupyter notebook ["Census_Clean_Data.ipynb"](Census_Clean_Data.ipynb).
  The notebook also provides a good overview of the data.

- Sanity check: 

  $ python -m check.sanitycheck

  Answer path question with "api_tests.py" as test file for a check of functionality to meet course specifications

- Train the model: 

  $ python -m ml.train_model

  After the model has been trained successfully, the following files will be saved:

  Metrics will be written to ["/artifacts/slice_output.txt"](./artifacts/slice_output.txt)
  Model will be saved to file ["/artifacts/model.joblib"](/artifacts/model.joblib)
  Encoder will be saved to ["/artifacts/encoder.joblib"](/artifacts/encoder.joblib)
  Label binarizer will be saved to ["/artifacts/lb.joblib"](/artifacts/lb.joblib)

  The output will be shown on screen and also be saved in ["/logs/census.log"](./logs/census.log)

- Census API tests: 

  Start the uvicorn server with:
  
  $ uvicorn main:app --reload
  
  The server is then accessible via: ["http://127.0.0.1:8000"](http://127.0.0.1:8000)

  Documents can be found here: ["http://127.0.0.1:8000/docs"](http://127.0.0.1:8000/docss)

  FastAPI tests can be performed by executing the command: 

  $ python -m api_tests

  and by using the Jupyter notebook ["Census_Tests_API.ipynb"](Census_Tests_API.ipynb).

# Model

Find model card for more detailed information: model_card.md

# GithubActions

If changes have been made, github actions is called.

    - with one exception: if a tag is pushed, github actions are also called
