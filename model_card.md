# Model Card

An introducing to the concept of model cards can be found in the [original paper "Model Cards for Model Reporting"](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

Michael Wolff created this model. The model is using [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) with parameters "max_iter=1000" and "random_state=42" in scikit-learn 1.4.0. 

## Intended Use

This model can be used to predict whether an income exceeds, or is below $50K/yr based on the provided census data.

## Training Data

The UC Irvine Machine Learning Repository is where you can find information on the [original dataset](https://archive.ics.uci.edu/dataset/20/census+income)

The trainings data was provided in the [Udacity nd0821-c3 project starter kit](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv)

You can gain information on the data from this Jupyter notebook: ["Census_Clean_Data.ipynb"](Census_Clean_Data.ipynb).

In the process of data cleaning, spaces from column names were removed and also hyphen were replaced with underscore.
The dataset contains 32561 rows and 15 features (9 categorical and 6 numerical features).

The dataset set is split into 80-20 for a train and test split. The target feature is "salary", in 2 categories, salaries over $50K/yr, and salaries below $50K/yr. 
The categorical features where encoded with a OneHotEncoder with parameters `parse=False` and `handle_unknown="ignore"`. The label is encoded with a LabelBinarizer with no parameters.

## Evaluation Data

The evaluation data comprised the same pre-processing steps and parameters as the training data, having the remaining 20% of the original data.

## Metrics

The overall metrics of the model ar ==Precision: 0.752, Recall: 0.567, Fbeta: 0.647==

## Ethical Considerations

The dataset provided appears to be fair and not biased. Neither is the model biased towards any particular group of people.

## Caveats and Recommendations

The dataset contains information about race and gender, which can, in cases, potentially discriminate against individuals in such brackets. Further evaluation of the bias/ethical information of the dataset is necessary.
