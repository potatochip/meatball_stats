'''
"So easy, an undergrad could do it!"
'''
from __future__ import division
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns


def prep_sample():
    df = pd.DataFrame.from_csv('processed.cleveland.data', header=-1,
                               index_col=None)
    features = df.drop['diagnosis', axis=1]
    response = 'heart_disease'
    return df, features, response

def choose_estimator():
    pass


def choose_variables():
    pass


def linear_regression():
    pass


def k_nearest_neighbor():
    pass


def logistic_regression():
    pass


def main():
    pass

if __name__ == '__main__':
    try:
        variables = sys.argv
    except:
        estimators = [linear_regression, k_nearest_neighbor,
                      logistic_regression]
        dataset, features, response = prep_sample()
    main(estimators)
