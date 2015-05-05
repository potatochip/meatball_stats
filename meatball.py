'''
"So easy, an undergrad could do it!"
'''
from __future__ import division
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression


def prep_dataset(dataset='processed.cleveland.data'):
    '''
    uses sample dataset from clevaland heart disease study if no dataset passed
    '''
    df = pd.DataFrame.from_csv(dataset, header=-1, index_col=None)
    df.columns = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol',
                  'blood_sugar', 'ecg', 'max_hr', 'exercise_induced_angina',
                  'st_depression', 'slope', 'num_major_vessels', 'thal',
                  'diagnosis']
    df.index.names = ['patient']
    df = df.convert_objects(convert_numeric=True)
    df.dropna(inplace=True)

    features = df.drop('diagnosis', axis=1)
    response = df.diagnosis
    return features, response


def estimator_database():
    columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    df = pd.DataFrame(columns=[columns])
    return df


def choose_multi_variable_estimator():
    pass


def choose_single_estimator():
    pass


def choose_variables():
    pass


def linear_regression():
    # using R-squared for accuracy here
    linear_reg = LinearRegression()
    linear_reg.fit(features, response)
    return linear_reg.score(features, response), linear_reg


def k_nearest_neighbor():
    pass


def logistic_regression():
    pass


def make_plot(X, y, model, X_classifier='cholesterol'):
    X = X[X_classifier]
    plt.scatter(X, y, color='black')
    plt.plot(X, model.predict(X), color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def main():
    X_train, X_test, y_train, y_test = train_test_split(features, response)

    estimator_df = estimator_database()
    for estimator in estimators:
        if estimator is linear_regression:
            rsquared, model = linear_regression()
            evaluation_metrics = [rsquared, 0, 0, 0, 0]
            make_plot(X_test, y_test, model)
        elif estimator is k_nearest_neighbor:
            pass
        elif estimator is logistic_regression:
            pass
        else:
            raise ValueError('Unknown estimator: {0}'.format(estimator))
        estimator_df.loc['linear regression'] = evaluation_metrics


if __name__ == '__main__':
    try:
        sys.argv[1]
    except:
        estimators = [linear_regression, k_nearest_neighbor,
                  logistic_regression]
        features, response = prep_dataset()
    else:
        estimators = sys.argv[1]
        features = sys.argv[2]
        response = sys.argv[3]
    main()
