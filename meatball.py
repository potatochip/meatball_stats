'''
"So easy, an undergrad could do it!"
'''
from __future__ import division
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV

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
    return features, response, df


def estimator_database():
    columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Best_Parameters']
    df = pd.DataFrame(columns=[columns])
    return df


def choose_multi_variable_estimator():
    pass


def choose_single_estimator():
    pass


def choose_variables():
    pass


def linear():
    # using R-squared for accuracy here
    linear_reg = LinearRegression()
    parameters = {'normalize':(True, False)}
    clf = GridSearchCV(linear_reg, parameters, n_jobs=-1)
    clf.fit(features, response)
    return clf.score(features, response), clf, clf.get_params()


def k_nearest_neighbor():
    pass


def logistic():
    pass


def make_plot(X, y, model, model_name, response='diagnosis'):
    feature = X.columns
    # X_choice = X[feature[5]]
    # plt.scatter(X_choice, y, color='black')
    # plt.plot(X_choice, model.predict(X), color='blue', linewidth=3)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False)
    sns.regplot(X[feature[4]], y, data, ax=ax1)
    sns.boxplot(X[feature[4]], y, color="Blues_r", ax=ax2)
    sns.residplot(X[feature[4]], (model.predict(X) - y) ** 2, color="indianred", lowess=True, ax=ax3)
    if model_name is 'linear':
        sns.interactplot(X[feature[3]], X[feature[4]], y, ax=ax4, filled=True,
                 scatter_kws={"color": "dimgray"}, contour_kws={"alpha": .5})
    elif model_name is 'logistic':
        pal = sns.blend_palette(["#4169E1", "#DFAAEF", "#E16941"], as_cmap=True)
        levels = np.linspace(0, 1, 11)
        sns.interactplot(X[feature[3]], X[feature[4]], y, levels=levels, cmap=pal, logistic=True)
    else:
        pass
    ax1.set_title('Regression')
    ax2.set_title(feature[4]+' Value')
    ax3.set_title(feature[4]+' Residuals')
    ax4.set_title('Two-value Interaction')
    f.tight_layout()
    plt.savefig(model_name+'_'+feature[4], bbox_inches='tight')


def main():
    X_train, X_test, y_train, y_test = train_test_split(features, response)

    estimator_df = estimator_database()
    for estimator in estimators:
        if estimator is linear:
            rsquared, model, parameters = linear()
            evaluation_metrics = [rsquared, 0, 0, 0, 0, parameters]
            make_plot(X_test, y_test, model, model_name='linear', response='diagnosis')
        elif estimator is k_nearest_neighbor:
            pass
        elif estimator is logistic:
            pass
        else:
            raise ValueError('Unknown estimator: {0}'.format(estimator))
        estimator_df.loc['linear regression'] = evaluation_metrics
    print estimator_df


if __name__ == '__main__':
    try:
        sys.argv[1]
    except:
        estimators = [linear, k_nearest_neighbor, logistic]
        features, response, data = prep_dataset()
    else:
        estimators = sys.argv[1]
        features = sys.argv[2]
        response = sys.argv[3]
    main()
