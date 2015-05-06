'''
"So easy, an undergrad could do it!"
'''
from __future__ import division
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from collections import defaultdict

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.learning_curve import learning_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def get_sample_dataset(dataset='processed.cleveland.data'):
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
    # change diagnosis from 0-4 scale to just 0 or 1
    df.diagnosis = df.diagnosis.apply(lambda x: 0 if x == 0 else 1)
    df.dropna(inplace=True)

    features = df.drop('diagnosis', axis=1)
    response = df.diagnosis
    return features, response, df


def create_estimator_database():
    columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'BestParams']
    df = pd.DataFrame(columns=[columns])
    return df


def choose_multi_variable_estimator():
    pass


def choose_single_estimator():
    pass


def classifier_full_report():
    scores = ['accuracy', 'precision', 'recall', 'f1']
    for score in scores:
        print("Working on "+model_name)
        print("# Tuning hyper-parameters for %s\n" % score)
        clf = GridSearchCV(estimator, parameters, cv=10, scoring=score)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:\n")
        print(clf.best_params_)
        print("\nGrid scores on development set:\n")
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        print("\nDetailed classification report:\n")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.\n")
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print("\n")


def grid_squid(estimator, parameters, X_train, X_test, y_train, y_test, model_name, tuning='accuracy'):
    # runs all the parameters specified and returns the best model
    # tuning tunes the hyper parameters to get the best score for that specific scoring test
    if model_name == 'Linear Regression':
        print("Working on "+model_name)
        clf = GridSearchCV(estimator, parameters, cv=10)
        clf.fit(X_train, y_train)
        return clf
    else:
        score_tuning = defaultdict(int)
        score_list = []
        # scores = ['accuracy', 'precision', 'recall', 'f1']
        scores = [tuning]
        for score in scores:
            # need to have this return auc as well
            print("Working on "+model_name)
            print("# Tuning hyper-parameters for %s\n" % score)
            clf = GridSearchCV(estimator, parameters, cv=10, scoring=score)
            clf.fit(X_train, y_train)
            print("Best parameters set found on development set:\n")
            print(clf.best_params_)
            # print("\nGrid scores on development set:\n")
            # for params, mean_score, scores in clf.grid_scores_:
            #     print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
            print("\nDetailed classification report:\n")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.\n")
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print("\n")
            score_tuning[score] = clf.best_params_
            score_list.append(clf.best_score_)
        precision = precision_score(y_test, clf.predict(X_test))
        score_list.append(precision)
        recall = recall_score(y_test, clf.predict(X_test))
        score_list.append(recall)
        f1 = f1_score(y_test, clf.predict(X_test))
        score_list.append(f1)
        probabilities = clf.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        score_list.append(roc_auc)
        return score_list, score_tuning[tuning]


def linear_foward_selection(X_train, y_train):
    '''
    forward selection of optimize adjusted R-squared by adding features that help
    the most one at a time until the score goes down or you run out of features
    not implemeneted yet
    '''
    remaining = {X_train.columns}
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


def linear(X_train, X_test, y_train, y_test):
    '''
    linear regression. using R-squared for accuracy here
    '''
    linear_reg = LinearRegression()
    parameters = {'normalize':[True, False]}
    clf = grid_squid(linear_reg, parameters, X_train, X_test, y_train, y_test, 'Linear Regression')
    return clf.score(X_test, y_test), clf, clf.best_params_
    # implement foward selection for the number of variables

def knn(X_train, X_test, y_train, y_test):
    '''
    K-nearest neighbor.
    '''
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors':range(1,50), 'weights':['uniform', 'distance'], 'algorithm':['ball_tree', 'kd_tree', 'brute', 'auto'], 'leaf_size':[10,20,30,40,50,60], 'p':[1,2]}
    scores, best = grid_squid(knn, parameters, X_train, X_test, y_train, y_test, 'KNN')
    return scores, best


def logistic(X_train, X_test, y_train, y_test):
    '''
    Logistic regression.
    '''
    log_reg = LogisticRegression()
    parameters = {'penalty':['l1','l2'], 'solver':['liblinear','lbfgs','newton-cg']}
    scores, best = grid_squid(log_reg, parameters, X_train, X_test, y_train, y_test, 'Logistic Regression')
    return scores, best


def gaussian(X_train, X_test, y_train, y_test):
    '''
    Gaussian NB
    '''
    gaussian = GaussianNB()
    parameters = {}
    scores, best = grid_squid(gaussian, parameters, X_train, X_test, y_train, y_test, 'Gaussian NB')
    return scores, best


def sample_size_learning_curve(model):
    train_sizes, train_scores, test_scores = learning_curve(model, classifiers, response, cv=10)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='train')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='test')
    plt.legend()


def make_plot(X, y, model, test_data, model_name, response='diagnosis'):
    feature = X.columns
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False)
    sns.regplot(X[feature[4]], y, test_data, ax=ax1)
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

    # Multi-variable correlation significance level
    f, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.blend_palette(["#00008B", "#6A5ACD", "#F0F8FF",
                              "#FFE6F8", "#C71585", "#8B0000"], as_cmap=True)
    sns.corrplot(test_data, annot=False, diag_names=False, cmap=cmap)
    ax.grid(False)
    ax.set_title('Multi-variable correlation significance level')
    plt.savefig(model_name+'_multi-variable_correlation', bbox_inches='tight')

    # complete coefficient plot - believe this is only for linear regression
    sns.coefplot("diagnosis ~ "+' + '.join(features), test_data, intercept=True)
    plt.xticks(rotation='vertical')
    plt.savefig(model_name+'_coefficient_effects', bbox_inches='tight')


def main():
    X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(features, response, data)

    estimator_df = create_estimator_database()
    for estimator in estimators:
        if estimator is linear:
            rsquared, model, parameters = linear(X_train, X_test, y_train, y_test)
            evaluation_metrics = [rsquared, 0, 0, 0, 0, parameters]
            if plots: make_plot(X_test, y_test, model, data_test, model_name='linear', response='diagnosis')
        elif estimator is knn:
            scores, best = knn(X_train, X_test, y_train, y_test)
            accuracy, precision, recall, f1, auc = scores
            evaluation_metrics = [accuracy, precision, recall, f1, auc, best]
        elif estimator is logistic:
            scores, best = logistic(X_train, X_test, y_train, y_test)
            accuracy, precision, recall, f1, auc = scores
            evaluation_metrics = [accuracy, precision, recall, f1, auc, best]
        elif estimator is gaussian:
            scores, best = gaussian(X_train, X_test, y_train, y_test)
            accuracy, precision, recall, f1, auc = scores
            evaluation_metrics = [accuracy, precision, recall, f1, auc, best]
        else:
            raise ValueError('Unknown estimator: {0}'.format(estimator))
        estimator_df.loc[estimator] = evaluation_metrics
    print estimator_df


if __name__ == '__main__':
    try:
        sys.argv[1]
    except:
        estimators = [linear, knn, logistic, gaussian]
        features, response, data = get_sample_dataset()
        plots = False
    else:
        estimators = sys.argv[1]
        features = sys.argv[2]
        response = sys.argv[3]
        data = sys.arvg[4]
        plots = sys.argv[5]
    main()
