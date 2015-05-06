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
import datetime
from collections import defaultdict
from StringIO import StringIO
import prettytable

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

# print dataframe to screen with enough room
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


def get_sample_dataset(dataset='processed.cleveland.data'):
    '''
    uses sample dataset from cleveland heart disease study if no dataset passed
    '''
    df = pd.DataFrame.from_csv(dataset, header=-1, index_col=None)
    df.columns = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol',
                  'blood_sugar', 'ecg', 'max_hr', 'exercise_induced_angina',
                  'st_depression', 'slope', 'num_major_vessels', 'thal',
                  'diagnosis']
    df.index.names = ['patient']
    df = df.convert_objects(convert_numeric=True)
    # changing diagnosis from 0-4 scale to just 0 or 1
    df.diagnosis = df.diagnosis.apply(lambda x: 0 if x == 0 else 1)
    df.dropna(inplace=True)

    features = df.drop('diagnosis', axis=1)
    response = df.diagnosis
    return features, response, df

def save_dataframe(dataframe, filename):
    dt = str(datetime.datetime.now())
    filename = filename + dt

    dataframe.to_pickle(filename+".pickle")

    dataframe.set_index(['Feature', 'Estimator'], inplace=True)

    dataframe.to_csv(filename+'.tsv', sep='\t')

    # print dataframe to screen and text file in nice format
    output = StringIO()
    dataframe.to_csv(output)
    output.seek(0)
    pt = prettytable.from_csv(output)
    print pt
    table_txt = pt.get_string()
    with open(filename+'.txt','w') as file:
        file.write(table_txt)


def create_estimator_database(estimators, features, response, caller):
    X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(features, response, data)
    columns = ['Feature', 'Estimator', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Accuracy_best', 'Precision_best', 'Recall_best', 'F1_best', 'AUC_best']
    estimator_df = pd.DataFrame(columns=[columns])

    # set tuning below to an evaluation metric to just judge that. set it to False to have it evaluate all metrics
    tuning = 'accuracy'
    # tuning = False
    for estimator in estimators:
        if estimator == 'linear':
            scores_dict, model = linear(X_train, X_test, y_train, y_test)
            score, best = scores_dict['accuracy']
            evaluation_metrics = [caller, estimator, score, 0, 0, 0, 0, best, 0, 0, 0, 0]
        elif estimator == 'knn':
            scores_dict, model = knn(features, response, tuning)
        elif estimator == 'logistic':
            scores_dict, model = logistic(features, response, tuning)
        elif estimator == 'gaussian':
            scores_dict, model = gaussian(features, response, tuning)
        elif estimator == 'svc':
            scores_dict, model = support_vector(features, response, tuning)
        elif estimator == 'decision_tree':
            scores_dict, model = decision_tree(features, response, tuning)
        elif estimator == 'random_forest':
            scores_dict, model = random_forest(features, response, tuning)
        else:
            raise ValueError('Unknown estimator: {0}'.format(estimator))
        if estimator == 'linear':
            pass
        else:
            for value in scores_dict.values():
                accuracy, accuracy_best = value if 'accuracy' in scores_dict else (0, 0)
                precision, precision_best = value if 'precision' in scores_dict else (0, 0)
                recall, recall_best = value if 'recall' in scores_dict else (0, 0)
                f1, f1_best = value if 'f1' in scores_dict else (0, 0)
                auc, auc_best = value if 'roc_auc' in scores_dict else (0, 0)
            evaluation_metrics = [caller, estimator, accuracy, precision, recall, f1, auc, accuracy_best, precision_best, recall_best, f1_best, auc_best]
        estimator_df = estimator_df.append(pd.DataFrame([evaluation_metrics], columns=columns))
    return estimator_df


def multi_feature(features, response):
    # for testing. remove next line after finished
    # estimators = ['gaussian', 'logistic']
    estimator_df = create_estimator_database(estimators, features, response, 'multi')
    return estimator_df


def single_feature(features, response):
    # for testing. remove next line after finished
    # estimators = ['gaussian', 'linear']
    single_feature_df = pd.DataFrame()
    for i in features.columns:
        print("Working on: {0}".format(i))
        single_feature = features[[i]]
        estimator_df = create_estimator_database(estimators, single_feature, response, i)
        single_feature_df = single_feature_df.append(estimator_df)
    return single_feature_df


def hyper_parameter_full_report(estimator, parameters):
    # spits this output to a file instead of screen. the whole enchilada.
    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    with open('hyper_parameter_full_report.txt', 'wb') as f:
        for score in scores:
            f.write("Working on "+model_name)
            f.write("# Tuning hyper-parameters for %s\n" % score)
            clf = GridSearchCV(estimator, parameters, cv=10, scoring=score)
            clf.fit(X_train, y_train)
            f.write("Best parameters set found on development set:\n")
            f.write(clf.best_params_)
            f.write("\nGrid scores on development set:\n")
            for params, mean_score, scores in clf.grid_scores_:
                f.write("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
            f.write("\nDetailed classification report:\n")
            f.write("The model is trained on the full development set.")
            f.write("The scores are computed on the full evaluation set.\n")
            y_true, y_pred = y_test, clf.predict(X_test)
            f.write(classification_report(y_true, y_pred))
            f.write("\n")


def grid_squid(estimator, parameters, x_and_y, model_name, tuning=False):
    '''
    runs all the parameters specified and returns the best model
    '''
    if model_name == 'Linear Regression':
        X_train, y_train = x_and_y
        print("Working on "+model_name)
        clf = GridSearchCV(estimator, parameters, cv=10, n_jobs=-1)
        clf.fit(X_train, y_train)
        return clf
    else:
        features, response = x_and_y
        if not tuning:
            scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        else:
            scores = [tuning]
        score_dict = defaultdict(int)
        for score in scores:
            print("Working on "+model_name)
            print("# Tuning hyper-parameters for %s" % score)
            clf = GridSearchCV(estimator, parameters, cv=10, scoring=score, n_jobs=-1)
            clf.fit(features, response)
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print("\n")
            score_dict[score] = (clf.best_score_, clf.best_params_)
        return score_dict, clf


def deep_sea_squid(estimator, parameters, X_train, X_test, y_train, y_test, model_name, tuning='accuracy'):
    # call squid grid again with narrower and more detailed parameters once model selected
    pass


def linear_foward_selection(X_train, y_train):
    '''
    forward selection of optimize adjusted R-squared by adding features that help
    the most one at a time until the score goes down or you run out of features
    not implemeneted yet. presently not called from within module.
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
    tuning = 'accuracy'
    x_and_y = [X_train, y_train]
    linear_reg = LinearRegression()
    parameters = {'normalize':[True, False]}
    clf = grid_squid(linear_reg, parameters, x_and_y, 'Linear Regression', tuning)
    return {tuning: (clf.score(X_test, y_test), clf.best_params_)}, clf
    # implement foward selection for the number of variables

def knn(features, response, tuning):
    '''
    K-nearest neighbor.
    '''
    x_and_y = [features, response]
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors':[10, 20, 30, 40, 50, 60], 'weights':['uniform', 'distance'], 'algorithm':['ball_tree', 'kd_tree', 'brute', 'auto'], 'leaf_size':[10,20,30,40,50,60], 'p':[1,2]}
    scores_dict, model = grid_squid(knn, parameters, x_and_y, 'KNN', tuning)
    return scores_dict, model


def logistic(features, response, tuning):
    '''
    Logistic regression.
    '''
    x_and_y = [features, response]
    log_reg = LogisticRegression()
    parameters = {'penalty':['l1','l2'], 'solver':['liblinear','lbfgs','newton-cg']}
    scores_dict, model = grid_squid(log_reg, parameters, x_and_y, 'Logistic Regression', tuning)
    return scores_dict, model


def gaussian(features, response, tuning):
    '''
    Gaussian NB
    '''
    x_and_y = [features, response]
    gaussian = GaussianNB()
    parameters = {}
    scores_dict, model = grid_squid(gaussian, parameters, x_and_y, 'Gaussian NB', tuning)
    return scores_dict, model


def support_vector(features, response, tuning):
    '''
    Support vector classification
    '''
    x_and_y = [features, response]
    svc = SVC()
    parameters = {'probability':[True], 'C': [1, 10, 100, 1000], 'degree':[1,3,5,7], 'gamma': [0.0, 0.001, 0.0001], 'shrinking':[True, False]}
    scores_dict, model = grid_squid(svc, parameters, x_and_y, 'SVC', tuning)
    return scores_dict, model


def decision_tree(features, response, tuning):
    '''
    Decision Tree classifiers
    '''
    x_and_y = [features, response]
    dtc = DecisionTreeClassifier()
    parameters = {'max_features':['auto', 'sqrt', 'log2', None], 'max_depth':[1, 5, 10, 15, 20, 25, 30], 'max_leaf_nodes':[2, 5, 10, 15, 20, 25, 30]}
    scores_dict, model = grid_squid(dtc, parameters, x_and_y, 'DTC', tuning)
    return scores_dict, model


def random_forest(features, response, tuning):
    '''
    Random forest classifier
    '''
    x_and_y = [features, response]
    rfc = RandomForestClassifier()
    parameters = {'max_features':['auto', 'sqrt', 'log2', None], 'max_depth':[1, 5, 10, 15, 20, 25, 30], 'max_leaf_nodes':[2, 5, 10, 15, 20, 25, 30], 'bootstrap':[True, False]}
    scores_dict, model = grid_squid(rfc, parameters, x_and_y, 'RFC', tuning)
    return scores_dict, model


def sample_size_learning_curve(model):
    train_sizes, train_scores, test_scores = learning_curve(model, classifiers, response, cv=10)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='train')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='test')
    plt.legend()


def max_scores():
    pass
    

def make_plot(X, y, model, test_data, model_name, features, response='diagnosis'):
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
    multi_df = multi_feature(f, r)
    single_feature_df = single_feature(f, r)
    combined_df = multi_df.append(single_feature_df)
    save_dataframe(combined_df, 'combined_df')


    if plots:
        #model =
        #model_name =
        make_plot(X_test, y_test, model, data_test, 'linear', features, response='diagnosis')


if __name__ == '__main__':
    try:
        sys.argv[1]
    except:
        estimators = ['linear', 'knn', 'logistic', 'gaussian', 'svc', 'decision_tree', 'random_forest']
        f, r, data = get_sample_dataset()
        plots = False
    else:
        estimators = sys.argv[1]
        features = sys.argv[2]
        response = sys.argv[3]
        data = sys.arvg[4]
        plots = sys.argv[5]
    main()
