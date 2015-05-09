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
from prettytable import PrettyTable
import itertools
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.learning_curve import learning_curve
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# print dataframe to screen with enough room
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


def grab_pickle(filename):
    with open(filename, "rb ") as f:
        return pickle.load(f)


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
    filename = filename + '-' + dt

    dataframe.to_pickle(filename+".pickle")

    # prints dataframe to screen and text file in nice format
    pt = PrettyTable()
    for i in dataframe.columns:
        pt.add_column(i, dataframe[i].tolist())
    print pt
    table_txt = pt.get_string()
    with open(filename+'.txt', 'w') as file:
        file.write(table_txt)

    # make multi-index
    dataframe.set_index(['Feature', 'Estimator'], inplace=True)

    # save with multi-index
    dataframe.to_csv(filename+'.tsv', sep='\t')
    dataframe.to_csv(filename+'.csv')


def create_estimator_database(features, response, caller, tuning):
    features = StandardScaler().fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features, response, test_size=0.3)
    columns = ['Feature', 'Estimator', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC',
                'Accuracy_best', 'Precision_best', 'Recall_best', 'F1_best', 'AUC_best']
    estimator_df = pd.DataFrame(columns=[columns])

    # # for testing. remove next line after finished
    # estimators = ['linear', 'gaussian', 'logistic']

    train = [X_train, y_train]
    test = [X_test, y_test]
    for estimator in estimators:
        if estimator == 'linear':
            scores_dict, model = linear(features, response, train, test)
            score, best = scores_dict['accuracy']
            evaluation_metrics = [caller, estimator, score, 0, 0, 0, 0, best, 0, 0, 0, 0]
        elif estimator == 'knn':
            scores_dict, model = knn(features, response, train, test, tuning)
        elif estimator == 'logistic':
            scores_dict, model = logistic(features, response, train, test, tuning)
        elif estimator == 'gaussian':
            scores_dict, model = gaussian(features, response, train, test, tuning)
        elif estimator == 'svc':
            scores_dict, model = support_vector(features, response, train, test, tuning)
        elif estimator == 'decision_tree':
            scores_dict, model = decision_tree(features, response, train, test, tuning)
        elif estimator == 'random_forest':
            scores_dict, model = random_forest(features, response, train, test, tuning)
        else:
            raise ValueError('Unknown estimator: {0}'.format(estimator))
        if estimator == 'linear':
            pass
        else:
            accuracy, accuracy_best = scores_dict.get('accuracy', (0, 0))
            precision, precision_best = scores_dict.get('precision', (0, 0))
            recall, recall_best = scores_dict.get('recall', (0, 0))
            f1, f1_best = scores_dict.get('f1', (0, 0))
            auc, auc_best = scores_dict.get('roc_auc', (0, 0))
            evaluation_metrics = [caller, estimator, accuracy, precision, recall, f1,
                                    auc, accuracy_best, precision_best, recall_best,
                                    f1_best, auc_best]
        estimator_df = estimator_df.append(pd.DataFrame([evaluation_metrics], columns=columns))
    return estimator_df


def all_features(features, response, tuning):
    estimator_df = create_estimator_database(features, response, 'all', tuning)
    return estimator_df


def single_feature(features, response, tuning):
    single_feature_df = pd.DataFrame()
    for i in features.columns:
        print("Working on: {0}".format(i))
        single_feature = features[[i]]
        estimator_df = create_estimator_database(single_feature, response, i, tuning)
        single_feature_df = single_feature_df.append(estimator_df)
    return single_feature_df


def double_feature(features, response, tuning):
    feature_list = []
    count = 0
    for num in range(2, 3):
        feature_list.extend([list(i) for i in itertools.combinations(features, num)])

    double_feature_df = pd.DataFrame()
    for i in feature_list:
        print("Working on: {0}".format(i))
        multi_f = features[i]
        caller = " : ".join(i)
        estimator_df = create_estimator_database(multi_f, response, caller, tuning)
        double_feature_df = double_feature_df.append(estimator_df)
        count += 1
        remaining = count / len(feature_list) * 100.0
        sys.stdout.write('\r' + str("Phase I: %f" % remaining) + "%\n")
        sys.stdout.flush()
    return double_feature_df


def multi_feature(features, response, tuning):
    # uses a trimmed feature set.
    # this is going to take forever. won't work for anything more than ten features.
    # combination_size is the upper limit so will not make combinations at that size but will make all the combinations below it
    feature_list = []
    count = 0
    for num in range(3, len(features.columns)):
    # for num in range(2, combination_size):
        feature_list.extend([list(i) for i in itertools.combinations(features, num)])

    multi_feature_df = pd.DataFrame()
    for i in feature_list:
        print("Working on: {0}".format(i))
        multi_f = features[i]
        caller = " : ".join(i)
        estimator_df = create_estimator_database(multi_f, response, caller, tuning)
        multi_feature_df = multi_feature_df.append(estimator_df)
        count += 1
        remaining = count / len(feature_list) * 100.0
        sys.stdout.write('\r' + str("Phase II: %f" % remaining) + "%\n")
        sys.stdout.flush()

        if count == 200:
            dt = str(datetime.datetime.now())
            estimator_df.to_pickle(dt+'.pkl')
    return multi_feature_df


def hyper_parameter_full_report(estimator, parameters, train, test, model_name):
    # spits this output to a file instead of screen. the whole enchilada.
    X_train, y_train = train
    X_test, y_test = test
    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    with open('hyper_parameter_full_report.txt', 'wb') as f:
        for score in scores:
            f.write("Working on "+model_name)
            f.write("# Tuning hyper-parameters for %s\n" % score)
            clf = GridSearchCV(estimator, parameters, cv=5, scoring=score)
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


def grid_squid(estimator, parameters, train, test, features, response, model_name, tuning=False):
    '''
    runs all the parameters specified and returns the best model
    '''
    X_train, y_train = train
    X_test, y_test = test
    X = features
    y = response
    if model_name == 'Linear Regression':
        print("Working on "+model_name)
        print("# Tuning hyper-parameters for %s" % tuning)
        clf = GridSearchCV(estimator, parameters, cv=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("\n")
        test_score = np.mean(cross_val_score(estimator, X, y, cv=10))
        return test_score, clf.best_params_, clf
    else:
        if not tuning:
            scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        else:
            scores = [tuning]
        score_dict = defaultdict(int)
        for score in scores:
            print("Working on "+model_name)
            print("# Tuning hyper-parameters for %s" % score)
            clf = GridSearchCV(estimator, parameters, cv=3, scoring=score, n_jobs=-1)
            clf.fit(X_train, y_train)
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print("\n")
            if score == 'accuracy':
                # test_score = accuracy_score(y_test, clf.predict(X_test))
                test_score = np.mean(cross_val_score(estimator, X, y, scoring=score, cv=10))
            elif score == 'precision':
                # test_score = precision_score(y_test, clf.predict(X_test))
                test_score = np.mean(cross_val_score(estimator, X, y, scoring=score, cv=10))
            elif score == 'recall':
                # test_score = recall_score(y_test, clf.predict(X_test))
                test_score = np.mean(cross_val_score(estimator, X, y, scoring=score, cv=10))
            elif score == 'f1':
                # test_score = f1_score(y_test, clf.predict(X_test))
                test_score = np.mean(cross_val_score(estimator, X, y, scoring=score, cv=10))
            elif score == 'roc_auc':
                # test_score = roc_auc_score(y_test, clf.predict(X_test))
                test_score = np.mean(cross_val_score(estimator, X, y, scoring=score, cv=10))
            score_dict[score] = (test_score, clf.best_params_)
        return score_dict, clf


def linear_foward_selection(X_train, y_train):
    '''
    forward selection of optimize adjusted R-squared by adding features that help
    the most one at a time until the score goes down or you run out of features
    not implemeneted yet. would only make sense for a linear model. not for categorical
    data presently not called from within module.
    '''
    remaining = {X_train.columns}
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
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


def linear(features, response, train, test,):
    '''
    linear regression. using R-squared for accuracy here
    '''
    tuning = 'accuracy'
    linear_reg = LinearRegression()
    parameters = {'normalize':[True, False]}
    score, best_param, model = grid_squid(linear_reg, parameters, train, test, features, response, 'Linear Regression', tuning)
    return {tuning: (score, best_param)}, model
    # implement foward selection for the number of variables


def knn(features, response, train, test, tuning):
    '''
    K-nearest neighbor.
    '''
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors':[10, 20, 30, 40, 50, 60], 'weights':['uniform', 'distance'], 'algorithm':['ball_tree', 'kd_tree', 'brute', 'auto'], 'leaf_size':[10,20,30,40,50,60], 'p':[1,2]}
    scores_dict, model = grid_squid(knn, parameters, train, test, features, response, 'KNN', tuning)
    return scores_dict, model


def logistic(features, response, train, test, tuning):
    '''
    Logistic regression.
    '''
    log_reg = LogisticRegression()
    parameters = {'penalty':['l1','l2'], 'solver':['liblinear','lbfgs','newton-cg']}
    scores_dict, model = grid_squid(log_reg, parameters, train, test, features, response, 'Logistic Regression', tuning)
    return scores_dict, model


def gaussian(features, response, train, test, tuning):
    '''
    Gaussian NB
    '''
    gaussian = GaussianNB()
    parameters = {}
    scores_dict, model = grid_squid(gaussian, parameters, train, test, features, response, 'Gaussian NB', tuning)
    return scores_dict, model


def support_vector(features, response, train, test, tuning):
    '''
    Support vector classification
    '''
    svc = SVC()
    parameters = {'probability':[True], 'C': [1, 10, 100, 1000], 'degree':[1,3,5,7], 'gamma': [0.0, 0.001, 0.0001], 'shrinking':[True, False]}
    scores_dict, model = grid_squid(svc, parameters, train, test, features, response, 'SVC', tuning)
    return scores_dict, model


def decision_tree(features, response, train, test, tuning):
    '''
    Decision Tree classifiers
    '''
    dtc = DecisionTreeClassifier()
    parameters = {'max_features':['auto', 'sqrt', 'log2', None], 'max_depth':[1, 5, 10, 15, 20, 25, 30], 'max_leaf_nodes':[2, 5, 10, 15, 20, 25, 30]}
    scores_dict, model = grid_squid(dtc, parameters, train, test, features, response, 'DTC', tuning)
    return scores_dict, model


def random_forest(features, response, train, test, tuning):
    '''
    Random forest classifier
    '''
    rfc = RandomForestClassifier()
    parameters = {'max_features':['auto', 'sqrt', 'log2', None], 'max_depth':[1, 5, 10, 15, 20, 25, 30], 'max_leaf_nodes':[2, 5, 10, 15, 20, 25, 30], 'bootstrap':[True, False]}
    scores_dict, model = grid_squid(rfc, parameters, train, test, features, response, 'RFC', tuning)
    return scores_dict, model


def sample_size_learning_curve(model, features, response):
    train_sizes, train_scores, test_scores = learning_curve(model, classifiers, response, cv=5)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='train')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='test')
    plt.legend()


def make_plot(X, y, model, test_data, model_name, features, response='diagnosis'):
    feature = X.columns
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False)
    sns.regplot(X[feature[4]], y, test_data, ax=ax1)
    sns.boxplot(X[feature[4]], y, color="Blues_r", ax=ax2)
    sns.residplot(X[feature[4]], (model.predict(X) - y) ** 2, color="indianred", lowess=True, ax=ax3)
    if model_name is 'linear':
        sns.interactplot(X[feature[3]], X[feature[4]], y, ax=ax4, filled=True, scatter_kws={"color": "dimgray"}, contour_kws={"alpha": .5})
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
    # ###### set tuning below to an evaluation metric to just judge that.
    # set it to False to have it evaluate all metrics
    tuning = 'accuracy'
    # tuning = False

    # evaluates all together, all alone, and all pairs for all features.
    # then evaluates for all combinations for a trimmed set of features (see parmesan).
    all_features_df = all_features(f, r, tuning)
    single_feature_df = single_feature(f, r, tuning)
    double_feature_df = double_feature(f, r, tuning)
    # trimmed feature set for multi
    # get new trimmed feature set
    dropping = ['ecg', 'blood_sugar', 'sex', 'slope']
    trimmed_features = f.drop(dropping, axis=1)
    multi_feature_df = multi_feature(trimmed_features, r, tuning)

    combined_df = all_features_df.append(multi_feature_df)
    combined_df = combined_df.append(double_feature_df)
    combined_df = combined_df.append(single_feature_df)
    save_dataframe(combined_df, 'combined_df')

    # if plots:
    #     model = 0
    #     model_name = 'linear'
    #     response = 'diagnosis'
    #     make_plot(X_test, y_test, model, test_data, model_name, features, response)


if __name__ == '__main__':
    try:
        sys.argv[1]
    except:
        estimators = ['linear', 'knn', 'logistic', 'gaussian', 'svc', 'decision_tree', 'random_forest']
        f, r, data = get_sample_dataset()   
        plots = True
    else:
        estimators = sys.argv[1]
        feature_labels = sys.argv[2]
        response_label = sys.argv[3]
        data_file = sys.arvg[4]
        plots = sys.argv[5]
        # f, r = data_processor(feature_labels, resonse_label, data_file)
    main()
