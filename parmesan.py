'''
meatball helper module
'''

import pandas as pd
from prettytable import PrettyTable
import datetime
from collections import defaultdict, Counter
import pickle
from prettytable import PrettyTable
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# print dataframe to screen with enough room
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


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


def data_processor(column_labels, response_label, data_file, header, index):
    if header and index:
        df = pd.DataFrame.from_csv(data_file, header=0, index_col=0)
    else:
        df = pd.DataFrame.from_csv(data_file, header=-1, index_col=None)
    df.columns = column_labels

    # # additional processing
    # df.family_history = pd.get_dummies(df.family_history).Present

    df = df.convert_objects(convert_numeric=True)
    df.dropna(inplace=True)

    features = df.drop(response_label, axis=1)
    response = df[response_label]
    return features, response, df


def recover_pickle(pickle, filename):
    dt = str(datetime.datetime.now())
    filename = filename + '-' + dt
    df = pd.read_pickle(pickle)
    df2 = df.copy()

    # save with multi-index .csv
    df.set_index(['Feature', 'Estimator'], inplace=True)
    df.to_csv(filename+'.csv')

    # save without multi-index as .txt and print to screen
    pt = PrettyTable()
    for i in df2.columns:
        pt.add_column(i, df2[i].tolist())
    print pt
    table_txt = pt.get_string()
    with open(filename+'.txt', 'w') as file:
        file.write(table_txt)


def deep_sea_squid(estimator, parameters, X_train, X_test, y_train, y_test,
                    model_name, tuning='accuracy'):
    # call squid grid again with narrower and more detailed parameters once model selected
    pass


def max_scores(dataframe, tuning=False):
    max_evaluators = defaultdict(int)
    evaluators = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    if not tuning:
        pass
    else:
        evaluators = [tuning]
    for evaluator in evaluators:
        # idx = dataframe.groupby(['Feature'])[evaluator].transform(max) == dataframe[evaluator]
        # max_evaluators[evaluator] = dataframe[idx][['Feature', 'Estimator', evaluator, evaluator+'_best']]
        max_evaluators = dataframe.sort(evaluator, ascending=False).groupby('Feature').first()[['Estimator', evaluator, evaluator+'_best']]
    return max_evaluators


def load_pickle(pickle):
    df = pd.read_pickle(pickle)
    return df


def make_plot(X_train, y_train, X, y, test_data, model, model_name, features, response):
    feature = X.columns
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False)
    sns.regplot(X[feature[4]], y, test_data, ax=ax1)
    sns.boxplot(X[feature[4]], y, color="Blues_r", ax=ax2)
    model.fit(X_train, y_train)
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


def make_model(model_data_file, feature_list, est):
    features = model_data_file[feature_list]
    model = est.fit(features, response)
    return model


def make_prediction(model_data_file, estimator_list, feature_list, response, feature_values=[]):
    # make prediction given feature(s) and model
    if not feature_values:
        for i in feature_list:
            x = float(raw_input("Value for {0}: ".format(i)))
            feature_values.append(x)
    # go through all the permutations since storing it to csv screwed up the order of all the combinations
    for i in itertools.permutations(feature_list):
        index = " : ".join(i)
        try:
            indexer = estimator_list.loc[index]
            break
        except:
            pass
    e = indexer.Estimator
    parameters = indexer.Accuracy_best
    accuracy = indexer.Accuracy
    if e == 'linear':
        est = LinearRegression(**parameters)
    elif e == 'knn':
        est = KNeighborsClassifier(**parameters)
    elif e == 'logistic':
        est = LogisticRegression(**parameters)
    elif e == 'gaussian':
        est = GaussianNB(**parameters)
    elif e == 'svc':
        est = SVC(**parameters)
    elif e == 'decision_tree':
        est = DecisionTreeClassifier(**parameters)
    elif e == 'random_forest':
        est = RandomForestClassifier(**parameters)
    model = make_model(model_data_file, feature_list, est)
    prediction = model.predict(feature_values)[0]
    print("We are {0}% sure that the your diagnosis for heart disease will be {1}".format(accuracy*100, prediction))


def rfe_trim(features, response, n_features_to_eliminate=4):
    # change to a percentage threshold to eliminate
    est = LogisticRegression()
    selector = RFE(est, n_features_to_select=n_features_to_eliminate)
    selector = selector.fit(features, response)
    # selected features are assigned rank 1
    # print(selector.support_)
    # print(selector.ranking_)
    trimmings = [i[0] for i in zip(features.columns, selector.support_) if i[1] == False]
    trimmed_features = features[trimmings]
    print(sorted(zip(selector.ranking_, features.columns)))
    return trimmed_features


def tree_trim(features, response, n_features_to_eliminate=4):
    # remove features above a certain threnshold
    clf = ExtraTreesClassifier()
    clf.fit(features, response)
    trimmings = [i[1] for i in sorted(zip(clf.feature_importances_, features.columns))][n_features_to_eliminate:]
    trimmed_features = features[trimmings]
    print([i for i in sorted(zip(clf.feature_importances_, features.columns), reverse=False)])
    return trimmed_features


def tree_sort(features, response):
    # sorted column names according to most important descending. the higher the number, the more important the feature.
    clf = ExtraTreesClassifier()
    clf.fit(features, response)
    print([i for i in sorted(zip(clf.feature_importances_, features.columns), reverse=True)])


def talking_to_trees(features, response):
    mylist = []
    limit = 4  # for the number of trees you want to drop
    clf = ExtraTreesClassifier()
    for i in range(1000):
        clf.fit(features, response)
        mylist.extend([i[1] for i in sorted(zip(clf.feature_importances_, features.columns),reverse=False)][:limit])
    print("Highest scores are the ones that appear most frequenly when asking the tree which {0} are least important.".format(limit))
    print(Counter(mylist))


def pretty_print_sorted(dataframe, column=False):
    # dataframe.reset_index(inplace=True)
    if column: dataframe.sort(column, inplace=True)
    pt = PrettyTable()
    for i in dataframe.columns:
        pt.add_column(i, max_accuracy[i].tolist())
    print pt


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


# # pickle recovery
# filename = 'recovery'
# pickle = 'combined_df2015-05-06 23:18:19.950979.pickle'
# recover_pickle(pickle, filename)

# # get max evaluators
# df = load_pickle('stanford_final.pickle')
# tuning='Accuracy'
# max_accuracy = max_scores(df, tuning)
# max_accuracy.to_csv('stanford_max_accuracy.csv')
# max_accuracy.to_pickle('stanford_max_accuracy.pickle')
# max_accuracy.reset_index(inplace=True)
# pretty_print_sorted(max_accuracy, 'Accuracy')

# # comparing trimming methods
# features, response, df = get_sample_dataset()
# rfe_features = rfe_trim(features, response)
# data = (rfe_features, response)
# save_pickle(data, 'rfe_features.csv')
# tree_features = tree_trim(features, response)
# data = (tree_features, response)
# save_pickle(data, 'tree_features.csv')

# rank important features
column_labels = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol',
                  'blood_sugar', 'ecg', 'max_hr', 'exercise_induced_angina',
                  'st_depression', 'slope', 'num_major_vessels', 'thal', 'diagnosis']
response_label = 'diagnosis'
data_file = 'all_processed_data.csv'
header = True
index = True
features, response, df = data_processor(column_labels, response_label, data_file, header, index)
talking_to_trees(features, response)

# # make prediction based on best estimator for cleveland data
# f, response, data_file = get_sample_dataset()
# best_estimators = load_pickle('cleveland_max_accuracy.pickle')
# interested_features = ['age', 'resting_bp', 'chest_pain']
# make_prediction(data_file, best_estimators, interested_features, response)

# # make plots
# features, response, df = get_sample_dataset()
# model = LogisticRegression()
# model_name = 'logistic'
# X_train, X_test, y_train, y_test, train_data, test_data = train_test_split(features, response, df)
# make_plot(X_train, y_train, X_test, y_test, test_data, model, model_name, features, 'diagnosis')
