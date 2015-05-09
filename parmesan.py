import pandas as pd
from prettytable import PrettyTable
import datetime
from collections import defaultdict, Counter
import pickle
from pprint import pprint

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


# print dataframe to screen with enough room
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
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
        idx = dataframe.groupby(['Feature'])[evaluator].transform(max) == dataframe[evaluator]
        max_evaluators[evaluator] = dataframe[idx][['Feature', 'Estimator', evaluator, evaluator+'_best']]
    return max_evaluators


def load_pickle(pickle):
    df = pd.read_pickle(pickle)
    return df


def make_plots():
    # colormap
    # multi plot for roc curve
    pass


def make_prediction():
    # make prediction given feature(s) and model
    pass


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


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


# # pickle recovery
# filename = 'recovery'
# pickle = 'combined_df2015-05-06 23:18:19.950979.pickle'
# recover_pickle(pickle, filename)


# df = load_pickle('all_estimators.pickle')
# tuning='Accuracy'
# max_evaluators = max_scores(df, tuning)[tuning]
# print max_evaluators

# # comparing trimming methods
features, response, df = get_sample_dataset()
# rfe_features = rfe_trim(features, response)
# data = (rfe_features, response)
# save_pickle(data, 'rfe_features.csv')
# tree_features = tree_trim(features, response)
# data = (tree_features, response)
# save_pickle(data, 'tree_features.csv')

talking_to_trees(features, response)
