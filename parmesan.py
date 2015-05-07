import pandas as pd
from prettytable import PrettyTable
import datetime
from collections import defaultdict


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


def max_scores(dataframe):
    max_evaluators = defaultdict(int)
    evaluators = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
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


def trim_features():
    pass


# # pickle recovery
# filename = 'recovery'
# pickle = 'combined_df2015-05-06 23:18:19.950979.pickle'
# recover_pickle(pickle, filename)


# df = load_pickle('all_evaluators.csv')
# max_evaluators = max_scores(df)
