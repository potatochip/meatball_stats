import pandas as pd
from prettytable import PrettyTable
import datetime


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


def max_scores():
    pass


def load_pickle(pickle):
    pass


def make_plots():
    # colormap
    pass


def make_prediction():
    # make prediction given feature(s) and model
    pass


# # pickle recovery
# filename = 'recovery'
# pickle = 'combined_df2015-05-06 23:18:19.950979.pickle'
# recover_pickle(pickle, filename)
