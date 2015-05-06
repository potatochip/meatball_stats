
# coding: utf-8

# In[529]:

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[530]:

df = pd.DataFrame.from_csv('processed.cleveland.data', header=-1, index_col=None)


# In[531]:

df.columns = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol', 'blood_sugar', 'ecg', 'max_hr', 'exercise_induced_angina', 'st_depression', 'slope', 'num_major_vessels', 'thal', 'diagnosis']
df.index.names = ['patient']
df = df.convert_objects(convert_numeric=True)
df.diagnosis = df.diagnosis.apply(lambda x: 0 if x == 0 else 1)
df.dropna(inplace=True)


# In[532]:

df.head()


# In[494]:

features = df.drop('diagnosis', axis=1)
response = df.diagnosis
# features = sm.add_constant(features)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, response)


# In[217]:

from sklearn.linear_model import LinearRegression


# In[260]:

linear_reg = LinearRegression()
linear_reg.fit(features, response)


# In[ ]:




# In[219]:

linear_reg.score(features, response)


# In[220]:

model = sm.OLS(response, features)
results = model.fit()
results.summary()


# In[221]:

plt.scatter(features.resting_bp, response)


# In[222]:

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 1, ax=ax)
ax.set_ylabel("Diagnosis")
ax.set_xlabel("Resting BP")
ax.set_title("Linear Regression")


# In[495]:

X_train, X_test, y_train, y_test = train_test_split(features, response)
X = X_test
y = y_test
plt.scatter(X.cholesterol, y, color='black')
plt.plot(X.cholesterol, linear_reg.predict(X), 'bo')
plt.xticks(())
plt.yticks(())
plt.show()


# In[241]:




# In[242]:

sns.lmplot('cholesterol', 'diagnosis', df)


# In[298]:

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
sns.regplot('cholesterol', 'diagnosis', df, ax=ax1)
sns.boxplot(df["cholesterol"], df["diagnosis"], color="Blues_r", ax=ax2).set_ylabel("")
sns.residplot(df["cholesterol"], (linear_reg.predict(features) - response) ** 2, color="indianred", order=1, lowess=True)
sns.residplot(df["cholesterol"], (linear_reg.predict(features) - response) ** 2, color="indianred", order=2, lowess=True)
ax1.set_title('Regression')
ax2.set_title('ax2 title')
f.tight_layout()

sns.residplot(X.cholesterol, (linear_reg.predict(X) - y) ** 2, color="indianred", lowess=True, ax=ax3)
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
# ax1.plot(x, y)
# ax1.set_title('Sharing x per column, y per row')
# ax2.scatter(x, y)
# ax3.scatter(x, 2 * y ** 2 - 1, color='r')
# ax4.plot(x, 2 * y ** 2 - 1, color='r')

# f, axarr = plt.subplots(2, 2)
# axarr[0, 0].plot(x, y)
# axarr[0, 0].set_title('Axis [0,0]')
# axarr[0, 1].scatter(x, y)
# axarr[0, 1].set_title('Axis [0,1]')
# axarr[1, 0].plot(x, y ** 2)
# axarr[1, 0].set_title('Axis [1,0]')
# axarr[1, 1].scatter(x, y ** 2)
# axarr[1, 1].set_title('Axis [1,1]')


# In[282]:

X.cholesterol


# In[165]:

from sklearn.grid_search import GridSearchCV


# In[339]:

sns.interactplot(X[X.columns[4]], X[X.columns[5]], y.values, ax=ax4, cmap="coolwarm", filled=True, levels=25)


# In[337]:

y.values


# In[504]:

from sklearn.svm import SVC
linear_reg = LinearRegression()
parameters = {'normalize':(True, False)}


# In[505]:

parameters = {'normalize':(True, False)}
clf = GridSearchCV(linear_reg, parameters, n_jobs=-1)
clf.fit(features,response)


# In[510]:

clf.score(features,response)


# In[511]:

linear_reg


# In[509]:

clf.get_params


# In[198]:

clf.score(features, response)


# In[299]:

len((linear_reg.predict(X)- y)**2)


# In[358]:

f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.blend_palette(["#00008B", "#6A5ACD", "#F0F8FF",
                          "#FFE6F8", "#C71585", "#8B0000"], as_cmap=True)
sns.corrplot(df, annot=False, diag_names=False, cmap=cmap)
ax.grid(False)
ax.set_title='Multi-variable correlation significance level'


# In[401]:

sns.coefplot("diagnosis ~ "+' + '.join(features), df, intercept=True, palette="Set1")


# In[442]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:

log_reg = LogisticRegression()
log_reg


# In[404]:

import matplotlib
import mpld3
from mpld3 import plugins, utils


# In[469]:

# X = features
# y = response

# # dither the data for clearer plotting
# X += 0.1 * np.random.random(X.shape)

# fig, ax = plt.subplots(4, 4, sharex="col", sharey="row", figsize=(8, 8))
# fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
#                     hspace=0.1, wspace=0.1)

# for i in range(4):
#     for j in range(4):
#         points = ax[3 - i, j].scatter(X[[j]], X[[i]], s=40, alpha=0.6)

# # remove tick labels
# for axi in ax.flat:
#     for axis in [axi.xaxis, axi.yaxis]:
#         axis.set_major_formatter(plt.NullFormatter())

# # Here we connect the linked brush plugin
# plugins.connect(fig, plugins.LinkedBrush(points))

# mpld3.display()


# In[ ]:

rfc = RandomForestClassifier()


# In[512]:

# n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None
parameters = {'n_neighbors':range(1,50)}
knn = KNeighborsClassifier()


# ValueError: 'Accuracy_weighted' is not a valid scoring value. Valid options are
# ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 
#  'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 
#  'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 
#  'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 
#  'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    clf = GridSearchCV(knn, parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# In[538]:

clf.best_estimator_


# In[540]:

clf.predict(X_test).best_score


# In[476]:

from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

