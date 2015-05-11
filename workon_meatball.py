
# coding: utf-8

# In[152]:

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[153]:

df = pd.DataFrame.from_csv('processed.cleveland.data', header=-1, index_col=None)


# In[154]:

df.columns = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol', 'blood_sugar', 'ecg', 'max_hr', 'exercise_induced_angina', 'st_depression', 'slope', 'num_major_vessels', 'thal', 'diagnosis']
df.index.names = ['patient']
df = df.convert_objects(convert_numeric=True)
df.diagnosis = df.diagnosis.apply(lambda x: 0 if x == 0 else 1)
df.dropna(inplace=True)


# In[155]:

df.head()


# In[156]:

features = df.drop('diagnosis', axis=1)
response = df.diagnosis
# features = sm.add_constant(features)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, response)


# In[157]:

from sklearn.linear_model import LinearRegression


# In[158]:

linear_reg = LinearRegression()
linear_reg.fit(features, response)


# In[ ]:

pd.DataFrame.from_csv(


# In[159]:

linear_reg.score(features, response)


# In[160]:

model = sm.OLS(response, features)
results = model.fit()
results.summary()


# In[161]:

plt.scatter(features.resting_bp, response)


# In[162]:

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 1, ax=ax)
ax.set_ylabel("Diagnosis")
ax.set_xlabel("Resting BP")
ax.set_title("Linear Regression")


# In[163]:

X_train, X_test, y_train, y_test = train_test_split(features, response)
X = X_test
y = y_test
plt.scatter(X.cholesterol, y, color='black')
plt.plot(X.cholesterol, linear_reg.predict(X), 'bo')
plt.xticks(())
plt.yticks(())
plt.show()


# In[ ]:




# In[164]:

sns.lmplot('cholesterol', 'diagnosis', df)


# In[165]:

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


# In[166]:

X.cholesterol


# In[167]:

from sklearn.grid_search import GridSearchCV


# In[168]:

sns.interactplot(X[X.columns[4]], X[X.columns[5]], y.values, ax=ax4, cmap="coolwarm", filled=True, levels=25)


# In[169]:

y.values


# In[170]:

from sklearn.svm import SVC
linear_reg = LinearRegression()
parameters = {'normalize':(True, False)}


# In[306]:

from sklearn.pipeline import Pipeline, FeatureUnion
svc=SVC()
lin_reg=LinearRegression()
log_reg=LogisticRegression()
dtc=DecisionTreeClassifier()
combined_features = FeatureUnion([("svc", svc), ("linear_reg", lin_reg), ("log_reg", log_reg)
                                 ]) 
pipeline = Pipeline([("features", combined_features), ("DecisionTreeClassifier", dtc)])
parameters = {}
clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
# clf.fit(features,response)
# clf.best_estimator_


# In[172]:

clf = LinearRegression()
temp=features['age'].values

# print(features)
print(clf.fit(features[["age"]],response).coef_)
print(temp.dtype)
print(response.dtype)


# In[ ]:




# In[173]:

df5=pd.DataFrame([[1,23,4,5]], columns=['a','b','c','d'])


# In[174]:

df5 = df5.append(pd.DataFrame([[1,2,3,4]], columns=['a','b','c','d']))


# In[175]:

df5[['a']]


# In[176]:

import prettytable


# In[177]:

df5[df5.columns.tolist()[1]].tolist()


# In[178]:

len(df5.columns)


# In[179]:

import itertools
feature_list = []
for num in range(2, len(df5.columns)):
    feature_list.extend([list(i) for i in itertools.combinations(df5, num)])
print(feature_list)
for i in thelist:
    print(df5[i])


# In[ ]:




# In[ ]:




# In[189]:

df5.set_index(['b','c']).index.names



# In[190]:

clf.get_params


# In[191]:

clf.score(features, response)


# In[192]:

len((linear_reg.predict(X)- y)**2)


# In[193]:

f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.blend_palette(["#00008B", "#6A5ACD", "#F0F8FF",
                          "#FFE6F8", "#C71585", "#8B0000"], as_cmap=True)
sns.corrplot(df, annot=False, diag_names=False, cmap=cmap)
ax.grid(False)
ax.set_title='Multi-variable correlation significance level'


# In[194]:

sns.coefplot("diagnosis ~ "+' + '.join(features), df, intercept=True, palette="Set1")


# In[195]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[196]:

log_reg = LogisticRegression()
log_reg


# In[197]:

import matplotlib
import mpld3
from mpld3 import plugins, utils


# In[198]:

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


# In[199]:

for i in features.columns:
    features[i]


# In[ ]:




# In[290]:

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
clf.C
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# In[293]:

clf.score('Accuracy')


# In[201]:

clf.best_estimator_


# In[202]:

clf.predict(X_test).best_score


# In[321]:


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


# In[220]:

classification_report(y_true, y_pred)


# In[ ]:




# In[249]:

estimator = LogisticRegression()
parameters = {'penalty':['l1','l2'], 'solver':['liblinear','lbfgs','newton-cg']}
model = GridSearchCV(estimator, parameters, scoring='accuracy')
model.fit(features, response)
print(model)
model.best_score_


# In[288]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, response)

from sklearn.metrics import accuracy_score
clf = LogisticRegression()
clf.fit(X_train, y_train)
accuracy_score(y_test, clf.predict(X_test))


# In[289]:

from sklearn.cross_validation import cross_val_score
X=features
y=df.index
print np.mean(cross_val_score(KNeighborsClassifier(), X, y, scoring='accuracy'))
print np.mean(cross_val_score(KNeighborsClassifier(), X, y=='democrat', scoring='precision'))
print np.mean(cross_val_score(KNeighborsClassifier(), X, y=='democrat', scoring='recall'))
print np.mean(cross_val_score(KNeighborsClassifier(), X, y=='democrat', scoring='f1'))


# In[362]:

from sklearn.feature_selection import RFE
est = LogisticRegression()
selector = RFE(est, n_features_to_select=1)
selector = selector.fit(features, response)
#selected features to eliminate are assigned rank 1
print(selector.support_)
print(selector.ranking_)
sorted(zip(selector.ranking_, features.columns))


# In[693]:

from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
mylist = []
X, y = features,response
# print(X.shape)
clf = ExtraTreesClassifier()
for i in range(1000):
    clf.fit(X, y)
#     print(X_new.shape)
#     print(clf.feature_importances_)
    # zip(sorted(zip(clf.feature_importances_, X.columns)), range(1,14))
    # (the higher, the more important the feature).
    mylist.extend([i[1] for i in sorted(zip(clf.feature_importances_, X.columns),reverse=False)][:5])
Counter(mylist)


# In[701]:

dropping = ['ecg', 'blood_sugar', 'sex', 'slope', 'resting_bp', 'cholesterol']
features.drop(dropping, axis=1)


# In[557]:

# sorted_features = ['thal', 'num_major_vessels', 'st_depression', 'chest_pain', 'slope', 'max_hr', 'resting_bp', 'age', 'cholesterol', 'exercise_induced_angina', 'sex', 'ecg', 'blood_sugar']
# important_feature_df = pd.DataFrame()
# new_feature_set = sorted_features
# for _ in sorted_features:
#     new_feature_set.pop()
#     print(features[new_feature_set])
# features[['age', 'sex']]


# In[702]:

def load_pickle(pickle):
    df = pd.read_pickle(pickle)
    return df
df9 = load_pickle('cleveland_final.pickle')


# In[98]:

# df9.set_index(['Feature', 'Estimator'], inplace=True)
# df9.set_index(['Feature'], inplace=True)


# In[709]:

# df9[(df9['Feature'] == 'multi')]['Accuracy'].max()
# df9.groupby('Feature')['Accuracy'].value_counts()
# df9.sort('Accuracy')
max_accuracy = df9.sort('Accuracy', ascending=False).groupby('Feature').first()[['Estimator','Accuracy','Accuracy_best']]
# gb = df9.groupby('Feature')
# gb.first()
max_accuracy.head()


# In[708]:

# df9[(df9['Feature'] == 'multi')]
# idx = df9.groupby(['Feature'])['Accuracy'].transform(max) == df9['Accuracy']
# df9[idx][['Feature', 'Estimator','Accuracy', 'Accuracy_best']].sort('Accuracy')


# In[725]:

max_accuracy.loc['age']


# In[734]:

x=max_accuracy.loc['age'].Accuracy_best
x


# In[759]:

model = SVC(**x)
model.fit(features[['age', 'sex', 'chest_pain', 'cholesterol']], response)
model.predict([67,1,4,200])


# In[761]:

df12 = pd.DataFrame.from_csv('stanford_heart_disease.csv')
df12.dtypes

