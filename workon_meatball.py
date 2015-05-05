
# coding: utf-8

# In[321]:

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[322]:

df = pd.DataFrame.from_csv('processed.cleveland.data', header=-1, index_col=None)


# In[323]:

df.columns = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol', 'blood_sugar', 'ecg', 'max_hr', 'exercise_induced_angina', 'st_depression', 'slope', 'num_major_vessels', 'thal', 'diagnosis']
df.index.names = ['patient']
df = df.convert_objects(convert_numeric=True)
df.dropna(inplace=True)


# In[324]:

df.head()


# In[326]:

features = df.drop('diagnosis', axis=1)
response = df.diagnosis
features = sm.add_constant(features)
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


# In[239]:

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


# In[285]:

(linear_reg.predict(X.cholesterol) - y) ** 2


# In[165]:

from sklearn.grid_search import GridSearchCV


# In[339]:

sns.interactplot(X[X.columns[4]], X[X.columns[5]], y.values, ax=ax4, cmap="coolwarm", filled=True, levels=25)


# In[337]:

y.values


# In[199]:

from sklearn.svm import SVC
linear_reg = LinearRegression()


# In[203]:

parameters = {'normalize':(True, False)}
clf = GridSearchCV(linear_reg, parameters, n_jobs=-1)
clf.fit(features,response)


# In[206]:

clf.score(features,response)


# In[178]:

svr = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}


# In[194]:

clf = GridSearchCV(svr, parameters, n_jobs=-1)


# In[195]:

clf.fit(features, response)


# In[214]:

clf.get_params()


# In[198]:

clf.score(features, response)


# In[299]:

len((linear_reg.predict(X)- y)**2)


# In[313]:



