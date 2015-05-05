
# coding: utf-8

# In[130]:

import pandas as pd
import numpy as np
import statsmodels.api as sm

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[110]:

df = pd.DataFrame.from_csv('processed.cleveland.data', header=-1, index_col=None)


# In[111]:

df.columns = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol', 'blood_sugar', 'ecg', 'max_hr', 'exercise_induced_angina', 'st_depression', 'slope', 'num_major_vessels', 'thal', 'diagnosis']
df.index.names = ['patient']
df = df.convert_objects(convert_numeric=True)
df.dropna(inplace=True)


# In[112]:

df.head()


# In[133]:

features = df.drop('diagnosis', axis=1)
response = df.diagnosis
features = sm.add_constant(features)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, response)


# In[114]:

from sklearn.linear_model import LinearRegression


# In[148]:

linear_reg = LinearRegression()
linear_reg.fit(features, response)


# In[ ]:




# In[126]:

linear_reg.score(features, response)


# In[134]:

model = sm.OLS(response, features)
results = model.fit()
results.summary()


# In[136]:

plt.scatter(features.resting_bp, response)


# In[141]:

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 1, ax=ax)
ax.set_ylabel("Diagnosis")
ax.set_xlabel("Resting BP")
ax.set_title("Linear Regression")


# In[164]:

X_train, X_test, y_train, y_test = train_test_split(features, response)
X = X_test['cholesterol']
y = y_test
plt.scatter(X, y, color='black')
plt.plot(X, linear_reg.predict(X), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


# In[160]:

model.predict()


# In[161]:

model.predict(X).shape


# In[ ]:



