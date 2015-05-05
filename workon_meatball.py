
# coding: utf-8

# In[40]:

import pandas as pd
import numpy as np


# In[46]:

df = pd.DataFrame.from_csv('processed.cleveland.data', header=-1, index_col=None)


# In[61]:

df.columns = ['age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol', 'blood_sugar', 'ecg', 'max_hr', 'exercise_induced_angina', 'st_depression', 'slope', 'num_major_vessels', 'thal', 'diagnosis']
df.index.names = ['patient']
df = df.convert_objects(convert_numeric=True)


# In[65]:

df.head()


# In[66]:

features = ['systolic_bp', 'tobacco_use', 'ldl_cholesterol',
            'abdominal_adiposity', 'family_history', 'type_a',
            'overall_obesity', 'alcohol_use', 'age']
response = 'heart_disease'


# In[63]:

from sklearn.linear_model import LinearRegression


# In[75]:

linear_reg = LinearRegression()
linear_reg.fit(df[features], df.response)


# In[79]:

df*features

