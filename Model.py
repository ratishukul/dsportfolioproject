#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This notebook seeks to develop a model that can predict medical charges faced by individuals using features such as Age, Sex, BMI, number of Children, Smoker, and Region. An XGBoost regressor model is chosen, that is trained using randomized grid-search and early stopping, with its performance being evaluated using k-fold validation.

# ## **About Dataset**
# Information provided on the data set page.

#
# **Overview of the dataset:**\
# The medical insurance dataset contains information about a number of factors that can affect medical expenses, including age, sex, BMI, smoking status, number of children, and region. This dataset can be used to train a machine learning model that can predict medical expenses for new customers.
#
# To provide insights into the key factors that contribute to higher insurance costs and help the company make more informed decisions regarding pricing and risk assessment.
#
# The dataset contains **2.7K rows and 7 columns**
# Columns include
#

#
# **Problem Statement:**
#
# 1. What are the most important factors that affect medical expenses?
# 2. How well can machine learning models predict medical expenses?
# 3. How can machine learning models be used to improve the efficiency and profitability of health insurance companies?

# ## Import modules and loading data

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from xgboost import XGBRegressor, cv, DMatrix, plot_importance
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = r""
df = pd.read_csv(path)


# # 1 | Exploratory Data Analysis
# Examining features, data types, missing values etc.

# In[3]:


df.head()


# The structure of the data set.

# In[4]:


df.info()


# As described in the introduction, it is a rather small data set with seven columns and a total of 2772 rows. There does not appear to be any missing values. The features consists of three numerical variables and three categorical variables, and the target variable is continuous.

# In[5]:


df.describe()


# Selected statistics for the numerical columns.
# * Age contains values between 18 to 64 years, so we are only considering adult individuals.
# * bmi spans from 16 to 53 with a mean of 31, and there appear to be a large variability considering the quantiles.
# * children is a discrete numerical variable, where people in the data set have between zero to five kids.
# * charges, the target variable, spans a wide range from 1122 to 63770, and it seems to be quite skewed comparing the mean to the median.

# ## Categorical Features

# In[6]:


sns.countplot(x=df["sex"])


# The data set seems to have an equal representation of females and males.

# In[7]:


sns.countplot(x=df["smoker"])


# There are around four times as many non-smokers as smokers in the data set.

# In[8]:


sns.countplot(x=df["region"])


# The four regions are quite equally represented.

# # 2 | Data Preprocessing

# With no missing values and no feature engineering used, this section is rather short.

# ## Dummy Variables

# In[9]:


df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)


# ## Correlation Analysis

# Having converted the categorical features to dummy variables, we can consider the correlations.

# In[10]:


plt.figure(figsize=(14, 9))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Train-Test Split

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(df.drop("charges", axis=1).values, df["charges"].values, train_size=0.8)


# # 3 | Model
# XGBoost Regressor | Grid Search | K-fold CV

# ## Finding Optimal Parameters
# Using randomized grid search

# First an XGBRegressor object is created to be used for prediction of continuous values. Only a few parameters are set, as the rest will be found using randomized grid search.

# In[12]:


xgb = XGBRegressor(n_estimators=500, n_jobs=4)


# Defining the parameter space of which a subset will be tested through searching and the best of those tested will be used.

# In[13]:


param_grid = {
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1],
    "gamma": [0.5, 1, 1.5, 3],
    "colsample_bytree": [0.6, 0.8, 1.0]
}


# Additional parameters for the fitting process. In this case, they are simply related to the early stopping process, i.e. the training will halt if the performance on the test set has not improved after a selected number of rounds.

# In[14]:


fit_params = {
    "eval_set":[(X_test, y_test)],
    "early_stopping_rounds": 10,
    "eval_metric": "rmse"
}


# Object for K-fold cross validation.

# In[15]:


kf = KFold(n_splits=5, shuffle = True, random_state = 42)


# Creating a random search CV object to search the defined parameter space using K-fold cross validation to find the optimal parameter set. A total of 10 parameter sets are tested from the defined parameter space.

# In[16]:


random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=10, scoring="neg_mean_squared_error", n_jobs=4, cv=kf.split(X_train,y_train), verbose=0, random_state=42)


# This search can take some time, so as an alternative to running this calibration one can select a random combination of the parameter set or e.g. use the following:\
# best_params = {'subsample': 1,
#  'learning_rate': 0.05,
#  'gamma': 0.5,
#  'colsample_bytree': 1.0}

# Running the fit method on the training data set. Usually, a train-test split is not needed for K-fold cross validation, however it is used here for the early stopping process.

# In[ ]:


random_search.fit(X_train, y_train, **fit_params)


# ## Evaluating Performance
# Using K-fold Cross Validation

# The best set of parameters identified from the random grid search.

# In[18]:


random_search.best_params_


# Creating the data set for the final K-fold cross validation of the model.

# In[19]:


data = DMatrix(data=df.drop("charges", axis=1), label=df["charges"], feature_names=df.drop("charges", axis=1).columns)


# For the final validation, the best set of parameters determined from the randomized search is used.

# In[20]:


params_final = {**{"objective": "reg:squarederror", "n_jobs": 4, "eval_metric": "rmse"}, **random_search.best_params_}
xgb_cv = cv(dtrain=data, params=params_final, nfold=5, num_boost_round=500, early_stopping_rounds=10)


# Looking at performance by looking at the RMSE on the last iteration.

# In[21]:


xgb_cv.tail(1)


# Note: Since early stopping did not stop the training, better performance could likely be obtained using a higher number of rounds. However, to stay consistent, the same number of rounds were used here as for the randomized grid search, and increasing the number of rounds in the grid search would make the process become even more time consuming.

# ## Feature Importance

# In order to determine the feature importance, a new XGBRegressor object is created with the best parameters (as far as I know, the cross-validation class does not have this method).

# In[22]:


xgb_ = XGBRegressor(n_estimators=500, n_jobs=4, **random_search.best_params_)


# In[23]:


xgb_.fit(X=df.drop("charges", axis=1), y=df["charges"])


# In[24]:


plot_importance(xgb_)


# As seen from the plot above, the BMI and age of the insurance buyer seem to be the most relevant factors for the charged price.
