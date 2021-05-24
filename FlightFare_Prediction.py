#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


os.chdir('C:\\Users\\Monali\\OneDrive\\Desktop\\Case Study-DS\\Flight Fare Prediction\\Flight Fare')


# In[3]:


os.getcwd()


# ### Import dataset

# In[4]:


data = pd.read_excel('Data_Train.xlsx')


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.shape


# In[8]:


data.info()


# #### EDA 

# In[9]:


## Missing value analysis


# In[10]:


data.isnull().sum()


# In[11]:


sns.heatmap(data.isnull(), cbar=True)


# In[12]:


data.dropna(inplace=True)


# In[13]:


data.isnull().sum()


# In[14]:


sns.pairplot(data)


# In[15]:


data.columns


# In[16]:


data['Airline'].value_counts()


# In[17]:


data['Duration'].value_counts()


# In[18]:


data['Day_of_Journey'] = pd.to_datetime(data.Date_of_Journey, format = '%d/%m/%Y').dt.day


# In[19]:


data['Month_of_Journey'] = pd.to_datetime(data.Date_of_Journey, format = '%d/%m/%Y').dt.month


# In[20]:


data.head()


# In[21]:


data.drop(['Date_of_Journey'], axis=1, inplace = True)


# In[22]:


data.head()


# In[23]:


data['Dep_hour'] = pd.to_datetime(data['Dep_Time']).dt.hour
data['Dep_minutes'] = pd.to_datetime(data['Dep_Time']).dt.minute


# In[24]:


data.drop(['Dep_Time'], axis=1, inplace=True)


# In[25]:


data.head()


# In[26]:


data['Arr_hour'] = pd.to_datetime(data['Arrival_Time']).dt.hour
data['Arr_minutes'] = pd.to_datetime(data['Arrival_Time']).dt.minute

data.drop(['Arrival_Time'], axis=1, inplace=True)


# In[27]:


data.head()


# In[28]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[29]:


# Adding duration_hours and duration_mins list to train_data dataframe
data["Duration_hours"] = duration_hours
data["Duration_mins"] = duration_mins


# In[30]:


data.drop(['Duration'], axis=1, inplace=True)


# In[31]:


data.head()


# In[32]:


data['Additional_Info'].value_counts()


# since, additional info column has more than 80% of missing values or no info so we cam directly drop that column

# In[33]:


data.drop(['Additional_Info'], axis=1, inplace=True)


# In[34]:


data.head()


# Handling Categorical Data
# One can find many ways to handle categorical data. Some of them categorical data are,
# 
# 1. **Nominal data** --> data are not in any order --> **OneHotEncoder** is used in this case
# 2. **Ordinal data** --> data are in order --> **LabelEncoder** is used in this case

# In[35]:


data['Airline'].value_counts()


# In[36]:


Airline = data[['Airline']]


# In[37]:


Airline = pd.get_dummies(Airline, drop_first = True)


# In[38]:


Airline.head()


# In[39]:


Source = data[['Source']]
Source = pd.get_dummies(Source, drop_first = True)
Source.head()


# In[40]:


data['Source'].value_counts()


# In[41]:


data['Destination'].value_counts()


# In[42]:


Dest = data[['Destination']]
Dest = pd.get_dummies(Dest, drop_first = True)
Dest.head()


# In[43]:


data.drop(['Route'], axis = 1, inplace=True)


# In[44]:


data.head()


# In[45]:


data['Total_Stops'].value_counts()


# In[46]:


data.replace({"non-stop":0, "1 stop": 1, "2 stops":2, "3 stops":3, "4 stops":4}, inplace = True)


# In[47]:


data.head()


# In[48]:


data = pd.concat([data, Airline, Source, Dest], axis = 1)


# In[49]:


data.head()


# In[50]:


data.drop(['Airline', 'Source', 'Destination'], axis = 1, inplace=True)


# In[51]:


data.head()


# In[52]:


data.shape


# In[87]:


os.chdir('C:\\Users\\Monali\\OneDrive\\Desktop\\Case Study-DS\\Flight Fare Prediction\\Flight Fare')


# In[119]:


data_Test = pd.read_excel("Test_set.xlsx")


# In[92]:


data_Test.isnull().sum()


# In[93]:


data_Test.head()


# In[122]:


data_Test.drop(["Additional_Info"], axis = 1, inplace = True)


# In[123]:


data_Test.head()


# In[125]:


data_Test.drop(["Total_Stops"], axis = 1, inplace = True)


# In[103]:


data_Test.drop(["Route"], axis = 1, inplace = True)


# In[126]:


data_Test.head()


# Handling categorical values for column Date_of_Journey

# In[135]:


data_Test['Day_of_Journey'] = pd.to_datetime(data_Test.Date_of_Journey, format = '%d/%m/%Y').dt.day


# In[136]:


data_Test.head()


# In[138]:


data_Test['Month_of_Journey'] = pd.to_datetime(data_Test.Date_of_Journey, format = '%d/%m/%Y').dt.month


# In[139]:


data_Test.head()


# In[142]:


data_Test.drop(["Date_of_Journey"], axis =1 , inplace = True)


# In[143]:


data_Test.head()


# In[145]:


#Handling  Arrival Time
data_Test['Arr_Hour'] = pd.to_datetime(data_Test['Arrival_Time']).dt.hour
                                     


# In[146]:


data_Test['Arr_mins'] = pd.to_datetime(data_Test['Arrival_Time']).dt.minute


# In[147]:


data_Test.head()


# In[149]:


data_Test.drop(['Arrival_Time'], axis = 1, inplace = True)


# In[151]:


data_Test.head()


# In[152]:


data_Test['Dep_hour'] = pd.to_datetime(data_Test['Dep_Time']).dt.hour


# In[154]:


data_Test['Dep_mins'] = pd.to_datetime(data_Test['Dep_Time']).dt.minute


# In[155]:


data_Test.head()


# In[156]:


data_Test.drop(['Dep_Time'], axis = 1, inplace = True)


# In[157]:


data_Test.head()


# In[159]:


data_Test.drop(['Route'], axis = 1, inplace = True)


# In[162]:


Airline = data_Test[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first = True )


# In[164]:


data_Test['Airline'].value_counts()


# In[166]:


Airline.head()


# In[168]:


Source = data_Test[["Source"]]
Source = pd.get_dummies(Source,drop_first = True )


# In[169]:


Source.head()


# In[170]:


Destination = data_Test[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first = True)


# In[171]:


Destination.head()


# In[173]:


data_Test = pd.concat([data_Test, Airline, Source, Destination], axis = 1)


# In[174]:


data_Test.head()


# In[176]:


data_Test.drop(['Airline'], axis = 1, inplace = True)


# In[177]:


data_Test.drop(['Source'], axis = 1, inplace = True)


# In[178]:


data_Test.drop(['Destination'], axis = 1, inplace = True)


# In[179]:


data_Test.head()


# In[180]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(data_Test["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[181]:


data_Test.head()


# In[182]:


data_Test["Duration_hours"] = duration_hours
data_Test["Duration_mins"] = duration_mins


# In[183]:


data_Test.head()


# In[184]:


data_Test.drop(['Duration'], axis = 1, inplace = True)


# In[185]:


data_Test.head()


# In[186]:


plt.figure(figsize=(20,20))
sns.heatmap(data_Test.corr(), annot=True, cmap='BuGn_r')

plt.show()


# In[ ]:





# #### Feature selection

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[54]:


# Find correlation between independent variable and dependent variable


# In[55]:


plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), annot=True, cmap='BuGn_r')

plt.show()


# In[56]:


### Data Cleaning done#####


# ## Model creation
# 1. Its a regression problem
# 2. Supervised problem
# 3. Decision tree regression 
# 4. Random Forest regresion
# 5. linear regression 

# In[57]:


## Data need to split in test and train set 


# In[58]:


data.columns


# In[59]:


X = data.loc[:,['Total_Stops', 'Day_of_Journey', 'Month_of_Journey',
       'Dep_hour', 'Dep_minutes', 'Arr_hour', 'Arr_minutes', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]


# In[60]:


Y = data.loc[:,['Price']]


# In[61]:


X.head()


# In[62]:


Y.head()


# In[63]:


## Random Forest 


# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[65]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)


# In[66]:


reg_rf = RandomForestRegressor()


# In[67]:


reg_rf.fit(X_train, Y_train)


# In[68]:


y_pred = reg_rf.predict(X_test)


# In[188]:


reg_rf.score(X_train, Y_train)


# In[189]:


reg_rf.score(X_test, Y_test)


# In[71]:


y_pred


# In[72]:


##Evaluate of Regression Model 


# In[73]:


## RMSE, MSE, MAE


# In[74]:


from sklearn import metrics


# In[75]:


print('MAE', metrics.mean_absolute_error(Y_test, y_pred))


# In[76]:


print('MSE', metrics.mean_squared_error(Y_test, y_pred))


# In[77]:


np.sqrt(metrics.mean_squared_error(Y_test, y_pred))


# In[78]:


## Hyperparameter tuning


# In[79]:


from sklearn.model_selection import RandomizedSearchCV


# In[80]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[81]:


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[82]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[83]:


rf_random.fit(X_train,Y_train)


# In[84]:


Prediction = rf_random.predict(X_test)


# In[85]:


print('MAE:', metrics.mean_absolute_error(Y_test, prediction))
print('MSE:', metrics.mean_squared_error(Y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, prediction)))


# In[ ]:


## Save the model


# In[ ]:


import pickle   ## Serialize file
# open a file where we want to store a file
file = open('flightPrice.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:


model = open('flightPrice.pkl','rb')
forest = pickle.load(model)


# In[190]:


y_prediction = forest.predict(X_test)


# In[ ]:




