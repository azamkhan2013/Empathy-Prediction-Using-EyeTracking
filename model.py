#!/usr/bin/env python
# coding: utf-8

# In[15]:


# importing libraries 

import glob
import pandas as pd
import numpy as np
import missingno as msno

import matplotlib.pyplot as plt
from fancyimpute import IterativeImputer
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # Uploading the dataset in a different way 
# 
# I am droping the variables which are not helpful plus I am tranforming every excel file to a single row by taking mean 

# In[2]:


path = r"C:\Users\ak22755\EyeT\EyeT_group_dataset_III_image_name_letter_card_participant_**_trial_*.csv"
file_names = glob.glob(path)     # glob is used to get a list of all CSV files in the directory 

dfs = []
for file_name in file_names:
    # Read in the data
    df = pd.read_csv(file_name)
    df = df.replace(',', '.', regex=True)
    # Group by Participant name and calculate the mean for each variable
    group_vars = ['Participant name']
    # taking only those columns which has some relation with empathy
    mean_vars = [col for col in df.columns if col not in ['Unnamed: 0','Sensor','Project name','Export date','Recording name','Recording date',
 'Recording date UTC','Recording start time','Recording start time UTC','Timeline name','Recording Fixation filter name',
 'Recording software version','Recording resolution height','Recording resolution width','Recording monitor latency','Event',
 'Event value','Gaze point left X','Gaze point left Y','Gaze point right X','Gaze point right Y','Gaze direction left X','Gaze direction right X',
 'Gaze direction right Y','Gaze direction right Z','Pupil diameter right','Validity left','Validity right','Eye position right X (DACSmm)',
 'Eye position right Y (DACSmm)','Eye position right Z (DACSmm)','Gaze point left X (DACSmm)','Gaze point left Y (DACSmm)',
 'Gaze point right X (DACSmm)','Gaze point right Y (DACSmm)','Gaze point X (MCSnorm)','Gaze point Y (MCSnorm)','Gaze point left X (MCSnorm)',
 'Gaze point left Y (MCSnorm)','Gaze point right X (MCSnorm)','Gaze point right Y (MCSnorm)','Presented Stimulus name','Presented Media name',
 'Presented Media width','Presented Media height','Presented Media position X (DACSpx)','Presented Media position Y (DACSpx)',
 'Original Media height', 'Eye movement type','Fixation point X','Fixation point Y','Fixation point X (MCSnorm)','Fixation point Y (MCSnorm)',
 'Mouse position X','Mouse position Y']]
    df['Pupil diameter left'] = df['Pupil diameter left'].astype(float)

    df = df.groupby(group_vars)[mean_vars].mean().reset_index()
    
    # Append to the list of DataFrames
    dfs.append(df)

# Concatenate all data into one DataFrame
new_data = pd.concat(dfs, ignore_index=True)

# Print the resulting DataFrame
print(new_data)  # is used to concatenate all DataFrames in the data list into a single DataFrame


# In[3]:


new_data.info()


# In[5]:


new_data['Participant name'] = new_data['Participant name'].astype(str)


# In[6]:


def participant_names_to_int(new_data):
    new_data['Participant name'] =new_data['Participant name'].str[-2:].astype(int)
    return new_data
new_data= participant_names_to_int(new_data)


# In[7]:


# loading Questionnaire datasetIB 
questonare_data = pd.read_csv('m:\\pc\\downloads\\Questionnaire_datasetIB.csv', encoding='ISO-8859-1')


# In[8]:


# extracting Extended empathy scores from data2 so that we can merge it with data1 later.

def extract_em_score(questonare_data):
    extended_empathy_scores = {}
    for participant, score in questonare_data[questonare_data.index % 2 == 0][['Participant nr', 'Total Score extended']].values:
        extended_empathy_scores[participant] = score
    return extended_empathy_scores
extended_empathy_scores = extract_em_score(questonare_data)


# In[9]:


def merging_sets(new_data,extended_empathy_scores):
    new_data['Empathy Score'] = [extended_empathy_scores.get(name, 0) for name in new_data['Participant name']]
    return new_data
merged_data= merging_sets(new_data,extended_empathy_scores)


# In[10]:


merged_data.info()


# i think this looks good, and we can proceed to model building

# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[12]:


#scalling the data set using RobustScaler
from sklearn.preprocessing import RobustScaler

# Initialize the RobustScaler object
scaler = RobustScaler()


# In[13]:


# Scale the dataset
data_scaled = scaler.fit_transform(merged_data)
data_scaled = pd.DataFrame(data_scaled, columns =merged_data.columns )


# In[16]:


# assigning Empathy Score column to y
y =  data_scaled['Empathy Score']
# assigning all columns except Empathy Score to X
X = data_scaled.drop('Empathy Score', axis=1)
#finding variance of target variabel
var = np.var(y)


# In[17]:


#finding variance of the target variable
var = np.var(y)


# ## model 1 = RandomForestRegressor

# In[19]:


model1 = RandomForestRegressor(n_estimators=20, random_state=50)


# The results are really good, lets cross check it by using GroupKFold

# In[20]:


from sklearn.model_selection import GroupKFold


# Create a GroupKFold object with k=10 folds
n_splits=20
gkf = GroupKFold(n_splits=n_splits)
sc=[]
# Loop through the splits and train/test the model
for train_index, test_index in gkf.split(X, y, groups=X['Participant name']):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train = X_train.drop('Participant name', axis=1)
    X_test = X_test.drop('Participant name', axis=1)
    
    # Train and test the model
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = 1 - (mse/var)
    sc.append(r2)
    
    print("MSE:", mse)
    print("R-squared score:", r2)


# In[21]:


#now lets check he model performance 
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = 1 - (mse/var)
print("Mean Squared Error (MSE): ", mse)
print("R-squared score: ", r2)


# some improvement can be seen, lets try using some different model

# ## model2 : GradientBoostingRegressor

# In[22]:


from sklearn.ensemble import GradientBoostingRegressor
# Create a GradientBoostingRegressor object
model2 = GradientBoostingRegressor()

# Create a GroupKFold object with k=10 folds
n_splits = 20
gkf = GroupKFold(n_splits=n_splits)

# Initialize a list to store the R-squared scores
sc = []

# Loop through the splits and train/test the model
for train_index, test_index in gkf.split(X, y, groups=X['Participant name']):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train = X_train.drop('Participant name', axis=1)
    X_test = X_test.drop('Participant name', axis=1)
    
    # Train and test the model
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = 1 - (mse/var)
    sc.append(r2)
    
    print("MSE:", mse)
    print("R-squared score:", r2)


# In[23]:


#lets see on model2 
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = 1 - (mse/var)
print("Mean Squared Error (MSE): ", mse)
print("R-squared score: ", r2)


# We summarize from the above analysis that we can use this model to predict Empathy, keeping in mind the limitation and EDA it needed

# In[ ]:




