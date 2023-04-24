#!/usr/bin/env python
# coding: utf-8

# # Empathy Estimator Model

# In[1]:


#pip install missingno
#pip install fancyimpute


# In[1]:


# importing libraries 

import glob
import pandas as pd

import missingno as msno

import matplotlib.pyplot as plt
from fancyimpute import IterativeImputer
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ## Loading Data sets

# In[2]:


# Get data file names
pa_nam = r"C:\Users\ak22755\EyeT\EyeT_group_dataset_III_image_name_letter_card_participant_**_trial_*.csv"
file_nam= glob.glob(pa_nam)     # glob is used to get a list of all CSV files in the directory 

ddf = []
for filename in file_nam:
    ddf.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
data1 = pd.concat(ddf, ignore_index=True)  # is used to concatenate all DataFrames in the data list into a single DataFrame


# In[3]:


# loading Questionnaire datasetIB 
data2 = pd.read_csv('m:\\pc\\downloads\\Questionnaire_datasetIB.csv', encoding='ISO-8859-1')


# In[4]:


data1.shape


# In[5]:


data2


# ## Exploring Data set 1

# In[6]:


data1


# In[7]:


data1.info()


# ## Preprocessing

# In[17]:


# Removing the variables which have no variance

import matplotlib.pyplot as plt

no_variance_cols = []
for col in data1.columns:
    if data1[col].nunique() <= 1:
        no_variance_cols.append(col)
        
print("Columns with no variance or constant values:")
print(no_variance_cols)

for col in no_variance_cols:
    plt.hist(data1[col])
    plt.title(col)
    plt.show()


# In[18]:


# droping the above columns with no variance or constant values
data1 = data1.drop(columns=no_variance_cols)


# ## Cleaning and transformation

# In[19]:


data1.isnull().sum()


# In[20]:


# creating a bar plot of the missing values
msno.bar(data1,color='red')
plt.show()


# The above plot illustrates the proportion of missing values in each column.

# In[21]:


# Pupil measurement were extracted at 40Hz frequency and its very important in prediction of our target variable. 
# Also NANs are alot because of the frequency change so we will simply drop the null values in it.

data1 = data1.dropna(subset=['Pupil diameter left', 'Pupil diameter right'])


# In[14]:


# now droping columns which has null values more than 70 %
def drop_null_columns(data, null_threshold=70):
    # Calculate the percentage of null values in each column
    null_percentages = (data.isnull().sum() / len(data)) * 100
    
    # Get the names of columns with null percentage greater than the threshold
    null_columns = null_percentages[null_percentages > null_threshold].index
    
    # Drop the null columns from the dataset
    data.drop(null_columns, axis=1, inplace=True)

    
drop_null_columns(data1, 70)


# In[22]:


#again visualizing null values

msno.bar(data1,color='green')


# The above plot can highlite that there are no variables which has null values greater than 70%

# In[23]:


# Now checking again if there is any variable which has no variance and droping it 

# finding columns with no variance or constant values
no_variance_cols = []
for col in data1.columns:
    if data1[col].nunique() <= 1:
        no_variance_cols.append(col)

# printing the columns with no variance or constant values
print('Columns with no variance or constant values:', no_variance_cols)
# droping the above columns with no variance or constant values
data1 = data1.drop(columns=no_variance_cols)


# In[24]:


data1.info()


# In[25]:


# our dataset has variables which are inccorectly filled lets tranform it.

data1 = data1.replace(',','.',regex = True)
    


# In[26]:


data1.info()


# ## Preprocessing of data2

# In[27]:


# extracting Extended empathy scores from data2 so that we can merge it with data1 later.

def extract_em_score(data2):
    extended_empathy_scores = {}
    for participant, score in data2[data2.index % 2 == 0][['Participant nr', 'Total Score extended']].values:
        extended_empathy_scores[participant] = score
    return extended_empathy_scores
extended_empathy_scores = extract_em_score(data2)


# In[28]:


extended_empathy_scores


# In[29]:


# Changing values of Particpant name to int so that it can be easily match with the other data set 
def participant_names_to_int(data1):
    data1['Participant name'] = data1['Participant name'].str[-2:].astype(int)
    return data1
data1= participant_names_to_int(data1)


# In[30]:


# merging the extended empathy scores with data1
def merging_sets(data1,extended_empathy_scores):
    data1['Empathy Score'] = [extended_empathy_scores.get(name, 0) for name in data1['Participant name']]
    return data1
merged_data= merging_sets(data1,extended_empathy_scores)


# In[31]:


merged_data.sort_values('Participant name',inplace= True)


# In[32]:


merged_data['Participant name'].unique()


# In[27]:


merged_data.info()


# In[33]:


# changing type of Eyetracker to datetime  
merged_data['Eyetracker timestamp'] = pd.to_datetime(merged_data['Eyetracker timestamp'], unit='ms')


# In[34]:


merged_data.info()


# In[35]:


# Trying to change variables which are obj or string to numeric form
def convert_to_numeric(merged_data):
    for col in merged_data.columns:
        if not pd.api.types.is_numeric_dtype(merged_data[col]):
            try:
                merged_data[col] = pd.to_numeric(merged_data[col])
            except:
                print(f"Column {col} could not be converted to numeric.")
    return merged_data


# In[36]:


convert_to_numeric(merged_data)


# In[37]:


merged_data.isnull().sum()   


# In[38]:


from sklearn.impute import SimpleImputer, IterativeImputer

# separate numeric and categorical columns
numeric_cols = merged_data.select_dtypes(include='number').columns
categorical_cols = merged_data.select_dtypes(include='object').columns

# impute missing values in numeric columns with IterativeImputer
imputer_numeric = IterativeImputer()
merged_data[numeric_cols] = imputer_numeric.fit_transform(merged_data[numeric_cols])

# impute missing values in categorical columns with SimpleImputer
imputer_categorical = SimpleImputer(strategy='most_frequent')
merged_data[categorical_cols] = imputer_categorical.fit_transform(merged_data[categorical_cols])


# In[39]:


# check if there are any remaining missing values
print(merged_data.isnull().sum())  #clearly there are no null values


# In[40]:


msno.bar(data1,color='blue')


# The above plot can summarise that there are no null values. 

# # Visualisation
# we will use different plots/graphs to see the distributions of different variables

# In[41]:


merged_data.hist(figsize=(20,20), bins=20, alpha=0.5) 


# In[42]:


# Visualizing sum of gaze event duration with empathy score 

# Create a dictionary of Empathy Scores keyed by Participant name
empathy_dict = dict(zip(merged_data['Participant name'], merged_data['Empathy Score']))

# Group the data by Participant ID and calculate the sum of Gaze event duration
grouped_data = merged_data.groupby('Participant name')['Gaze event duration'].sum().reset_index()

# Add a new column to the resulting DataFrame with the Empathy Score for each participant
grouped_data['Empathy Score'] = grouped_data['Participant name'].map(empathy_dict)

# Plot a scatter plot with the Empathy Score on the x-axis and the Total Gaze event duration on the y-axis
sns.scatterplot(x='Empathy Score', y='Gaze event duration', data=grouped_data)

# Add labels and a title to the plot
plt.xlabel('Empathy Score')
plt.ylabel('Total Gaze event duration')
plt.title('Total Gaze event duration vs Empathy Score')

# Show the plot
plt.show()


# Most of the Empathy scores are reported when total Gaze event duration is 0.5

# In[43]:


# Visualizing number of Eye movement types among Participants

# Set the figure size
plt.figure(figsize=(10, 8))

# Create the countplot
sns.countplot(x='Participant name', hue='Eye movement type', data=merged_data, palette='Set1')

# Add a title and axis labels
plt.title('Eye Movement Type by Participant')
plt.xlabel('Participant Name')
plt.ylabel('Count')

# Display the plot
plt.show()


# Most of the participants Eye movement type is Fixation 

# In[44]:


#Trying to generate a timeseries plot of Eyetracker timestamp with Participant name

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(merged_data['Participant name'], merged_data['Eyetracker timestamp'], color='red')
plt.xlabel('Participant name')
plt.ylabel('Eyetracker timestamp')
plt.title('Time Series Plot - Participant name vs Eyetracker timestamp')
plt.xlim([merged_data['Participant name'].min(), merged_data['Participant name'].max()])

plt.show()


# This plot generates a time series plot of the Participant name over time

# In[45]:


##Trying to generate a timeseries plot of Eyetracker timestamp with pupil diameter Right
plt.figure(figsize=(20, 20))
plt.subplot(2, 1, 2)
plt.plot(merged_data['Eyetracker timestamp'], merged_data['Pupil diameter right'], color='red',marker='o', label='Pupil diameter (right)')
plt.xlabel('Time Stamp')
plt.ylabel('Pupil diameter right')
plt.title('Time Series Plot - Pupil diameter (right) vs Time Stamp')

plt.xlim([0, 1e16])
# Add horizontal lines to indicate the average pupil diameter for the right and left eyes, respectively
average_diameter_right = merged_data['Pupil diameter right'].mean()
plt.axhline(average_diameter_right, color='blue', linestyle='--', label='Average diameter (right)')
plt.show()


# This plot generates a time series plot of the Right pupil diameter over time it decreases

# In[46]:


##Trying to generate a timeseries plot of Eyetracker timestamp with pupil diameter Left
plt.figure(figsize=(20, 20))
plt.subplot(2, 1, 2)
plt.plot(merged_data['Eyetracker timestamp'], merged_data['Pupil diameter left'], color='blue',marker='o', label='Pupil diameter (Left)')
plt.xlabel('Time Stamp')
plt.ylabel('Pupil diameter left')
plt.title('Time Series Plot - Pupil diameter (left) vs Time Stamp')

plt.xlim([0, 1e16])
# Add horizontal lines to indicate the average pupil diameter for the right and left eyes, respectively
average_diameter_left = merged_data['Pupil diameter left'].mean()
plt.axhline(average_diameter_left, color='red', linestyle='--', label='Average diameter (left)')
plt.show()


# This plot generates a time series plot of the Left pupil diameter over time it decreases

# In[48]:


# Distribution of Average empathy score among participants

average_empathy = merged_data.groupby('Participant name')['Empathy Score'].mean()

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(average_empathy.index, average_empathy)
plt.xlabel('Participant name')
plt.ylabel('Average Empathy Score')
plt.title('Average Empathy Score by Participant')
plt.show()


# Participant 11 has the highest Average score while Participant 7 has the lowest

# In[49]:


# computing the correlation matrix
corr_matrix = merged_data.corr()

# print the correlation matrix
print(corr_matrix)


# In[50]:


# ploting the correlation heatmap
corr = merged_data.corr()
sns.set(style='white')
plt.figure(figsize=(30,30))
sns.heatmap(merged_data.drop('Empathy Score', axis=1).corr(), annot=True, cmap='coolwarm')
# show the plot
plt.show()


# In[51]:


import numpy as np 
# Compute correlation matrix
corr_matrix = merged_data.drop('Empathy Score', axis=1).corr()

# Creating mask to display only upper triangle of the heatmap
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True

# Filter out correlation values less than 0.7
corr_matrix[corr_matrix.abs() < 0.7] = 0

# Ploting heatmap with filtered correlation values
sns.set(style='white')
plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, vmax=1.0, vmin=-1.0, square=True)

# Show the plot
plt.show()


# In[52]:


# droping the columns with correlation greater than 0.8
cols_to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > 0.8:
            colname = corr_matrix.columns[i]
            cols_to_drop.add(colname)

# Drop columns from merged_data
merged_data.drop(cols_to_drop, axis=1, inplace=True)


# In[53]:


merged_data.info()


# In[54]:


#based on my understanding and looking at the above plot i can drop few more variables, because I believe its not going to effect my predictions.


add_cols_to_drop = ['Unnamed: 0','Project name', 'Export date', 'Recording name', 'Recording date', 'Recording date UTC', 'Recording start time', 'Recording start time UTC', 'Timeline name', 'Presented Stimulus name', 'Presented Media name','Eye movement type']
merged_data.drop(add_cols_to_drop, axis=1, inplace=True)


# In[55]:


merged_data.info()  # now the data is ready for model building 


# # Model building 

# In[56]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[57]:


X = merged_data.drop('Empathy Score', axis=1)
y =  merged_data['Empathy Score']


# In[58]:


# Scaling the data set 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# ## RandomForestRegressor

# In[60]:



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[61]:


rf_regressor = RandomForestRegressor(n_estimators=20, random_state=42)


# In[62]:


rf_regressor.fit(X_train, y_train)


# In[63]:


y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE): ", mse)
print("R-squared score: ", r2)


# ## The above MSE and R-quared score is very unexpected.
# ## Reasons for it could be many such as Data Leakage, or the way we have imported the data.
# ## Also it is obvious that the empathy scores we extracted from Questionare dataset is 30 and is fixed , so it will be repeated when we merged with our main data set
# 

# ### To fix the above issue, we need to upload the data in a different way and then see if our model could be build better.
# ### Plus point is that now we know what features are important we dnt need to that whole thing again 

# # Uploading the dataset in a different way 
# 
# I am droping the variables which are not helpful plus I am tranforming every excel file to a single row by taking mean 

# In[64]:


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


# In[65]:


new_data.info()


# Now we need to change the Participant name data type so that we can build model, and also we need to merge the empathy score with it

# In[66]:


new_data['Participant name'] = new_data['Participant name'].astype(str)


# In[67]:


def participant_names_to_int(new_data):
    new_data['Participant name'] =new_data['Participant name'].str[-2:].astype(int)
    return new_data
new_data= participant_names_to_int(new_data)


# In[68]:


# loading Questionnaire datasetIB 
questonare_data = pd.read_csv('m:\\pc\\downloads\\Questionnaire_datasetIB.csv', encoding='ISO-8859-1')


# In[69]:


# extracting Extended empathy scores from data2 so that we can merge it with data1 later.

def extract_em_score(questonare_data):
    extended_empathy_scores = {}
    for participant, score in questonare_data[questonare_data.index % 2 == 0][['Participant nr', 'Total Score extended']].values:
        extended_empathy_scores[participant] = score
    return extended_empathy_scores
extended_empathy_scores = extract_em_score(questonare_data)


# In[70]:


def merging_sets(new_data,extended_empathy_scores):
    new_data['Empathy Score'] = [extended_empathy_scores.get(name, 0) for name in new_data['Participant name']]
    return new_data
merged_data= merging_sets(new_data,extended_empathy_scores)


# In[71]:


merged_data.info()


# i think this looks good, and we can proceed to model building

# In[81]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[82]:


#scalling the data set using RobustScaler
from sklearn.preprocessing import RobustScaler

# Initialize the RobustScaler object
scaler = RobustScaler()


# In[83]:


# Scale the dataset
data_scaled = scaler.fit_transform(merged_data)
data_scaled = pd.DataFrame(data_scaled, columns =merged_data.columns )


# In[84]:


# assigning Empathy Score column to y
y =  data_scaled['Empathy Score']
# assigning all columns except Empathy Score to X
X = data_scaled.drop('Empathy Score', axis=1)
#finding variance of target variabel
var = np.var(y)


# In[85]:


#import numpy as np
var = np.var(y)


# ## model 1 = RandomForestRegressor

# In[86]:


model1 = RandomForestRegressor(n_estimators=20, random_state=50)


# The results are really good, lets cross check it by using GroupKFold

# In[87]:


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


# In[88]:


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

# In[92]:


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


# In[93]:


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




