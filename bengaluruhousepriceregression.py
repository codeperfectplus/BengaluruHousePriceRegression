# -*- coding: utf-8 -*-
"""
Source=  [MachineHack](https://www.machinehack.com/course/predicting-house-prices-in-bengaluru/)

@Features
- Area_type–   describes the area  
- Availability  –   when it can be possessed or when it is ready(categorical and time-series)  
- Location  –   where it is located in Bengaluru  
- Price  –   Value of the property in lakhs(INR)  
- Size  –   in BHK or Bedroom (1-10 or more)  
- Society  –   to which society it belongs  
- Total_sqft  –   size of the property in sq.ft  
- Bath  –   No. of bathrooms  
- Balcony  –   No. of the balcony  
"""

import numpy as np  # For Linear Algebra and math
import pandas as pd # For Dataframe
import re     #regex module
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.preprocessing import LabelEncoder           # Change String Data to int value by LabelEncoding 
from sklearn.model_selection import train_test_split     # Split Data Into train_test for Check accuracy
from sklearn.metrics import mean_absolute_error          # Mean Absolute Error
from sklearn.ensemble import RandomForestRegressor       # Random_Forest_Regressor
from sklearn.metrics import r2_score                     # R2_Score
from joblib import dump                                  # To Save Model

# Load train Data into train_df variable
train_df = pd.read_csv('train.csv')

print("Train Dataset :\n")
print(train_df.head())

# Analyze DataType In Each-Columns
print("Data-types of each columns in train dataset:\n")
print(train_df.dtypes)
#There Is Need to Change Data Types

print("Training Dataset Size  :", train_df.shape)

#check missing values (NaNs) in train dataset 👇
print("missing values in train dataset column wise:\n")
print(train_df.isnull().sum())

print("percentage of Missing Values Society Column In Train dataset :",int(5502/train_df.shape[0]*100),"%")

""" Reason  For Missing Values
  - Maybe all these house doesn't belong to any society.
  - Or, because of some data gathering problem these records has missing society value.
"""

# Outlier Value Visualizations  

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(train_df['price'], ax=ax)
ax.set(xlabel= 'price in lakhs(INR)', title='Box plot of price')
#plt.savefig("Graph/Box plot of price")
plt.show()

# find mode (most frequent occuring price value) of price
print("Mode of Price :", train_df['price'].mode()[0] )

print('Descriptive statistics of price column:\n')
print(train_df['price'].describe())


# Missing Value Visualization  
### Column-Wise

fig, ax = plt.subplots(figsize=(10, 5))
train_couts = train_df['balcony'].fillna('missing').value_counts(normalize=True)
sns.barplot(y = train_couts.index, x= train_couts.values*100)
ax.set(title='Missing Value Visualization For Training Dataset \n Column : Balcony')
#plt.savefig("Graph/Balcony Data Visualization")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
train_couts = train_df['bath'].fillna('missing').value_counts(normalize=True)
sns.barplot(y = train_couts.index, x= train_couts.values*100)
ax.set(title='Missing Value Visualization For Training Dataset \n Column : Bathroom')
#plt.savefig("Graph/Bathroom Data Visualization")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
train_couts = train_df['area_type'].fillna('missing').value_counts(normalize=True)
sns.barplot(y = train_couts.index, x= train_couts.values*100)
ax.set(title='Missing Value Visualization \n Column : Area Type')
#plt.savefig("Graph/Area Type Data Visualization")
plt.show()

"""No. of House For Ready TO Move"""

print(train_df['availability'].value_counts()[0:1])

print(train_df.area_type.value_counts())

print(train_df['total_sqft'].unique())

"""----
----
"""

replace_area_type = {'Super built-up  Area': 0, 'Built-up  Area': 1, 'Plot  Area': 2, 'Carpet  Area': 3}
train_df['area_type'] = train_df.area_type.map(replace_area_type)

def replace_availabilty(my_string):
    if my_string == 'Ready To Move':
        return 0
    elif my_string == 'Immediate Possession':
        return 1
    else:
        return 2

train_df['availability'] = train_df.availability.apply(replace_availabilty)

def preprocess_total_sqft(my_list):
    if len(my_list) == 1:
        
        try:
            return float(my_list[0])
        except:
            strings = ['Sq. Meter', 'Sq. Yards', 'Perch', 'Acres', 'Cents', 'Guntha', 'Grounds']
            split_list = re.split('(\d*.*\d)', my_list[0])[1:]
            area = float(split_list[0])
            type_of_area = split_list[1]
            
            if type_of_area == 'Sq. Meter':
                area_in_sqft = area * 10.7639
            elif type_of_area == 'Sq. Yards':
                area_in_sqft = area * 9.0
            elif type_of_area == 'Perch':
                area_in_sqft = area * 272.25
            elif type_of_area == 'Acres':
                area_in_sqft = area * 43560.0
            elif type_of_area == 'Cents':
                area_in_sqft = area * 435.61545
            elif type_of_area == 'Guntha':
                area_in_sqft = area * 1089.0
            elif type_of_area == 'Grounds':
                area_in_sqft = area * 2400.0
            return float(area_in_sqft)
        
    else:
        return (float(my_list[0]) + float(my_list[1]))/2.0

train_df['total_sqft'] = train_df.total_sqft.str.split('-').apply(preprocess_total_sqft)

size_mode = train_df['size'].mode()[0]
train_df.loc[train_df['size'].isna(), 'size'] = size_mode

print(train_df['size'].unique())

train_df['size'] = train_df['size'].apply(lambda x: x.split(' ')[0])

train_df['size'] = train_df['size'].astype('float64')

print(train_df.head())

print(train_df.dtypes)

print(train_df.isnull().sum())

print(train_df.balcony.value_counts())

bath_med = train_df['bath'].median()
balcony_med = train_df['balcony'].median()

train_df.loc[train_df['bath'].isna(), 'bath'] = bath_med
train_df.loc[train_df['balcony'].isna(), 'balcony'] = balcony_med

print(train_df.isnull().sum())

print(train_df['location'].value_counts())

mode_loc =train_df['location'].mode()[0]

train_df.loc[train_df['location'].isna(), 'location'] = mode_loc

print(train_df.isnull().sum())


#Society : Society Has Lot Of Null Values So It Will Not Consider.

print(train_df.dtypes)

print(train_df['society'].value_counts())

corelation =train_df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corelation,cmap="YlGnBu")
plt.title("Corelation Graph")
#plt.savefig("Graph/Corelation Graph")
plt.show()

print(corelation)

# Apply RandomForest
X = train_df[['area_type', 'availability' ,'size', 'total_sqft', 'bath', 'balcony']]
y = train_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

rf = RandomForestRegressor(n_estimators=600,
                             max_depth= 5,
                             max_leaf_nodes= 15,
                             min_samples_leaf= 3,
                             min_samples_split= 10,
                             random_state=0)
rf.fit(X_train, y_train)

predict_y = rf.predict(X_test)

#Mean-Absolute-Error
print(mean_absolute_error(y_test,predict_y))

#Accuracy Score
print("Accuracy Score :",int(r2_score(y_test, predict_y)*100),"%")