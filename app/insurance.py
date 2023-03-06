import pandas as pd#import important libraries
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regression model

file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/insurance.csv"
insurance = pd.read_csv(file_name)#our data

# droping region
insurance.drop(["region"], axis=1, inplace=True) 

# Changing binary categories to 1s and 0s
insurance['sex'] = insurance['sex'].map(lambda s :1  if s == 'female' else 0)
insurance['smoker'] = insurance['smoker'].map(lambda s :1  if s == 'yes' else 0)

X = np.array(insurance.iloc[:, 0:5])#for our inputs
y = np.array(insurance.iloc[:, 5:])#for our outputs

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#spliting data

random_forest_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=13)  # Create a instance for Random Forest Regression model
random_forest_reg.fit(X_train, y_train)  # Fit data to the model

pickle.dump(random_forest_reg, open('insurance.pkl', 'wb'))#we download our model using pickle
