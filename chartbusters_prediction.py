import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from numpy import savetxt

data_train = pd.read_csv("Data_Train.csv")
data_test = pd.read_csv("Data_Test.csv")

data_views = data_train["Views"]
y = np.matrix(data_views).transpose()

data_train.drop("Views", axis=1, inplace=True)

X = np.concatenate((data_train, data_test), axis=0)

X = X.astype(np.str)

X = np.delete(X, 3, axis=1)

X = np.delete(X, 0, axis=1)

X = np.delete(X, 2, axis=1)

X = np.matrix(X)

labelencoder_name = LabelEncoder().fit(X[:, 0])
X[:, 0] = np.matrix(labelencoder_name.transform(X[:, 0])).transpose()

labelencoder_genre = LabelEncoder().fit(X[:, 1])
X[:, 1] = np.matrix(labelencoder_genre.transform(X[:, 1])).transpose()

time_stamp = np.char.split(X[:, 2], sep = ':')

for index in range(len(time_stamp)):
    time_x = np.char.split(X[index, 2], sep = ':')
    time_x = time_x.tolist()
    time_x = np.array(time_x)
    time_min = time_x[0]
    if "-" in time_min:
        t_m = float(time_x[1])
        t_s = float(time_x[2])
        X[index, 2] = str(t_m*60 + t_s)
    else:
        t_m = float(time_x[0])
        t_s = float(time_x[1])
        X[index, 2] = str(t_m*60 + t_s)
    
X[:, 4] = np.char.replace(X[:, 4], ',', '')

X[:, 4] = np.char.replace(X[:, 4], ',', '')
X[:, 4] = np.char.replace(X[:, 4], 'K', '*1000')
X[:, 4] = np.char.replace(X[:, 4], 'M', '*1000000')

for index in range(len(X[:, 4])):
    value = X[index, 4]
    if "*" in value:
        value_factor = np.char.split(value, sep = '*').tolist()
        f1 = float(value_factor[0])
        f2 = float(value_factor[1])
        v = f1*f2
        X[index, 4] = str(v)
        
X[:, 5] = np.char.replace(X[:, 5], ',', '')
X[:, 5] = np.char.replace(X[:, 5], 'K', '*1000')

for index in range(len(X[:, 5])):
    value = X[index, 5]
    if "*" in value:
        value_factor = np.char.split(value, sep = '*').tolist()
        f1 = float(value_factor[0])
        f2 = float(value_factor[1])
        v = f1*f2
        X[index, 5] = str(v)


#X = X.astype(float)

#savetxt('data1.csv', X, delimiter=',', fmt='%s')

onehotencoder = OneHotEncoder(categorical_features=[0, 1])
X = onehotencoder.fit_transform(X).toarray()

X_train = X[0:len(data_train), :]
X_test = X[len(data_train):, :]

X_train, X_t, y_train, y_t = train_test_split(X_train, y, test_size = 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
X_t = sc_X.transform(X_t)

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_t)
y_predtest = regressor.predict(X_test)

score_gb = regressor.score(X_t, y_t)
print (score_gb)

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_t, y_pred))

print (rmse)

uid = data_test["Unique_ID"]
solution = pd.DataFrame({'Unique_ID': uid, 'Views' : y_predtest})
solution.to_excel('Chartbusters_Prediction.xlsx', index = False)

