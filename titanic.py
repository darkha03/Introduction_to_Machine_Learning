import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('titanic.csv')

# Describe the data
# print(data.shape)
# print(data.head())
# print(data.describe())

# Clean the data (find null value)
# print(data.isnull().sum())

# Calculate average age group by sex and pclass
# print("The average age by sex and pclass:")
# print(data.groupby(['Sex','Pclass']).agg(
#     avg_age=('Age','mean')
# ))

# Find number of minor children
# print("Number of children by PClass:")
# print(data[data['Age'] < 18].groupby('Pclass').agg(
#     nb_child = ('Pclass','count')
# ))

# Replace the missing age data with average age by pclass

# Adjust the characteristic
# print("Change sex to numeric value (male:0, female:1)")
# data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
# print(data.head())

# Create a category for age
# data["AgeGroup"] = pd.cut(data["Age"], bins=[0,12,18,60,100], labels=["Child","Teen","Adult","Senior"])
# print(data.head())
# print(data.isnull().sum())

# data = pd.get_dummies(data, columns=["AgeGroup"], drop_first=True) #Drop first bc it can be deducted from 3 other columns
# print(data.head())


# Find correlation 
# print(data.corr(numeric_only=True)["Survived"].sort_values(ascending=False))

# Choose variable for ML
features = ['Fare', 'Parents/Children Aboard', 'Age', 'Pclass']
X = data[features]
y = data['Survived']


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)

importances = model.feature_importances_
features = X.columns

# pd.Series(importances, index=features).sort_values().plot(kind="barh")
# plt.title("Importance des variables")
# plt.xlabel("Importance")
# plt.show()

import seaborn as sns
data_test = X_test.copy()
data_test["Survived_real"] = y_test
data_test["Survived_pred"] = model.predict(X_test)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.barplot(data=data_test, x="Pclass", y="Survived_real")
plt.title("Taux de survie réel")
plt.subplot(1,2,2)
sns.barplot(data=data_test, x="Pclass", y="Survived_pred")
plt.title("Taux de survie prédit")
plt.show()
