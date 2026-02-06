import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('titanic.csv')

# Add age group category
data["AgeGroup"] = pd.cut(data["Age"], bins=[0,12,18,40,60,80,100], labels=[1,2,3,4,5,6])
features = ['Fare', 'Parents/Children Aboard', 'AgeGroup', 'Pclass']
X = data[features]
y = data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)

importances = model.feature_importances_
features = X.columns


data_test = X_test.copy()
data_test["Survived_real"] = y_test
data_test["Survived_pred"] = model.predict(X_test)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.barplot(data=data_test, x="AgeGroup", y="Survived_real")
plt.title("Taux de survie réel")
plt.subplot(1,2,2)
sns.barplot(data=data_test, x="AgeGroup", y="Survived_pred")
plt.title("Taux de survie prédit")
plt.show()

# ages = np.linspace(0, 80, 100)
# mean_values = pd.concat([X.mean().to_frame().T] * 100, ignore_index=True)
# mean_values["Age"] = ages
# proba = model.predict_proba(mean_values)[:,1]
# plt.plot(ages, proba)
# plt.xlabel("Âge")
# plt.ylabel("Probabilité prédite de survie")
# plt.title("Effet de l'âge sur la survie (selon le modèle)")
# plt.show()