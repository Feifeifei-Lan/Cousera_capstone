import pandas as pd
import numpy as np

cereal_df = pd.read_csv("Data-Collisions.csv", low_memory=False)

Feature = cereal_df[['LIGHTCOND','ROADCOND','WEATHER','JUNCTIONTYPE', 'COLLISIONTYPE', 'ADDRTYPE', 'SEVERITYCODE']]


Feature["LIGHTCOND"] = Feature["LIGHTCOND"].astype('category')
Feature["LIGHTCOND_cat"] = Feature["LIGHTCOND"].cat.codes
Feature["ROADCOND"] = Feature["ROADCOND"].astype('category')
Feature["ROADCOND_cat"] = Feature["ROADCOND"].cat.codes
Feature["WEATHER"] = Feature["WEATHER"].astype('category')
Feature["WEATHER_cat"] = Feature["WEATHER"].cat.codes
Feature["JUNCTIONTYPE"] = Feature["JUNCTIONTYPE"].astype('category')
Feature["JUNCTIONTYPE_cat"] = Feature["JUNCTIONTYPE"].cat.codes
Feature["COLLISIONTYPE"] = Feature["COLLISIONTYPE"].astype('category')
Feature["COLLISIONTYPE_cat"] = Feature["COLLISIONTYPE"].cat.codes
Feature["ADDRTYPE"] = Feature["ADDRTYPE"].astype('category')
Feature["ADDRTYPE_cat"] = Feature["ADDRTYPE"].cat.codes


X = Feature[['LIGHTCOND_cat','ROADCOND_cat','WEATHER_cat','JUNCTIONTYPE_cat', 'COLLISIONTYPE_cat', 'ADDRTYPE_cat']]
y = Feature["SEVERITYCODE"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

import matplotlib.pyplot as plt
corr = Feature.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(Feature.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(Feature.columns)
ax.set_yticklabels(Feature.columns)
plt.show()

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


gnb = GaussianNB()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)

print("f1 score", metrics.f1_score(y_test, predictions, average = "weighted"))
print("matrics accuracy score:", metrics.accuracy_score(y_test, predictions))
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

