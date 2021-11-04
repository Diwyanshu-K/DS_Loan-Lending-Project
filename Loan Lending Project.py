import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

loans = pd.read_csv('loan_data.csv')

loans.head()
loans.info
loans.describe()

# Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.

# plt.figure(figsize=(10,6))
# loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=1')
# loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red', bins=30,label='Credit.Policy=0')
# plt.legend()
# plt.xlabel('FICO')


# plt.figure(figsize=(10,6))
# loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='not.fully.paid=1')
# loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='not.fully.paid=0')
# plt.legend()
# plt.xlabel('FICO')


# Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.

# plt.figure(figsize=(11, 7))
# sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette='Set1')

# Let's see the trend between FICO score and interest rate. Recreate the following jointplot

# sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')

# Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy.
# Check the documentation for lmplot()
# if you can't figure out how to separate it into columns

# plt.figure(figsize=(11,7))
# sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',   col='not.fully.paid',palette='Set1')

loans.info()





cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=600)

rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))










plt.show()
