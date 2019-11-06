import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize

df= pd.read_csv('dataset.csv',header=None, names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"], na_values=['?'])
df = df.replace('?','NaN')

df['ca']=df['ca'].astype(float)
df['thal']=df['thal'].astype(float)

df_m=df.fillna(df.mean())


df_m['ca'] = pd.to_numeric(df_m['ca'], errors='coerce')
df_m[['age', 'sex', 'fbs', 'exang', 'ca']] = df_m[['age', 'sex', 'fbs', 'exang', 'ca']].astype(int)
df_m[['trestbps', 'chol', 'thalach', 'oldpeak']] = df_m[['trestbps', 'chol', 'thalach', 'oldpeak']].astype(float)
df_m['num'].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)
import seaborn as sns
sns.boxplot(x=df_m['ca'])


df_m['Ca']=winsorize(df_m['ca'],limits=[0.0,0.25])
df_m.drop("ca", axis=1, inplace=True) 
sns.boxplot(x=df_m['Ca'])

sns.boxplot(x=df_m['chol'])
df_m['Chol']=winsorize(df_m['chol'],limits=[0.0,0.25])
sns.boxplot(x=df_m['Chol'])
df_m.drop("chol", axis=1, inplace=True) 


sns.boxplot(x=df_m['oldpeak'])
df_m['Oldpeak']=winsorize(df_m['oldpeak'],limits=[0.03,0.05])
sns.boxplot(x=df_m['Oldpeak'])
df_m.drop("oldpeak", axis=1, inplace=True) 


#Box Plot
sns.boxplot(x=df_m['trestbps'])
# Winsorization
df_m['Trestbps']=winsorize(df_m['trestbps'],limits=[0.0,0.25])
sns.boxplot(x=df_m['Trestbps'])
df_m.drop("trestbps", axis=1, inplace=True) 

sns.boxplot(x=df_m['thal'])
df_m['Thal']=winsorize(df_m['thal'],limits=[0.03,0.05])
sns.boxplot(x=df_m['Thal'])
df_m.drop("thal", axis=1, inplace=True) 

sns.boxplot(x=df_m['thalach'])
df_m['Thalach']=winsorize(df_m['thalach'],limits=[0.03,0.05])
sns.boxplot(x=df_m['Thalach'])
df_m.drop("thalach", axis=1, inplace=True) 



import matplotlib.pyplot as plt

heat_map = sns.heatmap(df_m.corr())

plt.show()

#### Decision Tree Classifier ####


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn import metrics

feature_cols = ["age", "sex", "cp", "Trestbps", "Chol", "fbs", "restecg", "Thalach", "exang", "Oldpeak", "slope", "Ca", "Thal"]
X = df_m[feature_cols] 
y = df_m.num 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

## Accuracy: 0.7912087912087912







#### Logistic Regression ####

from sklearn.linear_model import LogisticRegression 
feature_cols = ["age", "sex", "cp", "Trestbps", "Chol", "fbs", "restecg", "Thalach", "exang", "Oldpeak", "slope", "Ca", "Thal"]
x = df_m[feature_cols] 
y = df_m.num 

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split( 
        x, y, test_size = 0.25, random_state = 0) 

from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
xtrain = sc_x.fit_transform(xtrain) 
xtest = sc_x.transform(xtest) 

#print (xtrain[0:10, :]) 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest) 

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred)   

#print ("Confusion Matrix : \n", cm) 
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytest, y_pred))

## Accuracy :  0.8289473684210527








#### Random Forest ####


feature_cols = ["age", "sex", "cp", "Trestbps", "Chol", "fbs", "restecg", "Thalach", "exang", "Oldpeak", "slope", "Ca", "Thal"]
X = df_m[feature_cols] 
y = df_m.num 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

## Accuracy: 0.7582417582417582










#### Naive Bayes####
# Import LabelEncoder
from sklearn import preprocessing

feature_cols = ["age", "sex", "cp", "Trestbps", "Chol", "fbs", "restecg", "Thalach", "exang", "Oldpeak", "slope", "Ca", "Thal"]
X = df_m[feature_cols] 
y = df_m.num 

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=109) # 70% training and 30% test
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train,y_train)

#Predict Output
y_pred= model.predict(X_test) 
#print("Predicted Value:", y_pred)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


## Accuracy: 0.7802197802197802


