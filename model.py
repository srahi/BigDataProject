import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize

df= pd.read_csv('dataset.csv',header=None, names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"], na_values=['?'])
df = df.replace('?','NaN')

df['ca']=df['ca'].astype(float)
df['thal']=df['thal'].astype(float)

imp = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
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



from scipy.stats import pearsonr
print(pearsonr(df_m['age'],df_m['num']))
print(pearsonr(df_m['sex'],df_m['num']))
print(pearsonr(df_m['cp'],df_m['num']))
print(pearsonr(df_m['chol'],df_m['num']))
print(pearsonr(df_m['trestbps'],df_m['num']))
print(pearsonr(df_m['fbs'],df_m['num']))
print(pearsonr(df_m['restecg'],df_m['num']))
print(pearsonr(df_m['thalach'],df_m['num']))
print(pearsonr(df_m['exang'],df_m['num']))
print(pearsonr(df_m['oldpeak'],df_m['num']))
print(pearsonr(df_m['slope'],df_m['num']))
print(pearsonr(df_m['ca'],df_m['num']))
print(pearsonr(df_m['thal'],df_m['num']))


df_m.info()
