import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

filename = 'Python_city.csv'
df = pd.read_csv(filename, encoding='utf-8')    

cityname_mapping = {
           '北京  ': 1,
           '杭州  ': 2,
           '武汉  ': 3,
           '成都  ': 4,
           '长沙  ': 5}
df['cityname'] = df['cityname'].map(cityname_mapping)

experience_mapping = {
           '5-10年': 6,
           '3-5年': 5,
           '1-3年': 4,
           '1年以内': 3,
           '应届生': 2,
           '经验不限': 1}
df['experience'] = df['experience'].map(experience_mapping)

company_size_mapping = {
           '10000人以上': 6,
           '1000-9999人': 5,
           '500-999人': 4,
           '100-499人': 3,
           '20-99人': 2,
           '0-20人':1}
df['company_size'] = df['company_size'].map(company_size_mapping)

education_mapping = {
           '博士': 5,
           '硕士': 4,
           '本科': 3,
           '大专': 2,
           '学历不限': 1}
df['education'] = df['education'].map(education_mapping)

df['low_salary']=df['salary']
df['high_salary']=df['salary']
df['mean_salary']=df['salary']
for temp in df['salary']:
    lowval=temp.split('-')[0].replace('k','')
    newval=temp.split('-')[1].replace('k','')
    df['low_salary'][df['low_salary']==temp]=lowval
    df['high_salary'][df['high_salary']==temp]=newval
    df['mean_salary'][df['mean_salary']==temp]=(float(newval)+float(lowval))/2

df.dropna(axis=0, how='any')
df.fillna(value=0)
#print(df)

#print(df.groupby('industry').count())
#print(df.groupby('cityname').count())
'''
filename = 'pandas_output.csv'
#df.to_csv(filename)
df.to_csv(filename, index=None, encoding='GBK')
'''

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score # K折交叉验证模块
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

X_column=['mean_salary','education','experience']
X=df[X_column]
y=df.cityname

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)