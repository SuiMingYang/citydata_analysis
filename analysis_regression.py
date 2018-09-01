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

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score # K折交叉验证模块
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,explained_variance_score #回归问题检验
from sklearn.metrics import accuracy_score, log_loss #分类问题检验
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

#预测
Regressor = [
    linear_model.Ridge(alpha = .5),
    #linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0]),
    linear_model.LinearRegression(),
    #linear_model.Lasso(alpha = 0.1),
    #linear_model.BayesianRidge(),
    #linear_model.LogisticRegression(solver='saga',multi_class='ovr',C=1,penalty='l1',fit_intercept=True,max_iter=1,random_state=42,),
    #QuadraticDiscriminantAnalysis(store_covariances=True),
    #LinearDiscriminantAnalysis(solver="svd", store_covariance=True),
    KernelRidge(alpha=1.0),
    svm.SVR(),
    #linear_model.SGDRegressor(),
    KNeighborsRegressor(n_neighbors=2),
    #GaussianProcessRegressor(kernel=gp_kernel),
    DecisionTreeRegressor(),
    MLPRegressor(),
]
X_column=['cityname','education','experience']
X=df[X_column]
y=df.mean_salary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
log_cols=["Regressor", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
for clf in Regressor:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    print('****Results****')
    print(name)
    train_predictions = clf.predict(X_test)
    evs = explained_variance_score(y_test, train_predictions)
    print("explained_variance_score: {:.4%}".format(evs))
    
    train_predictions = clf.predict(X_test)
    mae = mean_absolute_error(y_test, train_predictions)
    print("mean_absolute_error: {}".format(mae))
    
    log_entry = pd.DataFrame([[name, evs*100, mae]], columns=log_cols)
    log = log.append(log_entry)
'''
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Regressor', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Regressor Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Regressor', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Regressor Log Loss')
plt.show()
'''

'''clf = DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
calculate_test = clf.predict(X_test)
accuracy_count=accuracy_score(y_test, calculate_test)
log_count=log_loss(y_test, calculate_test)


print(accuracy_count)
print(log_count)'''









#分类