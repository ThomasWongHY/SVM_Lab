import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV

# Load & check the data
data_HoYin = pd.read_csv("breast_cancer.csv")
print(data_HoYin.info())
print(data_HoYin.isin(["?"]).sum())
# print(data_HoYin.isnull().sum())
print(data_HoYin.describe())

data_HoYin = data_HoYin.replace('?', np.nan).astype({'bare': 'float'})
data_HoYin.fillna(data_HoYin.median(), inplace=True)
data_HoYin.drop(columns=['ID'], inplace=True)

f,ax=plt.subplots(figsize = (7,7))
sns.heatmap(data_HoYin.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

data_HoYin.hist(bins=50, figsize=(20,15))

y = data_HoYin['class'].copy()
X = data_HoYin.drop(columns=['class'])
for index,columns in enumerate(X):
    plt.figure(index, figsize=(10,10))
    sns.stripplot(x='class', y=columns, data=data_HoYin, jitter=True, hue="class", palette="deep")
    plt.title('Class vs ' + str(columns))
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

model = ['linear', 'rbf', 'poly', 'sigmoid']
for i in model:
    if i == 'linear':
        clf_linear_HoYin = SVC(kernel=i, C = 0.1)
    else: 
        clf_linear_HoYin = SVC(kernel=i)
    
    clf_linear_HoYin.fit(X_train, y_train)
    y_train_pred = clf_linear_HoYin.predict(X_train)
    
    print(f"Accuracy of Training Data in {i.upper()} Model: ")
    print(accuracy_score(y_train, y_train_pred))
    
    clf_linear_HoYin.fit(X_test, y_test)
    y_test_pred = clf_linear_HoYin.predict(X_test)
    
    print(f"Accuracy of Testing Data in {i.upper()} Model: ")
    print(accuracy_score(y_test, y_test_pred))

    print(f"Accuracy Matrix for {i.upper()} Model: ")
    print(confusion_matrix(y_test, y_test_pred))

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data_HoYin_df2 = pd.read_csv("breast_cancer.csv")
data_HoYin_df2 = data_HoYin_df2.replace('?', np.nan).astype({'bare': 'float'})
data_HoYin_df2.drop(columns=['ID'], inplace=True)

y = data_HoYin_df2['class'].copy()
X = data_HoYin_df2.drop(columns=['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

num_pipe_HoYin = Pipeline([('imputer', SimpleImputer(strategy='median', missing_values=np.nan)), 
                          ('scaler', StandardScaler())])
print(num_pipe_HoYin)

pipe_svm_HoYin = Pipeline(
    [
        ('transformer', num_pipe_HoYin),
        ('svc', SVC(random_state=44))
    ]
)
print(pipe_svm_HoYin)

param_grid_svm = {
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    'svc__degree': [2, 3]
}
print(param_grid_svm)

grid_search_HoYin = GridSearchCV(estimator=pipe_svm_HoYin, param_grid=param_grid_svm, scoring='accuracy', refit=True, verbose=3)
grid_search_HoYin.fit(X_train, y_train)
print(grid_search_HoYin)

print("Best parameter:")
print(grid_search_HoYin.best_params_)

print("Best estimator:")
print(grid_search_HoYin.best_estimator_)

y_pred_svm = grid_search_HoYin.predict(X_test)
# print(y_pred_svm)
test_HoYin = grid_search_HoYin.fit(X_test, y_test)
print(test_HoYin)

print("Accuracy: ")
print(accuracy_score(y_test, y_pred_svm))

final_model = grid_search_HoYin.best_estimator_
grid_search_HoYin_final = GridSearchCV(estimator=final_model, param_grid=param_grid_svm, scoring='accuracy', refit=True, verbose=3)
y_pred_svm_final = grid_search_HoYin.predict(X_test)
# print(y_pred_svm)
test_HoYin_final = grid_search_HoYin_final.fit(X_test, y_test)
print(test_HoYin_final)

print("Best Model Accuracy in Training Data: ")
y_pred_svm_train = grid_search_HoYin.predict(X_train)
print(accuracy_score(y_train, y_pred_svm_train))
print("Best Model Accuracy in Testing Data: ")
print(accuracy_score(y_test, y_pred_svm_final))

best_model_HoYin_final = grid_search_HoYin_final.best_estimator_
print(best_model_HoYin_final)

import joblib

model_file = 'best_model_HoYin_final.sav'
joblib.dump(best_model_HoYin_final, model_file)

pipe_file = 'full_pipeline_final.sav'
joblib.dump(pipe_svm_HoYin, pipe_file)