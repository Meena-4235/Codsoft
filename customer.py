import warnings  
warnings.simplefilter(action='ignore', category=FutureWarning)  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
pip install seaborn --upgrade  
import seaborn as sns  
sns.set_style('darkgrid')  
from scipy.stats import chi2_contingency  
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict  
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB  
from sklearn.linear_model import LogisticRegression  
df = pd.read_csv('../input/predicting-churn-for-bank-customers/Churn_Modelling.csv')    
print('It contains {} rows and {} columns.'.format(df.shape[0], df.shape[1]))  
df.head()
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df.columns
df.info()
train_df, test_df = train_test_split(df, test_size=0.2, random_staterandom_state=random_state)
train_df.reset_index(drop=True, inplace=True)  
test_df.reset_index(drop=True, inplace=True)  
  
print('Train set: {} rows x {} columns'.format(train_df.shape[0],  
                                               train_df.shape[1]))  
print(' Test set: {} rows x {} columns'.format(test_df.shape[0],  
                                               test_df.shape[1]))
fig, ax = plt.subplots(figsize=(6, 6))
sns.countplot(x='Exited', data=train_df, palette=colors, axax=ax)
for index, value in enumerate(train_df['Exited'].value_counts()):
    label = '{}%'.format(round((value / train_df['Exited'].shape[0]) * 100, 2))  
    ax.annotate(label,  
                xy=(index, value + 250),  
                ha='center',  
                va='center',  
                color=colors[index],  
                fontweight='bold',  
                size=font_size + 4)
ax.set_xticklabels(['Retained', 'Churned'])  
ax.set_xlabel('Status')  
ax.set_ylabel('Count')  
ax.set_ylim([0, 7000]);
continuous = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categorical = ['Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']   
print('Continuous: ', ', '.join(continuous))  
print('Categorical: ', ', '.join(categorical))  
train_df[continuous].hist(figsize=(14, 12),  
                          bins=20,  
                          layout=(2, 2),  
                          color='steelblue',  
                          edgecolor='firebrick',  
                          linewidth=1.0);
fig, ax = plt.subplots(figsize=(8, 7))  
sns.heatmap(train_df[continuous].corr(),  
            annot=True,  
            annot_kws={'fontsize': 16},  
            cmap='Blues',  
            axax=ax)    
ax.tick_params(axis='x', rotation=45)  
ax.tick_params(axis='y', rotation=360);
df_churned = train_df[train_df['Exited'] == 1]  
df_retained = train_df[train_df['Exited'] == 0]  
plot_continuous('Age')
plot_continuous('CreditScore')
df_cat = train_df[categorical]  
fig, ax = plt.subplots(2, 3, figsize=(12, 8))  
for index, column in enumerate(df_cat.columns):  
    plt.subplot(2, 3, index + 1)  
    sns.countplot(x=column, data=train_df, palette=colors_cat)  
    plt.ylabel('Count')  
     if (column == 'HasCrCard' or column == 'IsActiveMember'):
           plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout();
plot_categorical('Geography')
chi2_array, p_array = [], []  
for column in categorical:  
  
    crosstab = pd.crosstab(train_df[column], train_df['Exited'])  
    chi2, p, dof, expected = chi2_contingency(crosstab)  
    chi2_array.append(chi2)  
    p_array.append(p)  
  
df_chi = pd.DataFrame({  
    'Variable': categorical,  
    'Chi-square': chi2_array,  
    'p-value': p_array  
})  
df_chi.sort_values(by='Chi-square', ascending=False)
train_df['Gender'] = LabelEncoder().fit_transform(train_df['Gender'])
train_df['Geography'] = train_df['Geography'].map({  
    'Germany': 1,  
    'Spain': 0,  
    'France': 0  
})
scaler = StandardScaler()  
  
scl_columns = ['CreditScore', 'Age', 'Balance']  
train_df[scl_columns] = scaler.fit_transform(train_df[scl_columns])
y_train = train_df['Exited']  
X_train = train_df.drop('Exited', 1)
over = SMOTE(sampling_strategy='auto', random_staterandom_state=random_state)
X_train, y_train = over.fit_resample(X_train, y_train)    
y_train.value_counts()
clf_list = [('Gaussian Naive Bayes', GaussianNB()),  
            ('Logistic Regression', LogisticRegression(random_staterandom_state=random_state))]  
  
cv_base_mean, cv_std = [], []  
for clf in clf_list:  
  
    cv = cross_val_score(estimator=clf[1],  
                         X=X_train,  
                         y=y_train,  
                         scoring=scoring_metric,  
                         cv=5,  
                         n_jobs=-1)  
  
    cv_base_mean.append(cv.mean())  
    cv_std.append(cv.std())  
  
print('Baseline Models (Recall):')  
  
for i in range(len(clf_list)):  
    print('   {}: {}'.format(clf_list[i][0], np.round(cv_base_mean[i], 2)))
    lr = LogisticRegression(random_staterandom_state=random_state)
    param_grid = {  
    'max_iter': [100],  
    'penalty': ['l1', 'l2'],  
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],  
    'solver': ['lbfgs', 'liblinear']  
}  
  
lr_clf = GridSearchCV(estimator=lr,  
                      param_gridparam_grid=param_grid,  
                      scoring=scoring_metric,  
                      cv=5,  
                      verbose=False,  
                      n_jobs=-1)  
  
best_lr_clf = lr_clf.fit(X_train, y_train)  
clf_performance(best_lr_clf, 'Logistic Regression', 'LR')  
    
    
