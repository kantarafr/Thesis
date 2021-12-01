# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:32:17 2021

@author: wikto
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:07:08 2021

@author: kanta
"""
#Loading the packages that will be needed
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import statsmodels.formula.api as smf
import statsmodels.formula.api as smf

#CLEANING ~~~~~~~~~~~~

#Reading the data
df = pd.read_csv('ESS5_with_NaNs_v2.csv' )

#Removing columns that are not needed
data = df.drop(['name', 'essround', 'edition', 'proddate', 'idno', 'ctzshipb', 
                'cntbrthb', 'fbrncnta', 'mbrncnta', 'stcbg2t', 'nacer2', 'mnactic' ,
                'tmprs', 'marstie', 'maritalb', 'eisced', 'emplnop', 'bstln5y', 'dngrefp',
                'iscoco',   'iscocop', 'lnghom1', 'lnghom2', 'plcarcr', 'yrbrn2',
                'dngnapp', 'dngdkp', 'dngothp', 'hswrkp', 'cmsrvp', 'rtrdp', 'emprm14',
                'dsbldp', 'uemplip', 'uemplap', 'edctnp', 'pdwrkp', 'eiscedf',
                'uemp3y', 'dngnap', 'dngdk', 'dngoth', 'hswrk', 'cmsrv', 'rtrd', 
                'rsnlvem', 'occf14b', 'wraccrp', 'vote', 'wrkctra', 'insclwr', 'freehms',
                'dsbld', 'uempli', 'uempla', 'edctnp', 'pdwrk', 'dngoth', 'gdsprt', 'wevdct',
                'jdgcbrb', 'brncntr', 'contplt', 'wrkprty', 'wrkorg', 'badge',
                'sgnptit', 'pbldmn', 'bctprd', 'clsprty', 'prtdgcl', 'prtyban',
                'scnsenv', 'imsmetn', 'imdfetn', 'impcntr', 'inmdisc', 'crmvct',
                'aesfdrk', 'brghmwr', 'brghmef', 'crvctwr', 'crvctef', 'ctzcntr', 'brncntr', 'livecnta',
                'facntr', 'mocntr', 'bystlwr', 'trfowr', 'insclct', 'bystlct', 'trfoct',
                'plccont', 'plcstf', 'plcvcrp', 'plcvcrc', 'plccbrg', 'plcarcr', 
                'bplcdc', 'plciplt', 'wraccrc', 'hrshsnta', 'dbctvrd', 'caplcst',
                'widprsn', 'wevdct', 'bstln5y', 'troff5y', 'lvgptnea', 'dvrcdeva',
                'icpart3', 'icpart2', 'iccohbt', 'chldhhe', 'fxltph', 'mainact',
                'crpdwk', 'pdjobev', 'jbtmppm' , 'estsz', 'brwmny', 'edulvlpb', 'eiscedp',
                'mnactp', 'crpdwkp', 'emplnop', 'jbspvp', 'njbspvp', 'wkhtotp',
                'emplnof', 'jbspvf', 'emplnom', 'jbspvm', 'occm14b', 'atncrse', 'wkhtot',
                'edul12m', 'fltlnla', 'jbtsktm', 'yrcremp', 'scrsefw', 'bsmw', 'uemp3m', 
                'ppwwkp', 'icnopfma', 'payprda', 'linwk3y', 'rdpay3y', 'wkshr3y', 'edctn', 'dngna',
                'lscjb3y', 'orgfd3y', 'npemp3y', 'icptn', 'pwkhsch', 'puemp3y', 'edulvlb', 
                'hwwkhs', 'phwwkhs', 'dsgrmnya', 'wkengtp', 'wkovtmp', 'ptnwkwe',
                'icmnart', 'rtryr', 'wntrtr', 'icago45', 'plnchld', 'edulvlmb', 'gndr2',
                'dngref', 'dngrefp', 'icmnact', 'facntr', 'eiscedm', 'wrkac6m', 'icagu70a',
                'flsin5y', 'cntry', 'icmnart', 'emprf14', 'yrbrn', 
                'edulvlfb'], axis = 1, inplace = False)

data.drop(df.columns[df.columns.str.startswith(('edlvd', 'prtvt', 'edlvfd', 
    'edlvmd', 'edlvpd', 'edupgb', 'prtcl', 'prtmb', 'rlgdn', 'rshipa', 'dscr', 'maristie'))], 
          axis = 1, inplace = True)

#OUTLIERS DROPPING in the dependent cateogry

data = data.iloc[(data['agertr'] < 81).values, :]
data = data.iloc[(data['agea'] < 71).values, :]

# Binarize the intentions retirement age


data[data['yrspdwka'] > 100] = np.nan

data[data['wkhct'] > 100] = np.nan

data['agertr_bin'] = [0 if x > 64 else 1 for x in data['agertr']]

#Looking at numerical columns vs categorical columns
num_cols = data.drop([ 'agertr_bin'], axis = 1).select_dtypes('number').columns
cat_cols = data.drop([ 'agertr_bin'], axis = 1).select_dtypes('object').columns

print(f"Number of numerical columns: {len(num_cols)}")
print(f"Number of categorical columns: {len(cat_cols)}")


#Droping columns that have more than 20pr missing values
cols_high_nans = num_cols[((data[num_cols].isna().sum() / len(data)) > 0.2).values]
data_cleaned = data.drop(cols_high_nans, axis = 1)

#Looking at the columns with high correlations with the dependent variable
cor_with_dep = data.corr()['agertr_bin']
cols_high_cor_with_dep = cor_with_dep.index[((cor_with_dep > 0.1) | (cor_with_dep < -0.1)).values]

#Columns with nas more than 20 that have also high correlation with the dependent variable

high_cor_high_na = [col for col in cols_high_cor_with_dep if col in cols_high_nans]

#Columns that have less than 40% nas but more than 20%

cols_nans_less_than_40 = num_cols[((data[num_cols].isna().sum() / 
            len(data)) > 0.2).values & ((data[num_cols].isna().sum() / 
            len(data)) < 0.4).values]

#Keeping the columns that have high correlation but nas between 20% and 40%
cols_high_nas_tokeep = [col for col in cols_high_cor_with_dep if col in cols_nans_less_than_40]

data_cleaned = data_cleaned.join(data[cols_high_nas_tokeep])






#EDA ~~~~~~~~~~


# ~~~~~~~  GRAPH 1 Distribution of agertr  ~~~~~~~
#Nice graphs theme
sns.set_theme()
sns.set_style("whitegrid")
data.head()

agertr_dist_fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, 
                        gridspec_kw = {"height_ratios": (.15, .85)})
plt.subplots_adjust(hspace=0.025)

sns.boxplot(data_cleaned['agertr'], ax = ax_box, palette = ['gray'])
ax_box.set(xlabel = '')

sns.histplot(x = 'agertr', data = data_cleaned, hue = 'gndr', fill = True, legend = True,
            hue_order = [1, 2], palette = ['red', 'blue'], stat = 'count', kde = True,
            multiple = 'stack', edgecolor = 'black')
plt.legend(title = 'Gender', loc = 'upper right', labels = ['Female', 'Male'])
plt.xlabel("Retirement age intention")
ax_box.set_title('Distribution of retirement age intention')
plt.savefig('agertr_dist_fig.png', dpi = 900)




# ~~~~~~~  GRAPH 2 Distribution of agea  ~~~~~~~
agea_dist_fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, 
            gridspec_kw = {"height_ratios": (.15, .85)})
plt.subplots_adjust(hspace = 0.025)

sns.boxplot(data_cleaned['agea'], ax = ax_box, palette = ['gray'], )
ax_box.set(xlabel = '')
sns.histplot(x = 'agea', data = data_cleaned, hue = 'gndr', fill = True,
            hue_order = [1, 2], palette = ['red', 'blue'], stat ='count', bins = 29,
            kde = True)

plt.legend(title = 'Gender', loc = 'upper right', labels = ['Female', 'Male'])
plt.xlabel("Age")
ax_box.set_title('Distribution of age')
plt.savefig('agea_dist_fig2.pdf', dpi = 900)


# ~~~~~~ GRAPH 3 Age vs retirement age intention ~~~~~~~~~~~
joint_fig = sns.jointplot(x = 'agea', y = 'agertr', data = data_cleaned, kind = 'hist',
                          joint_kws = dict(bins = 16, edgecolor = "white"),
                          marginal_kws = dict(bins = 28))
joint_fig.set_axis_labels(xlabel = "Age", ylabel = "Retirment age intention")
plt.savefig('joint_fig.pdf', dpi = 900)



# ~~~~~~ GRAPH 4 Distribution of the intentions to retire age binarized

fig = plt.subplots(figsize = (10, 5))
plt.pie(data_cleaned.agertr_bin.value_counts(), labels = ['Intentions to retire earlier than 65', 
                        'Intentions to retire later or equal to 65'], shadow = True,
                         autopct='%1.0f%%')
plt.title("Distribution of the binarized intentions to retire variable", 
          fontdict = {'fontsize':20})
fig[0].savefig('pie_agertr.pdf')

#Checking the min and max from each columns except the categorical ones

min_max_mean_df = data_cleaned.loc[:, ~data_cleaned.columns.isin(cat_cols)].describe().transpose()
min_max_mean_df.to_excel('Min_Max_Mean.xlsx')

print(min_max_mean_df)

#Dropping the agertr category as it has been binarized and it won't be needed anymore
data_cleaned.drop('agertr', axis = 1, inplace = True)


#One - hot encoding
#Getting dummies for the columns that although in the dataset are numerical, in the questionnaire
#the answers are categorical

df1_ohe = pd.get_dummies(data = data_cleaned, columns = [
                                   'chldhm', 'domicil', 
                                    'emplrel', 'blgetmg', 'mmbprty','rlgblg',
                                   'tporgwk',   'gndr', 'icpart1',
                                   'hincfel', 'hincsrca', 'jbspv', 'mbtru'], drop_first=True,
                                      dummy_na = True)

nan_df = df1_ohe.loc[:, df1_ohe.columns.str.endswith("_nan")]

#Specifying that nans in the columns after the dummy coding in order to fill them
#in with imputation
pattern = "^([^_]*)_"
regex = re.compile(pattern)

for index in df1_ohe.index:
    for col_nan in nan_df.columns:
        if df1_ohe.loc[index,col_nan] == 1:
            col_id = regex.search(col_nan).group(1)
            targets = df1_ohe.columns[df1_ohe.columns.str.startswith(col_id+'_')]
            df1_ohe.loc[index, targets] = np.nan
            
df1_ohe.drop(df1_ohe.columns[df1_ohe.columns.str.endswith('_nan')], axis=1, inplace=True)

data_cleaned = df1_ohe

data_cleaned.drop(data_cleaned[data_cleaned['gndr_2.0'].isna()].index, inplace=True) #Only one record has na for gender and it's being dropped now


#Checking for logistic regression assumptions linearity with the logodds, creating the plots
no_dummies = pd.DataFrame()

# For plotting/checking assumptions


gpa = sns.regplot(x= 'stflife', y= 'agertr_bin', data= data_cleaned, logistic= True).set_title("GPA Log Odds Linear Plot")
gpa.figure.savefig("gpa log lin.png")

not_binary_cols = []
for col in data_cleaned.columns:
    if len(data_cleaned[col][data_cleaned[col].notnull()].unique()) > 2: # if feature has more than 2 non-nan entires
        not_binary_cols.append(col) 

fsize = 8
plt.rcParams.update({'font.size': fsize})
sns.set_context(rc={'axes.labelsize': fsize,'axes.titlesize': fsize})
fig1, axes1 = plt.subplots(6,6, sharey=True)
fig2, axes2 = plt.subplots(6,6, sharey=True)
fig3, axes3 = plt.subplots(6,6, sharey=True)

#fig2, axes2 = plt.subplots(7,7)


i=0
for axes in [axes1, axes2, axes3]:
    for axy in axes:
        for j,axx in enumerate(axy):
            if i<len(not_binary_cols):
                axx.tick_params(axis='x', labelsize=fsize)
                axx.tick_params(axis='y', labelsize=fsize)
                sns.regplot(x = not_binary_cols[i], y= 'agertr_bin', data= data_cleaned, logistic= True, ax=axx)
                i+=1
            if j>0:         #if plot is not in first row remove ylabel
                axx.set(ylabel=None)
            
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt') #to make figures pop out

#Checking for multicollinearity
correlations= data_cleaned.corr()
print(correlations)

#MISSING VALUES IMPUTATION ~~~~~~~~~~~~

#Approach inspired by https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py

#Spliting into train, validation and test set
X_train, X_test, y_train, y_test = train_test_split(data_cleaned.drop('agertr_bin', axis = 1), 
                                                      data_cleaned['agertr_bin'], 
                                                      test_size = 0.1, 
                                                      random_state = 0, 
                                                      stratify = data_cleaned['agertr_bin'])

X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                      y_train, 
                                                      test_size = 0.2, 
                                                      random_state = 0,
                                                      stratify = y_train)


X_train.shape, X_test.shape, X_val.shape

#Looking at numerical columns vs categorical columns
num_cols_cleaned = data_cleaned.drop([ 'agertr_bin'], axis = 1).select_dtypes('number').columns
cat_cols_cleaned = data_cleaned.drop([ 'agertr_bin'], axis = 1).select_dtypes('object').columns

#Number of numerical and categorical columns that have missing values
num_cols_with_na = num_cols_cleaned[X_train[num_cols_cleaned].isna().mean() > 0]
cat_cols_with_na = cat_cols_cleaned[X_train[cat_cols_cleaned].isna().mean() > 0]

#Number of numerical and categorical columns that do not have missing values
num_cols_no_na = num_cols_cleaned[~(X_train[num_cols_cleaned].isna().mean() > 0)]
cat_cols_no_na = cat_cols_cleaned[~(X_train[cat_cols_cleaned].isna().mean() > 0)]

print(f"*** numerical columns that have NaN's ({len(num_cols_with_na)}): \n{num_cols_with_na}\n\n")
print(f"*** categorical columns that have NaN's ({len(cat_cols_with_na)}): \n{cat_cols_with_na}\n\n")
print(f"*** numerical columns that do not have NaN's ({len(num_cols_no_na)}): \n{num_cols_no_na}\n\n")
print(f"*** categorical columns that do not have NaN's ({len(cat_cols_no_na)}): \n{cat_cols_no_na}")


#Percentage of missing values in numerical columns
X_train[num_cols_with_na].isna().mean().sort_values(ascending = False)



#~~~~~ NUMERICAL MISSING VALUES IMPUTATION ~~~~~~~~

#Replacing na for the numerical features
imputer = IterativeImputer(max_iter = 10, random_state = 0)

# # fit the imputer on X_train. pass only numeric columns.
imputer.fit(X_train[num_cols_with_na])

#Transform the data using the fitted imputer
X_train_impute_num = imputer.transform(X_train[num_cols_with_na])
X_val_impute_num = imputer.transform(X_val[num_cols_with_na])
X_test_impute_num = imputer.transform(X_test[num_cols_with_na])

# put the output into DataFrame. remember to pass columns used in fit/transform
X_train_impute_num = pd.DataFrame(X_train_impute_num, columns = num_cols_with_na)
X_val_impute_num = pd.DataFrame(X_val_impute_num, columns = num_cols_with_na)
X_test_impute_num = pd.DataFrame(X_test_impute_num, columns = num_cols_with_na)


#Dropping the numerical columns and the columns that are imputed from the dataset in order to join the imputed ones
#and create the numerical features dataset
cols_no_na_train = X_train.drop(columns = num_cols_with_na, axis = 1)
cols_no_na_val = X_val.drop(columns = num_cols_with_na, axis = 1)
cols_no_na_test = X_test.drop(columns = num_cols_with_na, axis = 1)

X_train = X_train_impute_num.join(cols_no_na_train.reset_index(drop = True))
X_val = X_val_impute_num.join(cols_no_na_val.reset_index(drop = True))
X_test = X_test_impute_num.join(cols_no_na_test.reset_index(drop = True))

#Plot for the feature distributions

zero_fifthy =  data_cleaned.iloc[:, :50]
figure_numpy = zero_fifthy.hist()
plt.tight_layout()
figure = plt.gcf()
figure.savefig('AppendixDistributionGraph.pdf')
plt.show()

fifthy_hundred =  data_cleaned.iloc[:, 51:100]
figure2 = fifthy_hundred.hist()
plt.show()

hundred_hundredfifthy =  data_cleaned.iloc[:, 101:]
figure3 = hundred_hundredfifthy.hist()
plt.show()



# FEATURE TRANSFORMATION BEFORE FEATURE SELECTION AND RUNNING THE MODELS~~~~~~~~



#Rescaling the data
scaler = MinMaxScaler()
X_train_rescaled = scaler.fit_transform(X_train)
X_val_rescaled = scaler.fit_transform(X_val)
X_test_rescaled = scaler.fit_transform(X_test)

X_train_rescaled.shape , X_val_rescaled.shape , X_test_rescaled.shape

#Saving the data into a train dataset, test dataset and validation dataset.

pd.DataFrame(X_train_rescaled, columns = X_train.columns).to_csv('X_train_rescaled.csv', index=False)
pd.DataFrame(X_val_rescaled, columns = X_val.columns).to_csv('X_valid_rescaled.csv', index=False)
pd.DataFrame(X_test_rescaled, columns = X_test.columns).to_csv('X_test_rescaled.csv', index=False)


y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)



X_train_rescaled = pd.read_csv('X_train_rescaled.csv' )
X_test_rescaled = pd.read_csv('X_test_rescaled.csv' )
X_val_rescaled = pd.read_csv('X_valid_rescaled.csv' )


y_test = pd.read_csv('y_test.csv' )
y_val = pd.read_csv('y_val.csv' )
y_train = pd.read_csv('y_train.csv' )



# RECURSIVE FEATURE ELIMINATION WITH RANDOM FOREST 

#Defining the model
#Training the recursive feature elimination model  to select 10/20/30/40/50/60/70/80/90 features for the analysis


rfe2 = RFE(estimator=SVC(class_weight='balanced', kernel='linear'), n_features_to_select=10, step = 10)
selector2 = rfe2.fit(X_train_rescaled, y_train.values.ravel())
rfe3 = RFE(estimator=SVC(class_weight='balanced', kernel='linear'), n_features_to_select=20, step = 10)
selector3 = rfe3.fit(X_train_rescaled, y_train.values.ravel())
rfe1 = RFE(estimator=SVC(class_weight='balanced', kernel='linear'), n_features_to_select=40, step = 10)
selector1 = rfe1.fit(X_train_rescaled, y_train.values.ravel())
rfe4 = RFE(estimator=SVC(class_weight='balanced', kernel='linear'), n_features_to_select=30, step = 10)
selector4 = rfe4.fit(X_train_rescaled, y_train.values.ravel())
rfe5 = RFE(estimator=SVC(class_weight='balanced', kernel='linear'), n_features_to_select=50, step = 10)
selector5 = rfe5.fit(X_train_rescaled, y_train.values.ravel())
rfe6 = RFE(estimator=SVC(class_weight='balanced', kernel='linear'), n_features_to_select=60, step = 10)
selector6 = rfe6.fit(X_train_rescaled, y_train.values.ravel())
rfe7 = RFE(estimator=SVC(class_weight='balanced', kernel='linear'), n_features_to_select=70, step = 10)
selector7 = rfe7.fit(X_train_rescaled, y_train.values.ravel())
rfe8 = RFE(estimator=SVC(class_weight='balanced', kernel='linear'), n_features_to_select=80, step = 10)
selector8 = rfe8.fit(X_train_rescaled, y_train.values.ravel())
rfe9 = RFE(estimator=SVC(class_weight='balanced', kernel='linear'), n_features_to_select=90, step = 10)
selector9 = rfe9.fit(X_train_rescaled, y_train.values.ravel())


#Saving the different training sets with different number of features
X_train_rfe1 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector1.support_]), axis=1)
X_train_rfe2 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector2.support_]), axis=1)
X_train_rfe3 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector3.support_]), axis=1)
X_train_rfe4 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector4.support_]), axis=1)
X_train_rfe5 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector5.support_]), axis=1)
X_train_rfe6 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector6.support_]), axis=1)
X_train_rfe7 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector7.support_]), axis=1)
X_train_rfe8 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector8.support_]), axis=1)
X_train_rfe9 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector9.support_]), axis=1)

X_val_rfe1 = X_val_rescaled.drop(X_val_rescaled.columns.difference(X_val_rescaled.columns[selector1.support_]), axis=1)
X_val_rfe2 = X_val_rescaled.drop(X_val_rescaled.columns.difference(X_val_rescaled.columns[selector2.support_]), axis=1)
X_val_rfe3 = X_val_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector3.support_]), axis=1)
X_val_rfe4 = X_val_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector4.support_]), axis=1)
X_val_rfe5 = X_val_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector5.support_]), axis=1)
X_val_rfe6 = X_val_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector6.support_]), axis=1)
X_val_rfe7 = X_val_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector7.support_]), axis=1)
X_val_rfe8 = X_val_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector8.support_]), axis=1)
X_val_rfe9 = X_val_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector9.support_]), axis=1)

#Training a simple svm model in order to see which set of features
#performs better
clf = SVC(random_state = 0, class_weight='balanced', kernel = 'linear')
model_res1 = clf.fit(X_train_rfe1, y_train.values.ravel())
clf2 = SVC(random_state = 0, class_weight='balanced', kernel = 'linear')
model_res2 = clf2.fit(X_train_rfe2, y_train.values.ravel())
clf3 = SVC(random_state = 0, class_weight='balanced', kernel = 'linear')
model_res3 = clf3.fit(X_train_rfe3, y_train.values.ravel())
clf4 = SVC(random_state = 0, class_weight='balanced', kernel = 'linear')
model_res4 = clf4.fit(X_train_rfe4, y_train.values.ravel())
clf5 = SVC(random_state = 0, class_weight='balanced', kernel = 'linear')
model_res5 = clf5.fit(X_train_rfe5, y_train.values.ravel())
clf6 = SVC(random_state = 0, class_weight='balanced', kernel = 'linear')
model_res6 = clf6.fit(X_train_rfe6, y_train.values.ravel())
clf7 = SVC(random_state = 0, class_weight='balanced', kernel = 'linear')
model_res7 = clf7.fit(X_train_rfe7, y_train.values.ravel())
clf8 = SVC(random_state = 0, class_weight='balanced', kernel = 'linear')
model_res8 = clf8.fit(X_train_rfe8, y_train.values.ravel())
clf9 = SVC(random_state = 0, class_weight='balanced', kernel = 'linear')
model_res9 = clf9.fit(X_train_rfe9, y_train.values.ravel())

X_train_rfe1 = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector1.support_]), axis=1)

fig = plt.figure()

feat_imp = pd.Series(abs,8.27))
feat_imp.savefig('feature_importances.pdf')

#Renaming features
column_names = X_train_rfe3.rename({ 'agea': 'age', 'gndr_2.0': 'gender_female', 'tporgwk_5.0':'self-employed', 'imprich':'not important to be rich',
    
                                    'mnrgtjb':'men have more rights to do jobs when jobs are scarce','stfdem':'satisfied with the democracy at my country',
                    
                                    'wkhct':'total hours work excluding overtime','ipjbini':'important to take initiatives at job','wkjbndm': 'don\'t enjoy my work',
                                    'pdjbndm':'not enjoy having paid job even if I did not need money','hincsrca_6.0':'main source income:social benefits/grants', 
                                     'tporgwk_3.0': 'work for a state own enterprise','yrspdwka':'years of working','domicil_2.0':'living in outskirts of a big city',
                                     'trstprt':'trust in politician parties', 'health': 'bad health',
                                     'plccbrb':'police takes bribes', 'gvprppv':'government shouldn\'t do more to prevent poverty',
                                     'hincsrca_5.0':'main source income - unemployment allowance','hincsrca_3.0':'main source of income - income from farming'}, axis='columns', inplace=False)

#Creating the graphs for feature importances
feat_imp = pd.Series((abs(model_res3.coef_[0])), index=column_names.columns).nlargest(20).plot(kind='barh')
#ax.tick_params(axis='y', labelsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
fig.savefig('feature_importances.pdf')

#Another plot for feature importances
def plot_coefficients(classifier, feature_names):
    coef = classifier.coef_.ravel()
    num_features_pos = sum(coef>0)
    num_features_negative = len(coef) - num_features_pos
    top_positive_coefficients = np.argsort(coef)[-num_features_pos:]
    top_negative_coefficients = np.argsort(coef)[:num_features_negative]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    #plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0,  2 * top_features), feature_names[top_coefficients], rotation=90, ha='right', fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()
 
 
plot_coefficients(model_res3, column_names.columns)
plt.figure(figsize=(15, 5))

#Bar plot feature number vs accuracy score
sns.barplot(x=feature_no, y= axis_y, color='blue')

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 


fig, ax = plt.subplots(1, 1)
bar = sns.barplot(x=feature_no, y= axis_y, color='blue', axes = ax)

for i, v in enumerate(axis_y):
    ax.text(i-0.34, v +0.025, f"{v:.4f}", color='black',fontsize=25 )
    
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axes.yaxis.set_visible(False)

fig.savefig('accuracy vs feature number.pdf')


#Testing the accuracy on the validation set


val_model = model_res1.predict(X_val_rfe1)
balanced_accuracy_score(y_val, val_model)
val_model2 = model_res2.predict(X_val_rfe2)
balanced_accuracy_score(y_val, val_model2)

val_model3 = model_res3.predict(X_val_rfe3)
balanced_accuracy_score(y_val, val_model3)

val_model4 = model_res4.predict(X_val_rfe4)
balanced_accuracy_score(y_val, val_model4)

val_model5 = model_res5.predict(X_val_rfe5)
balanced_accuracy_score(y_val, val_model5)
val_model6 = model_res6.predict(X_val_rfe6)
balanced_accuracy_score(y_val, val_model6)
val_model7 = model_res7.predict(X_val_rfe7)
balanced_accuracy_score(y_val, val_model7)
val_model8 = model_res8.predict(X_val_rfe8)
balanced_accuracy_score(y_val, val_model8)
val_model9 = model_res9.predict(X_val_rfe9)
balanced_accuracy_score(y_val, val_model9)

feature_no = [X_val_rfe2.shape[1],X_val_rfe3.shape[1], X_val_rfe4.shape[1], X_val_rfe1.shape[1], 
              X_val_rfe5.shape[1], X_val_rfe6.shape[1], X_val_rfe7.shape[1], X_val_rfe8.shape[1], 
              X_val_rfe9.shape[1]]
axis_y = [balanced_accuracy_score(y_val, val_model2), balanced_accuracy_score(y_val, val_model3), 
          balanced_accuracy_score(y_val, val_model4), balanced_accuracy_score(y_val, val_model), 
          balanced_accuracy_score(y_val, val_model5), balanced_accuracy_score(y_val, val_model6), 
          balanced_accuracy_score(y_val, val_model7), balanced_accuracy_score(y_val, val_model8) ,
          balanced_accuracy_score(y_val, val_model9)]



#Looking at what columns the selected model has chosen 
print('Original features :', X_train_rescaled.columns)
print('Best features :', X_train_rescaled.columns[selector6.support_])
print('Importances for those features :', selector6.estimator_.coef_)




# Saving the final dataset with X features 
X_train_final = X_train_rescaled.drop(X_train_rescaled.columns.difference(X_train_rescaled.columns[selector3.support_]), axis=1)
X_val_final = X_val_rescaled.drop(X_val_rescaled.columns.difference(X_val_rescaled.columns[selector3.support_]), axis=1)
X_test_final = X_test_rescaled.drop(X_test_rescaled.columns.difference(X_test_rescaled.columns[selector3.support_]), axis=1)

#Saving the final dataset in order to use it for the analysis models
X_train_final.to_csv('X_train_final.csv', index=False)
X_test_final.to_csv('X_test_final.csv', index=False)
X_val_final.to_csv('X_val_final.csv', index=False)



# Assessing whether the feature selection method improved the models' performance

from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
#Logistic Regression
#Full Dataset
X_rfe = X_train_rescaled.append(X_val_rescaled)
y_rfe = y_train.append(y_val)

clf_rfe = LogisticRegression(random_state = 0, class_weight = 'balanced', max_iter=500)
model_res_rfe = clf_rfe.fit(X_rfe, y_rfe.values.ravel())

test_log_no_tun_rfe = model_res_rfe.predict(X_test_rescaled)

print("The accuracy for the logistic regression model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_log_no_tun_rfe))
      
print("The confusion matrix for the logistic regression model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_log_no_tun_rfe))


#After feature selection

X_train_val = X_train_final.append(X_val_final)
y_train_val = y_train.append(y_val)

clf = LogisticRegression(random_state = 0, class_weight = 'balanced', max_iter=500)
model_res = clf.fit(X_train_val, y_train_val.values.ravel())

test_log_no_tun = model_res.predict(X_test_rescaled)

print("The accuracy for the logistic regression model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_log_no_tun))
      
print("The confusion matrix for the logistic regression model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_log_no_tun))

#SVM


clf_svm_rfe= SVC(random_state = 0, class_weight = 'balanced')
model_res_svm_rfe = clf_svm_rfe.fit(X_rfe, y_rfe.values.ravel())

test_svm_no_tun_rfe = model_res_svm_rfe.predict(X_test_rescaled)

print("The accuracy for the SVM model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_svm_no_tun_rfe))
      
print("The confusion matrix for the SVM model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_svm_no_tun_rfe))



#After feature selection

clf_svm= SVC(random_state = 0, class_weight = 'balanced')
model_res_svm = clf_svm.fit(X_train_val, y_train_val.values.ravel())

test_svm_no_tun = model_res_svm.predict(X_test_rescaled)

print("The accuracy for the SVM model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_svm_no_tun))
      
print("The confusion matrix for the SVM model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_svm_no_tun))


#RF
clf_rf_rfe= RandomForestClassifier(random_state = 0, class_weight = 'balanced')
model_res_rf_rfe = clf_rf_rfe.fit(X_rfe, y_rfe.values.ravel())

test_rf_no_tun_rfe = model_res_rf_rfe.predict(X_test_rescaled)


print("The accuracy for the Random Forests model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_rf_no_tun_rfe))
      
print("The confusion matrix for the Random Forests model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_rf_no_tun_rfe))


#After feature selection


clf_rf= RandomForestClassifier(random_state = 0, class_weight = 'balanced')
model_res_rf = clf_rf.fit(X_train_val, y_train_val.values.ravel())

test_rf_no_tun = model_res_rf.predict(X_test_rescaled)


print("The accuracy for the Random Forests model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_rf_no_tun))
      
print("The confusion matrix for the Random Forests model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_rf_no_tun))

#MLP

#Full Dataset
clf_mlp_rfe =  MLPClassifier(hidden_layer_sizes = (50, 50, 50, 50), random_state = 0, max_iter =2000)
model_res_mlp_rfe = clf_mlp_rfe.fit(X_rfe, y_rfe.values.ravel())

test_rf_no_mlp_rfe = model_res_rf_rfe.predict(X_test_rescaled)


print("The accuracy for the MLP model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_mlp_no_tun_rfe))
      
print("The confusion matrix for the MLP model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_mlp_no_tun_rfe))

#After feature seleection


clf_mlp =  MLPClassifier(hidden_layer_sizes = (50, 50, 50, 50), random_state = 0, max_iter =2000)
model_res_mlp = clf_mlp.fit(X_train_val, y_train_val.values.ravel())

test_rf_no_mlp = model_res_rf.predict(X_test_rescaled)


print("The accuracy for the MLP model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_mlp_no_tun))
      
print("The confusion matrix for the MLP model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_mlp_no_tun))