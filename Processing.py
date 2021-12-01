# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 22:17:40 2021

@author: kanta
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
#Reading the data
X_train = pd.read_csv('X_train_final.csv') 
X_test = pd.read_csv('X_test_final.csv') 
X_val = pd.read_csv('X_val_final.csv') 
y_test = pd.read_csv('y_test.csv')
y_val = pd.read_csv('y_val.csv')
y_train = pd.read_csv('y_train.csv')
X_train_val = X_train.append(X_val)
y_train_val = y_train.append(y_val)


#LOGISTIC REGRESSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Model without tuning
clf = LogisticRegression(random_state = 0, class_weight = 'balanced', max_iter=500)
model_res = clf.fit(X_train_val, y_train_val.values.ravel())

test_log_no_tun = model_res.predict(X_test)

print("The accuracy for the logistic regression model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_log_no_tun))
      
print("The confusion matrix for the logistic regression model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_log_no_tun))
       

#FINE TUNING LOGISTIC REGRESSION

#https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
# Define models and parameters
model = LogisticRegression(random_state = 1, class_weight = 'balanced', max_iter=1000)
c_values = [100, 50, 20, 5, 1.0, 0.5, 0.1, 0.05, 0.01]
# Define grid search
grid = dict(C=c_values)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

#Fitting the model
random_search = RandomizedSearchCV(estimator = model, param_distributions = grid, 
                                   cv = cv, scoring = 'balanced_accuracy', n_iter = 20)
random_result = random_search.fit(X_train_val, y_train_val.values.ravel())

#Summarizing the results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
means = random_result.cv_results_['mean_test_score']
stds = random_result.cv_results_['std_test_score']
params = random_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#Selecting the parameters from the best performing model and testing it on test data

final_model_lg = random_result.best_estimator_
final_model_fitting = final_model_lg.fit(X_train_val, y_train_val.values.ravel())

#Predicting the data on the test set

test_log_tun = final_model_fitting.predict(X_test)


print("The accuracy for the logistic regression model with tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_log_tun))
      
print("The confusion matrix for the logistic regression model with tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_log_tun))
       
a1 = plot_confusion_matrix(final_model_lg, X_test, y_test.values.ravel(),
                      cmap  = 'Blues', colorbar = False, display_labels = ['Retire > 64','Retire early']) 
plt.grid(False)
plt.rcParams.update({'font.size': 20})
plt.xlabel('Predicted label',fontsize=20)
plt.ylabel('True label', fontsize=20) 
sns.set(font_scale=4.0) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()       

#SUPPORT VECTOR MACHINE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#The model without fine tuning 

clf_svm= SVC(random_state = 0, class_weight = 'balanced')
model_res_svm = clf_svm.fit(X_train_val, y_train_val.values.ravel())

test_svm_no_tun = model_res_svm.predict(X_test)

print("The accuracy for the SVM model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_svm_no_tun))
      
print("The confusion matrix for the SVM model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_svm_no_tun))


#FINE TUNING SUPPORT VECTOR MACHINE~~~~~~~~~~~~~~
#https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/

# Define model and parameters
model_svm = SVC(random_state = 0, class_weight = 'balanced')
kernel = ['linear', 'poly' , 'rbf', 'sigmoid']
C = [50, 30,40, 20,10, 5, 1.0, 0.5, 0.1, 0.05, 0.01]

# Define grid search
grid_svm = dict(C = C, kernel=kernel)
cv_svm = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = 0)
#Fitting the model
random_search_svm = RandomizedSearchCV(estimator = model_svm, 
        param_distributions = grid_svm, n_jobs = -1, cv = cv_svm, scoring = 'balanced_accuracy',
        error_score = 0, n_iter = 40)
random_result_svm = random_search_svm.fit(X_train_val, y_train_val.values.ravel())

# Summarize results

print("Best: %f using %s" % (random_result_svm.best_score_, random_result_svm.best_params_))
means_svm = random_result_svm.cv_results_['mean_test_score']
stds_svm = random_result_svm.cv_results_['std_test_score']
params_svm = random_result_svm.cv_results_['params']
for mean_svm, stdev_svm, param_svm in zip(means_svm, stds_svm, params_svm):
    print("%f (%f) with: %r" % (mean_svm, stdev_svm, param_svm))


#Selecting the parameters from the best performing model and testing it on test data

new_svm = SVC(random_state=0, class_weight='balanced', kernel ='rbf', C=1)
final_model_svm2 = new_svm.fit(X_train_val, y_train_val.values.ravel())
final_model_fitting_svm2 = final_model_svm2.fit(X_train_val, y_train_val.values.ravel())
test_model_svm = final_model_fitting_svm2.predict(X_test)
balanced_accuracy_score(y_test, test_model_svm)

final_model_svm = random_result_svm.best_estimator_
final_model_fitting_svm = final_model_svm.fit(X_train_val, y_train_val.values.ravel())

#Predicting the data on the test set

test_svm_tun = final_model_fitting_svm.predict(X_test)

print("The accuracy for the Support Vector Machines model with tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_svm_tun))
      
print("The confusion matrix for the Support Vector Machines model with tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_svm_tun))
       
a2 = plot_confusion_matrix(final_model_fitting_svm, X_test, y_test.values.ravel(),
                      cmap  = 'Blues', colorbar = False, display_labels = ['Retire > 64','Retire early']) 
plt.grid(False)
plt.xlabel('Predicted label',fontsize=20)
plt.ylabel('True label', fontsize=20) 
sns.set(font_scale=3.0) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()   
plt.show()


#RANDOM FOREST~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The model without fine tuning

clf_rf= RandomForestClassifier(random_state = 0, class_weight = 'balanced')
model_res_rf = clf_rf.fit(X_train_val, y_train_val.values.ravel())

test_rf_no_tun = model_res_rf.predict(X_test)


print("The accuracy for the Random Forests model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_rf_no_tun))
      
print("The confusion matrix for the Random Forests model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_rf_no_tun))


#FINE TUNING RANDOM FOREST~~~~~~~~~~~~
#https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
 

model_mlp = RandomForestClassifier(random_state = 0, class_weight='balanced')

# define the grid search parameters
n_estimators= [10,50, 100, 200, 500, 750, 1000]
max_features=[1, X_train_val.shape[1]]
param_grid_rf = dict(n_estimators=n_estimators, max_features=max_features)
cv_rf = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = 0)
#Fitting the model
random_search_rf = RandomizedSearchCV(model_rf, param_distributions=param_grid_rf,
                    n_jobs = -1, cv = cv_rf, scoring = 'balanced_accuracy', n_iter = 20)
random_result_rf = random_search_rf.fit(X_train_val, y_train_val.values.ravel())

# summarize results
print("Best: %f using %s" % (random_result_rf.best_score_, random_result_rf.best_params_))
means_rf = random_result_rf.cv_results_['mean_test_score']
stds_rf = random_result_rf.cv_results_['std_test_score']
params_rf = random_result_rf.cv_results_['params']
for mean_rf, stdev_rf, param_rf in zip(means_rf, stds_rf, params_rf):
    print("%f (%f) with: %r" % (mean_rf, stdev_rf, param_rf))

#Fitting the best model in the train and valid data

final_model_rf = random_result_rf.best_estimator_
final_model_fitting_rf = final_model_rf.fit(X_train_val, y_train_val.values.ravel())

#Predicting the data on the test set
#Selecting the parameters from the best performing model and testing it on test data

test_rf_tun = final_model_fitting_rf.predict(X_test)

print("The accuracy for the Random Forest model with tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_rf_tun))
      
print("The confusion matrix for the Random Forest model with tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_rf_tun))


a3 = plot_confusion_matrix(final_model_fitting_rf, X_test, y_test.values.ravel(),
        cmap  = 'Blues', colorbar = False, display_labels = ['Retire > 64','Retire early']) 
plt.grid(False)
plt.xlabel('Predicted label',fontsize=20)
plt.ylabel('True label', fontsize=20) 
sns.set(font_scale=3.0) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
plt.show()

#MULTILAYER PERCEPTRON

clf_mlp =  MLPClassifier(hidden_layer_sizes = (50, 50, 50, 50), random_state = 0, max_iter =2000)
model_res_mlp = clf_mlp.fit(X_train_val, y_train_val.values.ravel())

test_rf_no_mlp = model_res_rf.predict(X_test)


print("The accuracy for the MLP model without tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_mlp_no_tun))
      
print("The confusion matrix for the MLP model without tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_mlp_no_tun))


#FINE TUNING MULTILAYER PERCEPTRON ~~~~~~~~~~~~
#Code inspired from: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

# create model

model_mlp = MLPClassifier(hidden_layer_sizes = (50, 50, 50, 50), random_state = 0, max_iter =2000)

# define the grid search parameters

batch_size = [ 100, 150, 200, 500, 1000, 1200, 1500]
#epochs = [ 250, 500, 700, 1000, 1200, 1500]max_iter = epochs, 
activation = ['identity', 'logistic',  'relu', 'tanh']
param_grid_mlp = dict(batch_size = batch_size, activation = activation)
cv_mlp = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = 0)
#Fitting the model
random_search_mlp = RandomizedSearchCV(model_mlp, param_distributions=param_grid_mlp,
                    n_jobs = -1, cv = cv_mlp, scoring = 'balanced_accuracy', n_iter = 20)
random_result_mlp = random_search_mlp.fit(X_train_val, y_train_val.values.ravel())

# summarize results
print("Best: %f using %s" % (random_result_mlp.best_score_, random_result_mlp.best_params_))
means_mlp = random_result_mlp.cv_results_['mean_test_score']
stds_mlp = random_result_mlp.cv_results_['std_test_score']
params_mlp = random_result_mlp.cv_results_['params']
for mean_mlp, stdev_mlp, param_mlp in zip(means_mlp, stds_mlp, params_mlp):
    print("%f (%f) with: %r" % (mean_mlp, stdev_mlp, param_mlp))

#Fitting the best model in the train and valid data

final_model_mlp = random_result_mlp.best_estimator_
final_model_fitting_mlp = final_model_mlp.fit(X_train_val, y_train_val.values.ravel())

#Predicting the data on the test set
#Selecting the parameters from the best performing model and testing it on test data

test_mlp_tun = final_model_fitting_mlp.predict(X_test)

print("The accuracy for the Multilayer Perceptron model with tuning is:", 
      balanced_accuracy_score(y_test.values.ravel(), test_mlp_tun))
      
print("The confusion matrix for the Multilayer Perceptron model with tuning is:", 
       confusion_matrix(y_test.values.ravel(), test_mlp_tun))

a4 = plot_confusion_matrix(final_model_fitting_mlp, X_test, y_test.values.ravel(),
                      cmap  = 'Blues', colorbar = False, display_labels = ['Retire > 64','Retire early']) 
plt.grid(False)
plt.xlabel('Predicted label',fontsize=20)
plt.ylabel('True label', fontsize=20) 
sns.set(font_scale=3.0) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show() 
plt.show()    


