#%%

#^ Homework2 Instructions
#? 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each

#? 2. Expand to include larger number of classifiers and hyperparameter settings

#? 3. Find some sample data

#? 4. generate matplotlib plots that will assist in identifying the optimal clf and parameters settings

#? 5. Please set up your code to be run and save the results to the directory that its executed from

#? 6. Investigate grid search function


#%%
# Library Install
import numpy as np
import os as os
import operator
import pprint as pp
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from collections import defaultdict
import pickle

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# %%
#? Death to grid search commencing...
bc_data = load_breast_cancer()
X = bc_data.data
y = bc_data.target
n_folds = 5
data = (X, y, n_folds)

#? Input - put your desired algorithm and parms
inputs = \
  [\
    {
      'clf': [RandomForestClassifier()],
      'clf__n_estimators' : [100, 300, 500, 800, 1200],
      'clf__max_depth' : [5, 8, 15, 25, 30],
      'clf__min_samples_split' : [2, 5, 10, 15, 100], 
      'clf__min_samples_leaf' : [1, 2, 5, 10]
    },
    {
      'clf': [SVC()],
      'clf__C': [0.001, 0.1, 1, 10, 100, 1000],
      'clf__kernel': ['linear', 'rbf'],
      'clf__gamma' : [1, 0.1, 0.001, 0.0001]
    },
     {
      'clf': [LogisticRegression()],
      'clf__penalty' : ['none', 'l1', 'l2', 'elasticnet'],
      'clf__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
      'clf__C' : [0.01, 0.1, 10, 100]
    },
    {
      'clf': [LogisticRegression()],
      'clf__penalty' : ['l2'],
      'clf__solver' : ['lbfgs', 'sag'],
      'clf__C' : [0.01, 0.1, 10, 100]
    }
    ]

#? Build the Function
def deathgrid(data, inputs):
  X, y, n_folds = data # unpack data
  result = [] # store results
  scores = []
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
  for params in inputs:
    clf = params['clf'][0]
    params.pop('clf')
    steps = [('clf',clf)]
    grid = GridSearchCV(Pipeline(steps), param_grid = params, refit = 'accuracy', cv = n_folds, return_train_score = False)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    predictions = [round(value) for value in y_pred]
    result.append\
    (
      {
        'Grid Search Parameters': grid,
        'Best Parameters': grid.best_params_,
        'Best Score': '{0:.4f}'.format(grid.best_score_),
        'Accuracy' : '{0:.4f}'.format(accuracy_score(y_test, predictions)),
        'Precision' : '{0:.4f}'.format(precision_score(y_test, predictions, average = 'weighted')),
        'Recall' : '{0:.4f}'.format(recall_score(y_test, predictions, average = 'weighted')),
        'MSE' : '{0:.4f}'.format(mean_squared_error(y_test, predictions)),
        'R2' : '{0:.4f}'.format(r2_score(y_test, predictions)),
        '# of Folds': grid.cv,
      }
    )
    scores.append\
    (
      {
        'Grid Search Parameters': grid,
        'Results' : grid.cv_results_,
      }
    )
    
  results = sorted(result, key = operator.itemgetter('Accuracy'),reverse = True)

  with open('results.txt', 'w') as f:
      for item in results:
          f.write("%s\n" % item) # write out results to file

  with open('scores.txt', 'w') as f:
      for item in scores:
          f.write("%s\n" % item) # write out scores to file

  return results

deathgrid(data, inputs)



#%%
#? Let's generate some plots!

with open('scores.txt') as f:
     scores = [line.rstrip('\n').strip() for line in f]

# I literally banged my head against the wall trying to figure out why my code wouldn't work for this part. I had the results writing to a txt file to then bring in and the goal was the break down the list/dict to generate plots. I couldn't figure out how to get the data out of the list in a proper format to test. At a certain point, I had to stop for time constraints and other deadlines.

# %%
#? Grid Search Function
# Link : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
dir(GridSearchCV)

# The grid search function runs the specified model on the specified dataset while iterating over 'parameter' grids specified by the user. It is designed to not only do cross validation, but also determine the optimal parameters given the inputed parameter grid.

# best_params_ will show the optimized paramters chosen

# cv_results_ will show the entirety of the grid search results, allowing plots to be made off of them and to see how the metrics change at each run.  This is what I was trying to make work in the above plotting section but couldn't get it working.

# best_estimator will give the estimator that achieved the highest results from the run

# best_score will provide the mean cross-validated score of the best_estimator