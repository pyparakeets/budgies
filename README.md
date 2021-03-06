# threshold_optimizer
This python library allows you to conveniently evaluate predicted probablilities during a binary classification task by presenting you with the optimum probability thresholds.

## Introduction
*Classification* tasks in machine learning involves models or algorithms learning to *assign class lables* to elements of a set. **Binary Classification** is the process of assigning elements to two class labels on the basis of a classification rule. Some of the examples of binary classification includes classifying mails under 'spam' or 'not a spam', medical tests ('cancer detected' or 'cancer not detected') and churn prediction ('churn' or 'not').           

Evaluating machine learning models is an important aspect of building models. These evaluations are done using classification metrics, the metrics used depends on the nature of the problem you're solving and the cost of falsely predicted values. Some of these metrics include: confusion matrix, accuracy, precision, recall, F1 score and ROC curve. However these decisions by the metrics are based on a set threshold. 

For instance, in order to map a probability representation from logistic regression to a binary category, you must define a classification threshold (also called the decision threshold). In say a cancer patient classification, a value above that threshold indicates "Patient has cancer"; a value below indicates "Patient does not have cancer." It is tempting to assume that the classification threshold should always be 0.5, but thresholds are problem-dependent, and are therefore values that you must tune.

This library allows you to output the optimum threshold value for the metric you're using to evaluate your classification model. The metrics for which you can get the optimum threshold outputs are: 
> Accuracy

> F1 Score

> Recall

> Specificity

> Precision


### Requirements
> **scikit-learn** == 0.24.0

> **pandas** == 0.25.1

> **numpy** == 1.17.1


### Installation



### Usage

**Code To Follow**
```
from threshold_optimizer import ThresholdOptimizer
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# load data sets
X, y = datasets.load_breast_cancer(return_X_y=True)

# train, val, test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# fit estimator
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# predict probabilities
predicted_probabilities = clf.predict_proba(X_val)

# apply optimization
thresh_opt = ThresholdOptimizer(
        y_score = predicted_probabilities,
        y_true = y_val
    )

# optimize for accuracy and f1 score
thresh_opt.optimize_metrics(
        metrics=['accuracy', 'f1'],
        verbose=True
    )

# display results
print(thresh_opt.optimized_metrics)

# access threshold per metric
accuracy_threshold = thresh_opt.optimized_metrics.accuracy.best_threshold
f1_threshold = thresh_opt.optimized_metrics.f1.best_threshold

# use best accuracy threshold for test set to convert probabilities to classes
predicted_probabilities = clf.predict_proba(X_test)
classes = np.where(predicted_probabilities[:,1], > accuracy_threshold, 1, 0)
print(classes)
```


### Key Terminologies
:TODO
