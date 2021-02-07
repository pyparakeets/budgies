# Budgies
This python library allows you to conveniently evaluate predicted probablilities during a binary classification task by presenting you with the optimum probability thresholds.

## Introduction
*Classification* tasks in machine learning involves models or algorithms learning to *assign class lables* to elements of a set. **Binary Classification** is the process of assignig elements to two class labels on the basis of a classification rule. Some of the examples of binary classification includes classifying mails under 'spam' or 'not a spam', medical tests ('cancer detected' or 'cancer not detected') and churn prediction ('churn' or 'not').           

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

> 1. load data and create train, validation and test sets
> 2. run model on train data
> 3. predict probabilities on validation set
> 4. import threshold_optimizer
> 5. create threshold_optimizer object
> 6. pass predicted probabilities into threshold optimizer object
> 7. call threshold_optimizer.optimze_accuracy (or whichever metric) and save returned probability_threshold_value
> 8. predict probabilities on test set
> 9. use saved threshold to create binary classes
> 10. evaluate optimized classes with metric optimized for


### Key Terminologies
*No need for one yet*
