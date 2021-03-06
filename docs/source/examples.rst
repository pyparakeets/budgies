Examples
========

Installation/Usage:
*******************
pip install -U threshold-optimizer

Example Usages
**************************************************
.. code-block:: python

    # import all packages
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