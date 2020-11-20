# Budgies data

All data used are stored in this folder.

### sklearn datasets
* Primary source of dataset for initial development
```python
# toy dataset - binary classification
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# toy dataset - multi-class classification
from sklearn.datasets import load_wine
data = load_wine()

# real world dataset - multi-class classification
from sklearn.datasets import fetch_20newsgroups_vectorized
```

