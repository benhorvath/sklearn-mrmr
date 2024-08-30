# `sklearn-mrmr`: MRMR feature selection for sklearn

_**Release date**: August 30, 2024, v.0.1_

This repo provides a Python library that implements `scikit-learn`-compatible feature selection via Minimum Redunancy - Maximum Relevance. It aims to work seamlessly with `scikit-learn`'s pipelines, hyperparameter optimization, and models. Both regression and classification tasks are supported. The number of features selected by MRMR is itself a hyperparameter, and can be tuned using `scikit-learn`'s pipeline and grid search functionality.

Other repos on Github implement MRMR in Python, they often lack compatibility with `scikit-learn`, limiting their utility.

MRMR evaluates a feature's score based on its _relevance_ to the target variable and its _redundancy_ with other features. The goal is to select the features that have strong relationships with the target variable, and also minimally redundant.

Four variants of MRMR are implemented. The canonical variant uses mututal information (MI) to calculate redundancy and relevance. However, since MI can be an resource-heavy process, other formulations have been proposed. (Although this library use's `scikit-learn`'s implementaiton of mututal information, which is quite optimized and offers parallel processing.) A second variant was developed, that uses the F-test to calculate relevance and Pearson correlation to calculate relevance. This proves to be much faster, and without a clear loss in performance. Additionally, variants may use substraction or division. 

Variants using subtraction:

MI: $$f^{canonical}(X_i) = MI(Y, X_i) - \frac{1}{S} \sum_{X_s \in S} MI(X_s, X_i)$$

F-test: $$f^{Ftest}(X_i) = F(Y, X_i) - \frac{1}{S} \sum_{X_s \in S} \rho(X_s, X_i)$$

Note that MRMR is not guaranteed to improve your model's performance. As with anthing ML, its effectiveness depends on your data and modeling strategy. My (anecdotal) experience seem to suggest that MRMR is particularly beneficial in scenarios involving high model complexity / many correlated features. The benefit can come as either improved performance or decreased variance.

## Installation

To install from this Github repo, clone this repo and install:

```
python setup.py install
```

## Example

See [demo.py](https://github.com/benhorvath/sklearn-MRMR/blob/main/demo.ipynb) for a example of how to use this library with `scikit-learn`'s functionality.

## References

* Original MRMR paper: "[Minimum redundancy feature selection from microarray gene expression data](https://pubmed.ncbi.nlm.nih.gov/15852500/)"
* Uber's more recent paper: "[Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://arxiv.org/pdf/1908.05376)"
* Samuele Mazzanti's "[MRMR Explained: Exactly How You Wished Someone Explained to You](https://towardsdatascience.com/MRMR-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b)"
