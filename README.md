# sklearn-mRMR: mRMR feature selection for sklearn

This repo provides a Python library that implements `sklearn`-compatible feature selection via Minimum Redunancy - Maximum Relevance. It aims to work seamlessly with `sklearn`'s pipelines, hyperparameter optimization, and models. Both regression and classification tasks are supported.

Other repos on Github implement MRMR in Python, but they are not specifically designed to work with `sklearn`, limiting their utility.

A feature's MRMR score is determined by its _relevance_ score and its _redundancy_ score. This aims to get the set of features that has strong relationship with the target variable, while keeping them as independent from eachother as possible.

Four variants of MRMR are implemented. The canonical variant uses mututal information (MI) to calculate redundancy and relevance. However, since MI can be an resource-heavy process, other formulations have been proposed. (Although this library use's `sklearn`'s implementaiton of mututal information, which is quite optimized and offers parallel processing.) A second variant was developed, that uses the F-test to calculate relevance and Pearson correlation to calculate relevance. This proves to be much faster, and without a clear loss in performance. Additionally, variants may use substraction or division. 

Variants using subtraction:

MI: $$f^{canonical}(X_i) = MI(Y, X_i) - \frac{1}{S} \sum_{X_s \in S} MI(X_s, X_i)$$

F-test: $$f^{Ftest}(X_i) = F(Y, X_i) - \frac{1}{S} \sum_{X_s \in S} \rho(X_s, X_i)$$

## Installation

To install from this Github repo, clone to repo and run:

```
python setup.py install
```

## Example

See `demo.py` for a example of how to use this library with `sklearn`'s functionality.

## References

* Original MRMR paper: "[Minimum redundancy feature selection from microarray gene expression data](https://pubmed.ncbi.nlm.nih.gov/15852500/)"
* Uber's more recent paper: "[Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://arxiv.org/pdf/1908.05376)"
* Samuele Mazzanti's "[MRMR Explained: Exactly How You Wished Someone Explained to You](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b)"