# sklearn-mRMR: mRMR feature selection for sklearn

This repo provides a Python library that implements `sklearn`-compatible feature selection via minimum Redunancy - Maximum Relevance. It aims to work seamlessly with sklearn's pipelines, hyperparameter optimization, and models. Both regression and classification tasks are supported.

The idea of the mRMR algorithm is to ...

Currently, two variants of mRMR are implemented. The canonical variant uses mututal information (MI) to calculate redundancy and relevance. However, since MI can be an resource-heavy process -- although this library makes it easy to use more than one process to calculate MI. A second variant was developed, that uses F-test to calculate relevance and Pearson correlation to calculate relevance. This proves to be much faster, and without a clear loss in performance. Additionally, some variants may use substraction or division. This is also supported.

Other repos on Github implement mRMR in Python, but they are not specifically designed to work with sklearn, limiting their utility.

MI: $$f^{canonical}(X_i) = MI(Y, X_i) - \frac{1}{S} \sum_{X_s \in S} MI(X_s, X_i)$$

F-test: $$ f^{Ftest}(X_i) = F(Y, X_i) - \frac{1}{S} \sum_{X_s \in S} \rho(X_s, X_i)$$


## Installation

To install from this Github repo, clone to repo and run:

```
python setup.py blah blah
```

To install from PyPi, you can use pip:

```
pip3 install sklearn-mrmr
```


## Example





TODOS:

0. Set up requirements.txt
1. Test not using `sparse_output=False`
3. Need to pass in kwargs somewhere to get to base function?
5. Study on multiple data sets and feature selection and comparing results
    - Random forest: No feature selection, mRMR, f_regression/mututal_info_regression
    - Lasso (OR JUST LR??): No feature selection, mRMR, f_regression/mututal_info_regression
6. Re-read original papers for any missed clues
8. Testing apparatus 
9. Submit to PyPi


TESTING

- Verify that attributes such as k_best, score_func, etc., are correctly set during initialization.
- Test the fit method with sample data to ensure it executes without errors; and that attributes are properly set
- Test the transform method with sample data
- Test the fit transform method with sample data
- Test the transformer's integration within an sklearn pipeline

## Future work

- Test using xi for mRMR; will need a fast implementation