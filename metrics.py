import numpy as np
import logging

# export all the below
__ALL__=[
    "msqr",
    "pearson_corr_coef",
    "r2_score"
]

def msqr(actual, pred):
    return np.average(np.square(np.linalg.norm(actual-pred, axis=1)))

def pearson_corr_coef(X,y):
    """
    Correlation only gives a how two variables are linearly depedent
    it's dimensionless and doesn't specify by what extent does both varibales relate
    it lies between -1 to 1
    -1 strong negative correlation, +1 strong positive
    it can also be calculated as the slope of the regression line
    :param X: feature x
    :param y: feature y
    :return: correlation among them
    """
    logging.warning("The value of pearson correlation coefficient can never cross the bounds [-1,1]. If it crosses, it could be due to floating point arithmetic")
    # summation of all cov of all datapoints
    cov_x_y = np.sum((X-np.mean(X))*(y-np.mean(y)), axis=0)
    # summation of all std dev of all data points
    std_dev_x = np.sqrt(np.sum(np.square(X-np.mean(X)), axis=0))
    std_dev_y = np.sqrt(np.sum(np.square(y-np.mean(y)), axis=0))
    return cov_x_y /(std_dev_x * std_dev_y)

def compute_all_metrics(actual, pred):
    mean_sqr_err = msqr(actual, pred)
    pearson_coeff = pearson_corr_coef(actual, pred)
    return mean_sqr_err, pearson_coeff

def r2_score(actual, pred):
    # residual sum of squares (Ypred - Yactual)**2
    residual_ss = np.sum(np.square(pred - actual))

    #total sum of sqaures
    total_ss = np.sum(np.square(actual - np.mean(actual)))
    # in case of a zero denomitor set r2_score to 0 to avoid -inf
    return 1 - (residual_ss/total_ss) if total_ss !=0 else 0








