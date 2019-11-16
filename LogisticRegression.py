import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as logr
from sklearn.preprocessing import StandardScaler
import logging
from metrics import *
from utils import get_outliers, sigmoid
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.utils import resample
import seaborn as sns
import matplotlib
#matplotlib.use('TkAgg')

from mpl_toolkits import mplot3d


f1_score = sklearn.metrics.f1_score



class LogisiticRegression():
    def __init__(self, X,y, learning_rate=0.1, n_iter=1000, tolerance=1e-4, regularization=None, regularization_alpha=0.001):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.lr = learning_rate
        self.n_iter = n_iter
        self.n_cols= self.X.shape[-1] + self.y.shape[-1]
        self.tolerance= tolerance
        self.regularization_type = regularization
        self.alpha = regularization_alpha

    def __init_theta(self):
        # one per each independent var and a bias
        return np.array([0 for i in range(self.n_cols)]).reshape(-1, 1)

    def update_thetas(self, regularization=None, alpha=0.001):
        if(regularization == "Ridge"):
            # can also be squared normally!
            penalty = np.asscalar(alpha * (self.thetas[1:].T.dot(self.thetas[1:]))/self._X.shape[0])

            # hack to disable penality on theta_0
            penalty = np.hstack((np.repeat(penalty, repeats=self.thetas.shape[0]-1, axis=0), 0)).reshape(-1,1)
        elif(regularization== "Lasso"):
            penalty = 0
            raise NotImplementedError("Lasso regression is not implemented!")
        else:
            penalty= 0
        theta_correction = self.lr*((np.matmul(self._X.T, self.residuals)/self._X.shape[0]) + penalty)
        thetas_corrected = self.thetas - theta_correction
        return thetas_corrected

    def cost(self, actual, predicted):
        return -(actual * np.log(predicted) + (
                (1 - actual) * np.log(1 - predicted))).mean()

    def fit(self):
        self._X = np.hstack((self.X, np.ones((self.X.shape[0], 1))))
        self.thetas  = self.__init_theta()
        n_samples = self._X.shape[0]
        self.categories_h_theta_x = {category:None for category in np.unique(self.y)}
        self.categories_thetas = {category:None for category in np.unique(self.y)}
        self.loss_trend = {category:[] for category in np.unique(self.y)}
        self.y_copy = self.y
        #self.learning_rate_trend={lr:None for lr in [0.1, 0.01, 0.001, 0.0001]}
        #for lr in [0.1, 0.01 ,0.001, 0.0001]:
        for category in self.categories_h_theta_x:
            #reset for every category
            self.thetas = self.__init_theta()
            self.y = self.y_copy
            self.y = np.where(self.y == category, 1, 0)
            # self.lr_copy = self.lr
            # self.lr = lr
            for i in range(self.n_iter):
                old_thetas = self.thetas
                self.h_theta_X = sigmoid(np.matmul(self._X, self.thetas))
                self.residuals =  self.h_theta_X - self.y
                #print(self.loss(pred=self.h_theta_X, actual=self.y))
                self.thetas = self.update_thetas(alpha= self.alpha, regularization=self.regularization_type)
                self.loss_trend[category].append(self.cost(actual=self.y, predicted=self.h_theta_X))
                # if (np.all(np.isclose(old_thetas, self.thetas, atol=self.tolerance, rtol=self.tolerance))):
                #     print("break at {}".format(i))
                #     break
            self.categories_thetas[category] = self.thetas
            self.categories_h_theta_x[category] = self.h_theta_X
            # self.y = self.y_copy
            # self.learning_rate_trend[lr] = self.loss_trend
            # self.loss_trend = {category: [] for category in np.unique(self.y)}

        # print(self.categories_thetas)
        self.y = self.y_copy
        #print({category:self.categories_h_theta_x[category][0] for category in self.categories_h_theta_x})
        labels = list(self.categories_h_theta_x)
        self.train_pred=[labels[index[0]] for index in np.argmax(list(self.categories_h_theta_x.values()), axis=0)]
        actual=np.array(self.y_copy.reshape(-1))
        print(np.count_nonzero(np.equal(self.train_pred, actual)))
        return self

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        n_categories= len(self.categories_h_theta_x)
        n_observations = len(X)
        probs = {category: sigmoid(X.dot(theta)) for category, theta in self.categories_thetas.items()}
        probs = np.array(list(probs.values()))
        predicted_outputs= np.argmax(probs, axis=0)
        categories = list(self.categories_h_theta_x)
        labels = findlabels(categories)(predicted_outputs)
        return predicted_outputs,labels


def findlabels(labels):
    return np.vectorize(lambda x: labels[x])


def preprocess_data(data, remove_outliers=True, normalize=True,seed=12):
    #data=data.fillna(0)
    # data =StandardScaler().fit(data).transform(data)
    target = np.array(data.iloc[:, -1]).reshape(-1,1)
    data = data.iloc[:, :-1]
    if remove_outliers:
        cols = data.shape[-1]
        clean_df = pd.DataFrame()
        outliers_df = pd.DataFrame()
        for col in range(cols):
            clean_df = data[~data.iloc[:, col].isin(get_outliers(data.iloc[:, col]))]
            outliers_df = data[data.iloc[:, col].isin(get_outliers(data.iloc[:, col]))]
        logging.warning("\n {} datapoints are more than 2 std dev away from mean and removed".format(outliers_df))
    else:
        clean_df = data
    if(normalize):
        clean_df = StandardScaler().fit(clean_df).transform(clean_df)

    # if(smote):
    #     sm=SMOTE(ratio=1.0)
    #     for category in np.unique(target):
    #         y = np.where(target == category, 1, 0)
    #         X,y = sm.fit_samples(data, y)
    #         clean_df.append(X)
    #         target.append(y)
    #         y = np.where(target == 1, category, )
    clean_df = np.hstack((clean_df, target))
    train, test = train_test_split(clean_df, test_size=0.2, random_state=seed)



    # selecting 0 to n-1 cols
    X_train = np.array(train[:, :-1], dtype=np.float64)
    X_test = np.array(test[:, :-1], dtype=np.float64)
    y_train = np.array(train[:, -1]).reshape(-1,1)
    y_test = np.array(test[:, -1]).reshape(-1,1)
    # selecting nth columns as target column
    #y = np.array(norm_data[:, -1])
    # X = np.array(clean_df.iloc[:, :-1])
    # y = np.array(clean_df.iloc[:, -1])
    return X_train,X_test,y_train,y_test

def get_majority_class_count(data):
    index,_,counts=np.unique(data,return_counts=True,return_index=True)
    print(index, counts)
    return index[np.argmax(counts)], np.max(counts)

def resample_col(data,  n_samples, seed):
    majority_label, majority_count= get_majority_class_count(data)
    return resample(data, replace=True, n_samples=majority_count, random_state=seed)

if __name__ == "__main__":
    SEED = 123
    data_raw = pd.read_csv("BSOM_DataSet_for_HW2.csv")
    cols_required = ["all_mcqs_avg_n20","all_NBME_avg_n4","LEVEL"] #,"CBSE_02"
    data=data_raw[cols_required]


    conf = {"regularization": False, "normalization": False}
    # data["LEVEL"], mapping_index = pd.Series(data["LEVEL"])
    data = data.dropna()
    data=data.append(resample(data[data["LEVEL"]=="D"], replace=True, n_samples=46-len(data[data["LEVEL"]=="D"]), random_state=SEED), ignore_index=True)
    data = data.append(resample(data[data["LEVEL"] == "A"], replace=True, n_samples=46-len(data[data["LEVEL"]=="A"]), random_state=SEED), ignore_index=True)
    data = data.append(resample(data[data["LEVEL"] == "C"], replace=True, n_samples=46-len(data[data["LEVEL"]=="C"]), random_state=SEED), ignore_index=True)
    #data = data.append(resample(data[data["LEVEL"] == "B"], replace=True, n_samples=100-len(data[data["LEVEL"]=="B"]), random_state=123), ignore_index=True)
    print(get_majority_class_count(data["LEVEL"]))
    X_train, X_test, y_train, y_test = preprocess_data(data[cols_required], remove_outliers=False ,normalize=conf["normalization"])
    # selecting nth columns as targ
    #X,y = preprocess_data(data) #"all_NBME_avg_n4"
    # print(y.shape)
    # #y = np.array(data["LEVEL"])
    # print(y.shape)
    # print(X,y)




    # log_reg=LogisiticRegression(X,y, learning_rate=0.1, n_iter=10000, tolerance=1e-4).fit()
    # y_pred, y_pred_labels=log_reg.predict(X)
    # print(y_pred_labels)
    # print(confusion_matrix(y_pred=y_pred_labels.reshape(-1), y_true=y))
    # print(f1_score(y_pred=y_pred_labels, y_true=y, average="macro"))
    # X = np.hstack((X, np.ones((X.shape[0], 1))))
    # _logr=logr()
    # _logr.fit(X,y)
    # y_pred = _logr.predict(X)
    # print(confusion_matrix(y_pred, y))
    # print(f1_score(y_pred=y_pred, y_true=y, average="macro"))
    # print(_logr.coef_)



    # for all testing purposes only!
    #X,y = make_classification(n_samples=100, n_features=3, n_informative=3, n_classes=4, n_redundant=0, n_repeated=0, random_state=123)
    # print(X,y)
    # print(X, y)




    # # upsampling
    # data=data.append(resample(data[data["LEVEL"]=="D"], replace=True, n_samples=46-len(data[data["LEVEL"]=="D"]), random_state=123), ignore_index=True)
    # data = data.append(resample(data[data["LEVEL"] == "A"], replace=True, n_samples=46-len(data[data["LEVEL"]=="A"]), random_state=123), ignore_index=True)
    # data = data.append(resample(data[data["LEVEL"] == "C"], replace=True, n_samples=46-len(data[data["LEVEL"]=="C"]), random_state=123), ignore_index=True)
    # print(get_majority_class_count(data["LEVEL"]))
    # X = data[["all_mcqs_avg_n20", "all_NBME_avg_n4"]]
    # y = data[["LEVEL"]]
    #
    # # 5 fold validation
    # folds = KFold(n_splits=5)
    # folds.get_n_splits(X)
    # k=0;
    # for train_index, test_index in folds.split(X):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     X_train = np.array(X_train)
    #     X_test = np.array(X_test)
    #     y_train = np.array(y_train)
    #     y_test = np.array(y_test)
    #     log_reg=LogisiticRegression(X_train,y_train, learning_rate=0.1, n_iter=10000, tolerance=1e-4).fit()
    #     y_pred, y_pred_labels = log_reg.predict(X_test)
    #     print("f1_score for fold {} is {}".format(k, f1_score(y_pred=y_pred_labels, y_true=y_test, average="macro")))
    #     k += 1;
    # _X = data[["all_mcqs_avg_n20", "all_NBME_avg_n4"]] #preprocess_data(data[["all_mcqs_avg_n20", "all_NBME_avg_n4"]],normalize=False)
    # y = data["LEVEL"]
    # X = np.array(data.iloc[:, :-1])
    # y = np.array(data.iloc[:, -1])

    # for testing
    #X,y = make_classification(n_samples=100, n_features=3, n_informative=3, n_classes=4, n_redundant=0, n_repeated=0, random_state=123)

    # X = np.array(X[:, :-1])
    # y = np.array(y[:, -1])
    print("---------------" * 5)
    print("""
    logisitc regression with columns 
    @indepedent "all_mcqs_avg_n20", "all_NBME_avg_n4",
    @target-"LEVEL"
    Normalization - Disabled
    Regression - Disabled
    Upsampling - Enabled
    
    """)
    _logr = logr()
    _logr.fit(X_train,y_train)
    y_pred = _logr.predict(X_test)
    print(confusion_matrix(y_pred=y_pred.reshape(-1), y_true=y_test), end="****\n")
    print("f1_score -micro {}".format(f1_score(y_pred=y_pred, y_true=y_test, average="micro")))
    print("f1-score -macro {}".format(f1_score(y_pred=y_pred, y_true=y_test, average="macro")))
    precision, recall, _f1_score,_ = precision_recall_fscore_support(y_pred= y_pred, y_true=y_test, average="macro")
    print(precision, recall, _f1_score)
    print(precision_recall_fscore_support(y_pred= y_pred, y_true=y_test))

    print("---------------"*5)
    #f1_score_trend=[]
    #alphas= [0.00001, 0.0001, 0.001, 0.1, 0]
    #for alpha in alphas:
    log_reg=LogisiticRegression(X_train,y_train, learning_rate=0.1, n_iter=10000, tolerance=1e-4, regularization=conf["regularization"], regularization_alpha=0.01).fit()
    y_pred, y_pred_labels = log_reg.predict(X_test)
    print(confusion_matrix(y_pred=y_pred_labels.reshape(-1), y_true=y_test), end="****\n")
    print("f1_score -micro {}".format(f1_score(y_pred= y_pred_labels, y_true=y_test,  average="micro")))
    print("f1-score -macro {}".format(f1_score(y_pred= y_pred_labels, y_true=y_test,  average="macro")))
    precision, recall, _f1_score,_ = precision_recall_fscore_support(y_pred= y_pred_labels, y_true=y_test, average="macro")
    print(precision, recall, _f1_score)
    print(precision_recall_fscore_support(y_pred= y_pred_labels, y_true=y_test))
    plt.bar("f1_macro", _f1_score, width=0.3)
    plt.bar("f1_macro_w_scaling",0.5573762838468721, width=0.3)
    plt.bar("precision", precision, width=0.3)
    plt.bar("precision_w_scaling", 0.5534188034188035 , width=0.3)
    plt.bar("recall", recall, width=0.3)
    plt.bar("recall_w_scaling", 0.5659722222222223, width=0.3)
    plt.plot([_f1_score, 0.5573762838468721, precision, 0.5534188034188035, recall, 0.5659722222222223], linestyle=":")
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Scaling vs No Scaling Metrics")
    plt.show()
    print("training - scores")
    print(confusion_matrix(y_true=y_train, y_pred=log_reg.train_pred), "train")
    precision, recall, _f1_score, _ = precision_recall_fscore_support(y_pred=y_pred_labels, y_true=y_test,)
    print("precision for training ", precision)
    print("recall for training ", recall)
    print("f1 for training ", _f1_score)
    print(confusion_matrix(y_true=y_train, y_pred=log_reg.train_pred), "train")
    colors = np.where(y_train == "A","red" , "green").reshape(-1)
    plt.scatter(log_reg.categories_h_theta_x["A"], log_reg.categories_h_theta_x["A"] - np.where(y_train == "A",1 , 0), c=colors, label=colors)
    # plt.scatter(_logr)
    plt.show()



    #print(precision, recall, _f1_score)
    #f1_score_trend.append(_f1_score)

    ######## visualize model ##########
    probs = np.array(list(log_reg.categories_h_theta_x.values()))
    predicted_outputs = np.argmax(probs, axis=0)
    categories = list(log_reg.categories_h_theta_x)
    labels = findlabels(categories)(predicted_outputs)


    fig = plt.figure()

    for category in log_reg.categories_h_theta_x:
        ax = plt.axes(projection='3d')
        pred_probability=log_reg.categories_h_theta_x[category]
        pred_probability = np.where(pred_probability >= 0.4 , 1, 0)
        A = X_train[:, :-1]
        B = X_train[:, -1]

        color = ["red" if prob==0 else "green" for prob in pred_probability]
        seq=np.arange(-2,+2,0.25).reshape(-1,2)
        ax.scatter(A[:,0], B,sigmoid(np.matmul(np.hstack((X_train, np.ones((147, 1)))), log_reg.categories_thetas[category])).reshape(-1), c =color)
        #ax.plot(seq[:,0], seq[:,1],sigmoid(np.matmul(np.hstack((seq, np.ones((seq.shape[0], 1)))), log_reg.categories_thetas[category])).reshape(-1))
        #ax.plot(seq[:,0], seq[:,1],sigmoid(np.matmul(np.hstack((seq, np.ones((seq.shape[0], 1)))), log_reg.categories_thetas[category])))

        # vals=np.arange(0, 1, 0.1)
        # x, y = np.zeros(vals.shape[0]), np.zeros(vals.shape[0])
        # ax.plot(x, y, 0.4, linestyle=":")
        ax.set_xlabel(cols_required[0])
        ax.set_ylabel(cols_required[1])
        ax.set_zlabel("predicted probabilities")
        plt.title("model plot for category {}".format(category))
        plt.show()

    # plt.plot(log_reg.loss_trend["A"], label="A")
    # plt.plot(log_reg.loss_trend["B"], label="B")
    # plt.plot(log_reg.loss_trend["C"], label="C")
    # plt.plot(log_reg.loss_trend["D"], label="D")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(log_reg.learning_rate_trend[0.1]["A"], label="0.1_A")
    # plt.plot(log_reg.learning_rate_trend[0.01]["A"], label="0.01_A")
    # plt.plot(log_reg.learning_rate_trend[0.001]["A"], label="0.001_A")
    # plt.plot(log_reg.learning_rate_trend[0.0001]["A"], label="0.0001_A")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(log_reg.learning_rate_trend[0.1]["B"], label="0.1_B")
    # plt.plot(log_reg.learning_rate_trend[0.01]["B"], label="0.01_B")
    # plt.plot(log_reg.learning_rate_trend[0.001]["B"], label="0.001_B")
    # plt.plot(log_reg.learning_rate_trend[0.0001]["B"], label="0.0001_B")
    # plt.legend()
    # plt.show()
    #
    #
    # plt.plot(log_reg.learning_rate_trend[0.1]["C"], label="0.1_C")
    # plt.plot(log_reg.learning_rate_trend[0.01]["C"], label="0.01_C")
    # plt.plot(log_reg.learning_rate_trend[0.001]["C"], label="0.001_C")
    # plt.plot(log_reg.learning_rate_trend[0.0001]["C"], label="0.0001_C")
    # plt.legend()
    # plt.show()
    #
    #
    # plt.plot(log_reg.learning_rate_trend[0.1]["D"], label="0.1_D")
    # plt.plot(log_reg.learning_rate_trend[0.01]["D"], label="0.01_D")
    # plt.plot(log_reg.learning_rate_trend[0.001]["D"], label="0.001_D")
    # plt.plot(log_reg.learning_rate_trend[0.0001]["D"], label="0.0001_D")
    # # plt.plot(log_reg.loss_trend["B"], label="B")
    # # plt.plot(log_reg.loss_trend["C"], label="C")
    # # plt.plot(log_reg.loss_trend["D"], label="D")
    # plt.legend()
    # plt.show()




    print("----------" * 5)

    print("""
    Conf - 
    Columns 
    @independent "all_mcqs_avg_n20" ,"CBSE_01"
    0.625, recall 0.607638888888889 , F_score 0.5757575757575758
    @independent "all_mcqs_avg_n20" ,"CBSE_02"
    for alpha 0.1, precision 0.75, recall 0.6770833333333333 , F_score 0.6734006734006733
    @independent "all_mcqs_avg_n20" ,"CBSE_02", "CBSE_01"
    for alpha 0.1, precision 0.7472222222222222, recall 0.673611111111111 , F_score 0.6713366899125413
    @target "LEVEL"
    Regulariaztion - Enabled - Ridge (alpha = 0.001)
    Normalization - Enabled
    Upsampling - Enabled
    
    Best model with precision of 0.75,
    recall of 0.67 and f1_score of 0.67
    with following cols = "all_mcqs_avg_n20" ,"CBSE_01", "CBSE_02", "LEVEL"
    """)
    print("----------" * 5)

    data_raw = pd.read_csv("BSOM_DataSet_for_HW2.csv")
    #cols_required = ["all_mcqs_avg_n20" ,"CBSE_02", "LEVEL"] # "all_NBME_avg_n4"
    data = data_raw[cols_required]
    # data["LEVEL"], mapping_index = pd.Series(data["LEVEL"])
    corr = data.corr()
    sns.heatmap(corr)
    plt.title("Correlation Map")
    plt.show()

    data = data.dropna()
    data = data.append(
        resample(data[data["LEVEL"] == "D"], replace=True, n_samples=46 - len(data[data["LEVEL"] == "D"]),
                 random_state=SEED), ignore_index=True)
    data = data.append(
        resample(data[data["LEVEL"] == "A"], replace=True, n_samples=46 - len(data[data["LEVEL"] == "A"]),
                 random_state=SEED), ignore_index=True)
    data = data.append(
        resample(data[data["LEVEL"] == "C"], replace=True, n_samples=46 - len(data[data["LEVEL"] == "C"]),
                 random_state=SEED), ignore_index=True)
    # data = data.append(resample(data[data["LEVEL"] == "B"], replace=True, n_samples=100-len(data[data["LEVEL"]=="B"]), random_state=123), ignore_index=True)
    print(get_majority_class_count(data["LEVEL"]))
    X_train, X_test, y_train, y_test = preprocess_data(data[cols_required],
                                                       remove_outliers=False, normalize=conf["normalization"])

    _logr = logr()
    _logr.fit(X_train, y_train)
    y_pred = _logr.predict(X_test)
    print(confusion_matrix(y_pred=y_pred.reshape(-1), y_true=y_test), end="****\n")
    print("f1_score -micro {}".format(f1_score(y_pred=y_pred, y_true=y_test, average="micro")))
    print("f1-score -macro {}".format(f1_score(y_pred=y_pred, y_true=y_test, average="macro")))
    precision, recall, _f1_score, _ = precision_recall_fscore_support(y_pred=y_pred, y_true=y_test, average="macro")
    print(precision, recall, _f1_score)
    print(precision_recall_fscore_support(y_pred=y_pred, y_true=y_test))

    print("--------check-------" * 5)
    f1_score_trend=[]
    precision_trend=[]
    recall_trend=[]
    f1_score_trend_1 = []
    precision_trend_1 = []
    recall_trend_1 = []
    alphas= [0.00001, 0.0001, 0.001, 0.005 ,0.01 ,0.1, 0]
    for alpha in alphas:
        _logr =  logr(random_state=SEED)
        _logr.fit(X_train, y_train)
        y_pred=_logr.predict(X_test)
        precision, recall, _f1_score, _ = precision_recall_fscore_support(y_pred=y_pred, y_true=y_test,
                                                                          average="macro")
        f1_score_trend_1.append(_f1_score)
        precision_trend_1.append(precision)
        recall_trend_1.append(recall)

        log_reg = LogisiticRegression(X_train, y_train, learning_rate=0.1, n_iter=10000, tolerance=1e-4,
                                      regularization=conf["regularization"], regularization_alpha=alpha).fit()
        y_pred, y_pred_labels = log_reg.predict(X_test)
        print(confusion_matrix(y_pred=y_pred_labels.reshape(-1), y_true=y_test), end="****\n")
        print("f1_score -micro {}".format(f1_score(y_pred=y_pred_labels, y_true=y_test, average="micro")))
        print("f1-score -macro {}".format(f1_score(y_pred=y_pred_labels, y_true=y_test, average="macro")))
        precision, recall, _f1_score, _ = precision_recall_fscore_support(y_pred=y_pred_labels, y_true=y_test,
                                                                          average="macro")
        print("for alpha {}, precision {}, recall {} , F_score {}".format(alpha, precision, recall, _f1_score))
        f1_score_trend.append(_f1_score)
        precision_trend.append(precision)
        recall_trend.append(recall)
        print(precision_recall_fscore_support(y_pred=y_pred_labels, y_true=y_test))
        #print(log_reg.categories_thetas)





    plt.plot([str(alpha) for alpha in alphas], f1_score_trend, label="f1_macro")
    plt.plot([str(alpha) for alpha in alphas], precision_trend, label= "Precision")
    plt.plot([str(alpha) for alpha in alphas], recall_trend, label= "Recall")
    #plt.xlim(alphas[0], alphas[-1])
    #print(f1_score_trend)
    #plt.axvline(x=alphas.index(0.01), linestyle=":")
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("score")
    plt.title("Regularization constant vs F1-SCORE, Precision and Recall")

    # plt.plot([str(alpha) for alpha in alphas], f1_score_trend_1, label="f1_macro_sklearn")
    # plt.plot([str(alpha) for alpha in alphas], precision_trend_1, label="Precision_sklearn")
    # plt.plot([str(alpha) for alpha in alphas], recall_trend_1, label="Recall_sklearn")
    # plt.axvline(x=alphas.index(0.01), linestyle=":")
    # plt.legend()
    # plt.xlabel("alpha")
    # plt.ylabel("score")
    # plt.title("Regularization constant vs F1-SCORE, Precision and Recall")
    print("copy these ",
          "===========")
    print(f1_score_trend)
    print(precision_trend)
    print(recall_trend)
    print("sklearn's")
    print(f1_score_trend_1)
    print(precision_trend_1)
    print(recall_trend_1)

    # sklearn
    # 's
    # [0.6734006734006733, 0.6734006734006733, 0.6734006734006733, 0.6734006734006733, 0.6734006734006733,
    #  0.6734006734006733, 0.6734006734006733]
    # [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    # [0.6770833333333333, 0.6770833333333333, 0.6770833333333333, 0.6770833333333333, 0.6770833333333333,
    # #scaled
    # [0.6734006734006733, 0.6734006734006733, 0.6734006734006733, 0.6734006734006733, 0.6734006734006733,
    #  0.6734006734006733, 0.6734006734006733]
    # [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    # [0.6770833333333333, 0.6770833333333333, 0.6770833333333333, 0.6770833333333333, 0.6770833333333333,
    #  0.6770833333333333, 0.6770833333333333]
    #  0.6770833333333333, 0.6770833333333333]
    f1_score_trend = [0.6734006734006733, 0.6734006734006733, 0.6734006734006733, 0.6734006734006733, 0.6734006734006733,
     0.6734006734006733, 0.6734006734006733]
    precision_trend = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    recall_trend = [0.6770833333333333, 0.6770833333333333, 0.6770833333333333, 0.6770833333333333, 0.6770833333333333,
     0.6770833333333333, 0.6770833333333333]
    f1_score_trend_1 =[0.5573762838468721, 0.5573762838468721, 0.5573762838468721, 0.5573762838468721, 0.5573762838468721,0.5573762838468721, 0.5573762838468721]
    precision_trend_1 =[0.5534188034188035, 0.5534188034188035, 0.5534188034188035, 0.5534188034188035, 0.5534188034188035,0.5534188034188035, 0.5534188034188035]
    recall_trend_1 =[0.5659722222222223, 0.5659722222222223, 0.5659722222222223, 0.5659722222222223, 0.5659722222222223,0.5659722222222223, 0.5659722222222223]
    plt.plot([str(alpha) for alpha in alphas], f1_score_trend, label="f1_macro_w_scaling", marker="*")
    plt.plot([str(alpha) for alpha in alphas], precision_trend, label="precision_w_scaling", marker="o")
    plt.plot([str(alpha) for alpha in alphas], recall_trend, label="recall_w_scaling", marker="+")
    # plt.plot([str(alpha) for alpha in alphas], f1_score_trend_1, label="f1_macro_sklearn_w/o_scaling")
    # plt.plot([str(alpha) for alpha in alphas], f1_score_trend_1, label="precision_sk_w/o_scaling")
    # plt.plot([str(alpha) for alpha in alphas], f1_score_trend_1, label="recall_sk_w/o_scaling")
    plt.legend()
    plt.show()


















    # cdict = {"A": 'red', "B": 'blue', "C": 'green', "D": "orange"}
    # colors= [cdict[category] for category in y]
    #
    # thetas_A =log_reg.categories_thetas["A"]
    # thetas_B= log_reg.categories_thetas["B"]
    # thetas_C= log_reg.categories_thetas["C"]
    # thetas_D=log_reg.categories_thetas["D"]
    # X= X[:, 0]
    # #x= np.tile(np.arange(-0.5, 0.5 ,0.1, dtype=np.float64), reps=3).reshape(-1,3)
    # X = np.hstack((X, np.ones((X.shape[0], 1))))
    # y_A = sigmoid(X.dot(thetas_A[1]))
    # y_B = sigmoid(X.dot(thetas_B[1]))
    # y_C = sigmoid(X.dot(thetas_C[1]))
    # y_D = sigmoid(X.dot(thetas_D[1]))
    #

    # # dimensionality reduction
    # pca = PCA()
    # pca.fit(X)
    # print(pca.explained_variance_ratio_)
    # components = pca.transform(X)
    # print(components[0])
    # x=components[:, 0]
    # _y = components[:, 1]
    # log_reg = LogisiticRegression(components[:, 0].reshape(-1,1), y, learning_rate=0.1, n_iter=10000, tolerance=1e-4).fit()
    # plt.scatter(x,_y)
    # plt.plot(x, sigmoid(x * log_reg.categories_h_theta_x["B"]))
    # plt.show()





