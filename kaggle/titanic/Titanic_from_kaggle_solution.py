#https: // www.kaggle.com/aaron874/titanic-beginner-solution/notebook
from math import log
import numpy as np
import pandas as pd

#Graphic Components
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.base.optimizer import Optimizer

import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
#Import Data

Titanic_train = pd.read_csv(
    r'/work/ML/machine_learning_practice/dataset/kaggle/titanic/train.csv')
# print(Titanic_train.head(5))

Titanic_predict = pd.read_csv(
    r'/work/ML/machine_learning_practice/dataset/kaggle/titanic/test.csv')
# print(Titanic_predict.head(5))

# print(Titanic_train.describe())

# print(Titanic_train.describe(include=['O']))

# print('Train Set:', Titanic_train.shape)
# print('Test Set:', Titanic_predict.shape)

# print(Titanic_train.dtypes)
# print(Titanic_train.nunique())

# for i in Titanic_train.columns:
#     print(f"{i}:", Titanic_train[i].isna().sum())

Titanic_train = Titanic_train.drop(columns=['PassengerId','Ticket','Name'])
Titanic_train = Titanic_train.drop(columns=['Cabin'])

# #Exploratory Analysis

# #Kendall's coefficient
Titanic_train.select_dtypes(include=['float64', 'int64'])

Correlation_df = Titanic_train.copy()
num_col = Titanic_train.select_dtypes(include=['float64', 'int64']).columns
# print(num_col)
category_col = Titanic_train.select_dtypes(include=['object']).columns
# print(category_col)
Correlation_df_category = pd.DataFrame(
    OneHotEncoder(drop = 'first').fit(Correlation_df[category_col]).transform(Correlation_df[category_col]).toarray(), 
    columns=OneHotEncoder(drop='first').fit(Correlation_df[category_col]).get_feature_names(Correlation_df[category_col].columns))

# print(OneHotEncoder(drop='first').fit(
#     Correlation_df[category_col]).transform(Correlation_df[category_col]).toarray())
# print(OneHotEncoder().fit(
#     Correlation_df[category_col]).transform(Correlation_df[category_col]).toarray())
# print(OneHotEncoder(drop='first').fit(Correlation_df[category_col]).transform(
#     Correlation_df[category_col]).toarray())
# print(OneHotEncoder(drop='first').fit(Correlation_df[category_col]).get_feature_names(
#     Correlation_df[category_col].columns))

Correlation_df_category = Correlation_df_category[Correlation_df_category['Embarked_nan'] == 0.0].drop(columns= ['Embarked_nan'])
# print(Correlation_df_category)

#Normalization
Correlation_df_num = pd.DataFrame(StandardScaler().fit(Correlation_df[num_col]).transform(Correlation_df[num_col]), columns=Correlation_df[num_col].columns)
# print(Correlation_df_num)

Correlation_df = pd.concat([Correlation_df_num, Correlation_df_category], axis=1)
# print(Correlation_df.head(20))
Correlation_df = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(Correlation_df), columns=Correlation_df.columns)
# print(Correlation_df.head(20))
Correlation_df_corr = Correlation_df.corr(method='kendall', min_periods=1)
# print(Correlation_df_corr)
# plt.figure(figsize=(10, 7))
# sns.heatmap(Correlation_df_corr)
# plt.savefig("output_figures/kaggle/titanic/Correlation_df_corr.pdf")

Biserial_df = pd.DataFrame()
for i in Correlation_df.drop(columns = ['Survived']).columns:
    Biserial_df = Biserial_df.append(pd.DataFrame({
        'Variable' : i,
        'Correlation': stats.pointbiserialr(Correlation_df[i], Correlation_df['Survived']).correlation,
        'p-Value': round(stats.pointbiserialr(Correlation_df[i], Correlation_df['Survived']).pvalue, 3)
    },
    index = [0]))

Biserial_df.index = Biserial_df['Variable']

# print(Biserial_df)
# plt.figure(figsize=(10, 7))
# sns.heatmap(Biserial_df[['Correlation']])
# plt.savefig("output_figures/kaggle/titanic/Biserial_df.pdf")

#Data Cleansing for Model
Titanic_train_x = Titanic_train.drop(columns=['Survived'])
Titanic_train_y = Titanic_train['Survived']

#Split to Numeric and Categorical Data for Normalization
num_col = Titanic_train_x.select_dtypes(include=['int64','float64']).columns
category_col = Titanic_train_x.select_dtypes(include=['object']).columns

#Normalization
Titanic_train_num = pd.DataFrame(StandardScaler().fit(Titanic_train_x[num_col]).transform(Titanic_train_x[num_col]),columns=Titanic_train_x[num_col].columns)
Titanic_train_category = pd.DataFrame(OneHotEncoder(drop='first').fit(Titanic_train_x[category_col]).transform(Titanic_train_x[category_col]).toarray(),
                                      columns=OneHotEncoder(drop='first').fit(Titanic_train_x[category_col]).get_feature_names(Titanic_train_x[category_col].columns))

# print(Titanic_train_category)
# print(Titanic_train_category[Titanic_train_category['Embarked_nan'] == 0.0])
# print(Titanic_train_category[Titanic_train_category['Embarked_nan'] == 0.0].drop(
#     columns=['Embarked_nan']))

Titanic_train_category = Titanic_train_category[Titanic_train_category['Embarked_nan'] == 0.0].drop(columns=['Embarked_nan'])        
Titanic_train_x = pd.concat([Titanic_train_num, Titanic_train_category], axis=1)
#Fill NA Data
Titanic_train_x = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(Titanic_train_x), columns=Titanic_train_x.columns)
Titanic_train_x, Titanic_test_x, Titanic_train_y, Titanic_test_y = train_test_split(Titanic_train_x, Titanic_train_y, test_size=0.3, random_state=42)

# print(Titanic_train_x)
# print(Titanic_test_x)
# print(Titanic_train_y)
# print(Titanic_test_y)

#Statistical Inference & Feature Selection
logistic_regression_model = sm.Logit(Titanic_train_y, Titanic_train_x).fit()
# print(logistic_regression_model.summary())

#Select the Useful Feature
SFS = SequentialFeatureSelector(LogisticRegression(random_state=0),
        direction= 'backward',
        scoring='roc_auc',
        cv = 5,
        n_features_to_select=3).fit(Titanic_train_x, Titanic_train_y)

SFS_Results = pd.DataFrame({
    'Variable':Titanic_train_x.columns,
    'Chosen': SFS.get_support()})
# print(SFS_Results)

SFS_Variable = SFS_Results[SFS_Results['Chosen'] == True]['Variable']
logistic_regression_model = sm.Logit(Titanic_train_y, Titanic_train_x[SFS_Variable]).fit()
# print(logistic_regression_model.summary())

#Model Selection
Performance_df = pd.DataFrame(columns=['Model', 'Feature Selection', 'Accuracy', 'Log Loss', 'ROC'])


def RandomForest():
    ##Random Forest for All Variables
    alphas = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(
        Titanic_train_x, Titanic_train_y)['ccp_alphas']
    random_parameters = {
        'n_estimators': [10, 100, 1000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 100, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balbanced', 'balanced_subsample'],
        'ccp_alpha': alphas
    }
    RFC = RandomizedSearchCV(RandomForestClassifier(),
                            param_distributions=random_parameters, n_iter=100, scoring='accuracy',
                            n_jobs=10, cv=3, verbose=2, random_state=0, return_train_score=True)
    RFC.fit(Titanic_train_x, Titanic_train_y)
    pred = RFC.predict(Titanic_test_x)
    global Performance_df
    Performance_df = Performance_df.append(
        pd.DataFrame(
            [['RFC', 'Full', accuracy_score(Titanic_test_y, pred), log_loss(
                Titanic_test_y, pred), roc_auc_score(Titanic_test_y, pred)]],
            columns=['Model', 'Feature Selection', 'Accuracy', 'Log Loss', 'ROC']
        ),
        sort=False
    )

    print(Performance_df)
    print('Confusion Matrix:\n', confusion_matrix(Titanic_test_y, pred))

    plt.figure(figsize=(10, 7))
    plot_roc_curve(RFC, Titanic_test_x, Titanic_test_y)
    plt.savefig("output_figures/kaggle/titanic/ROC_Curve_AllParameters.pdf")


    ###Test the Performance of Best Parameters
    Best_Parameter = RFC.best_params_
    RFC = RandomForestClassifier(
        n_estimators=Best_Parameter['n_estimators'],
        criterion=Best_Parameter['criterion'],
        max_depth=Best_Parameter['max_depth'],
        max_features=Best_Parameter['max_features'],
        bootstrap=Best_Parameter['bootstrap'],
        class_weight=Best_Parameter['class_weight'],
        ccp_alpha=Best_Parameter['ccp_alpha'],
    )
    RFC.fit(Titanic_train_x, Titanic_train_y)

    pred = RFC.predict(Titanic_test_x)
    Performance_df = Performance_df.append(
        pd.DataFrame(
            [['RFC', 'Full', accuracy_score(Titanic_test_y, pred), log_loss(
                Titanic_test_y, pred), roc_auc_score(Titanic_test_y, pred)]],
            columns=['Model', 'Feature Selection', 'Accuracy', 'Log Loss', 'ROC']
        ),
        sort=False
    )

    print(Performance_df)
    print('Confusion Matrix:\n', confusion_matrix(Titanic_test_y, pred))

    plt.figure(figsize=(10, 7))
    plot_roc_curve(RFC, Titanic_test_x, Titanic_test_y)
    plt.savefig("output_figures/kaggle/titanic/ROC_Curve_BestParameters.pdf")

    ###Random Forest for Significant Variables
    ####Find the Alpha
    alphas = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(
        Titanic_train_x[SFS_Variable], Titanic_train_y)['ccp_alphas']
    random_parameters = {
        'n_estimators': [10, 100, 1000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 100, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample'],
        'ccp_alpha': alphas
    }
    RFC = RandomizedSearchCV(
        RandomForestClassifier(),
        param_distributions=random_parameters,
        n_iter=100,
        scoring='accuracy',
        n_jobs=10,
        cv=3,
        verbose=2,
        random_state=0,
        return_train_score=True
    )
    RFC.fit(Titanic_train_x[SFS_Variable], Titanic_train_y)

    Best_Parameter = RFC.best_params_

    ####Test the Preforamnce of Best Parameters
    RFC = RandomForestClassifier(n_estimators=Best_Parameter['n_estimators'],
                                criterion=Best_Parameter['criterion'],
                                max_depth=Best_Parameter['max_depth'],
                                max_features=Best_Parameter['max_features'],
                                bootstrap=Best_Parameter['bootstrap'],
                                class_weight=Best_Parameter['class_weight'],
                                ccp_alpha=Best_Parameter['ccp_alpha']
                                )
    RFC.fit(Titanic_train_x[SFS_Variable], Titanic_train_y)
    #####Validation
    pred = RFC.predict(Titanic_test_x[SFS_Variable])
    Performance_df = Performance_df.append(pd.DataFrame([['RFC', 'Selected', accuracy_score(Titanic_test_y, pred), log_loss(Titanic_test_y, pred), roc_auc_score(Titanic_test_y, pred)]],
                                                        columns=['Model', 'Feature Selection', 'Accuracy', 'Log Loss', 'ROC']), sort=False)

    print('Accuracy:', accuracy_score(Titanic_test_y, pred))
    print('Log Loss:', log_loss(Titanic_test_y, pred))
    print('ROC Accuracy:', roc_auc_score(Titanic_test_y, pred))
    print('Confusion Matrix:\n',
        confusion_matrix(Titanic_test_y, pred))

    plt.figure(figsize=(10, 7))
    plot_roc_curve(RFC, Titanic_test_x[SFS_Variable], Titanic_test_y)
    plt.savefig("output_figures/kaggle/titanic/ROC_Curve_SignificantParameters.pdf")


# RandomForest()

def SVM():
    #Support Vector Machine
    random_parameters = {
        'C': stats.expon(scale=100),
        'gamma': stats.expon(scale=.1),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'class_weight': ['balanced', None]
    }
    #Randomized Cross Validation for Hyperparameters Tuning
    Support_Vector = RandomizedSearchCV(
        SVC(),
        param_distributions=random_parameters,
        n_iter=100,
        scoring='accuracy',
        n_jobs=10,
        cv=3,
        verbose=2,
        random_state=0,
        return_train_score=True
    )
    Support_Vector.fit(Titanic_train_x, Titanic_train_y)

    Best_Parameter = Support_Vector.best_params_
    ##Test the Performance of Best Parameters
    Support_Vector = SVC(C=Best_Parameter["C"], gamma=Best_Parameter["gamma"],
                        kernel=Best_Parameter["kernel"], class_weight=Best_Parameter["class_weight"])
    Support_Vector.fit(Titanic_train_x, Titanic_train_y)

    ###Validation
    pred = Support_Vector.predict(Titanic_test_x)
    global Performance_df
    Performance_df = Performance_df.append(pd.DataFrame(
        [['SVC', 'Full', accuracy_score(Titanic_test_y, pred), log_loss(
            Titanic_test_y, pred), roc_auc_score(Titanic_test_y, pred)]],
        columns=['Model', 'Feature Selection', 'Accuracy', 'Log Loss', 'ROC']
    ), sort=False)

    print('Accuracy:', accuracy_score(Titanic_test_y, pred))
    print('Log Loss:', log_loss(Titanic_test_y, pred))
    print('ROC Accuracy:', roc_auc_score(Titanic_test_y, pred))
    print('Confusion Matrix:\n',
        confusion_matrix(Titanic_test_y, pred))

    plt.figure(figsize=(10, 7))
    plot_roc_curve(Support_Vector, Titanic_test_x, Titanic_test_y)
    plt.savefig("output_figures/kaggle/titanic/SVM.pdf")

    ##SVC Significant Variables
    random_parameters = {
        'C': stats.expon(scale=100),
        'gamma': stats.expon(scale=1),
        'kernel': ['linear', 'ploy', 'rbf', 'sigmoid'],
        'class_weight': ['balbanced', None]
    }
    Support_Vector = RandomizedSearchCV(
        SVC(),
        param_distributions=random_parameters,
        n_iter=100,
        scoring='accuracy',
        n_jobs=10,
        cv=3,
        verbose=2,
        random_state=0,
        return_train_score=True
    )
    Support_Vector.fit(Titanic_train_x[SFS_Variable], Titanic_train_y)

    Best_Parameter = Support_Vector.best_params_
    ##Test the Performance of Best Parameters
    Support_Vector = SVC(C=Best_Parameter["C"], gamma=Best_Parameter["gamma"],
                        kernel=Best_Parameter["kernel"], class_weight=Best_Parameter["class_weight"])
    Support_Vector.fit(Titanic_train_x[SFS_Variable], Titanic_train_y)

    ###Validation
    pred = Support_Vector.predict(Titanic_test_x[SFS_Variable])
    Performance_df = Performance_df.append(pd.DataFrame(
        [['SVC', 'Selected', accuracy_score(Titanic_test_y, pred), log_loss(
            Titanic_test_y, pred), roc_auc_score(Titanic_test_y, pred)]],
        columns=['Model', 'Feature Selection', 'Accuracy', 'Log Loss', 'ROC']
    ), sort=False)

    print('Accuracy:', accuracy_score(Titanic_test_y, pred))
    print('Log Loss:', log_loss(Titanic_test_y, pred))
    print('ROC Accuracy:', roc_auc_score(Titanic_test_y, pred))
    print('Confusion Matrix:\n',
        confusion_matrix(Titanic_test_y, pred))

    plt.figure(figsize=(10, 7))
    plot_roc_curve(Support_Vector, Titanic_test_x[SFS_Variable], Titanic_test_y)
    plt.savefig("output_figures/kaggle/titanic/SVM_SFS.pdf")

# SVM()

def XGBoost():
    XGB_Train_df = xgb.DMatrix(Titanic_train_x, label=Titanic_train_y)
    XGB_Test_df = xgb.DMatrix(Titanic_test_x, label=Titanic_test_y)

    parameters = {
        'max_depth': 6,
        'min_child_weight':1,
        'eta':0.3,
        'subsample':0.7,
        'colsample_bytree':1,
        'objective':'binary:hinge'
    }

    Best_Params = {
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': 0.3,
        'subsample': 0.7,
        'colsample_bytree': 1,
        'objective': 'binary:hinge'
    }

    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(9,12)
        for min_child_weight in range(5,8)
    ]

    max_auc = 0
    best_params = None
    for max_depth,min_child_weight in gridsearch_params:
        print("CV with  max_depth={}, min_child_weight={}".format(max_depth,min_child_weight))
        parameters["max_depth"]=max_depth
        parameters["min_child_weight"]=min_child_weight

        cv_results = xgb.cv(
            parameters,
            XGB_Train_df,
            num_boost_round = 999,
            seed=42,
            nfold=5,
            metrics={'auc'},
            early_stopping_rounds=10
        )
        mean_auc=cv_results['test-auc-mean'].max()
        boost_rounds=cv_results['test-auc-mean'].argmax()
        print("\tROC AUC {} for {} rounds".format(mean_auc, boost_rounds))
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = (max_depth, min_child_weight)

        Best_Params['max_depth'],Best_Params['min_child_weight'] = best_params
        print("Best params:{},{},ROC AUC:{}".format(best_params[0], best_params[1], max_auc))

    #Cross Validation for Learning Rate
    max_auc = 0
    best_params = None
    for eta in [.3,.2,.1,.05,.01,.005]:
        print("CV with eta={}".format(eta))
        parameters['eta'] = eta
        cv_results = xgb.cv(
            parameters,
            XGB_Train_df,
            num_boost_round=999,
            seed=42,
            nfold=5,
            metrics=['auc'],
            early_stopping_rounds=10
        )
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].argmax()
        print("\tROC AUC {} for {} rounds".format(mean_auc, boost_rounds))
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = eta

        Best_Params['eta'] = best_params
        print("Best params:{},ROC AUC:{}".format(best_params,max_auc))
    best_model = xgb.train(
        Best_Params,
        XGB_Train_df,
        num_boost_round=999,
        evals=[(XGB_Test_df, "Test")],
        early_stopping_rounds = 10
    )
    #Validation
    pred = best_model.predict(XGB_Test_df).astype(int)
    global Performance_df
    Performance_df = Performance_df.append(
        pd.DataFrame(
            [['XGBC', 'Full', accuracy_score(Titanic_test_y, pred), log_loss(Titanic_test_y, pred), roc_auc_score(Titanic_test_y, pred)]],
            columns = ['Model', 'Feature Selection', 'Accuracy', 'Log Loss', 'ROC']
            ),
        sort = False
        )
    print('Accuracy:', accuracy_score(Titanic_test_y, pred))
    print('Log Loss:', log_loss(Titanic_test_y, pred))
    print('ROC Accuracy:', roc_auc_score(Titanic_test_y, pred))
    print('Confusion Matrix:\n',confusion_matrix(Titanic_test_y,pred))


# XGBoost()


def DeepLearning():
    early_stopping = EarlyStopping(
        min_delta = 0.001,
        patience = 20,
        restore_best_weights = True,
    )

    DL_Model = keras.Sequential([
        layers.Dense(256, input_shape=[8]),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    DL_Model.compile(
        # optimizer='adam',
        # loss='Binary_Crossentropy',
        # metrics='binary_accuracy'
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()]
    )

    History = DL_Model.fit(
        Titanic_train_x, Titanic_train_y,
        validation_data = (Titanic_test_x, Titanic_test_y),
        callbacks = [early_stopping],
        batch_size = 100,
        epochs=1000,
        verbose=0
    )

    history_df = pd.DataFrame(History.history)

    plt.figure(figsize=(10, 7))
    history_df.loc[:,['loss', 'val_loss']].plot()
    plt.savefig("output_figures/kaggle/titanic/DeepLearning_LOSS.pdf")
    plt.figure(figsize=(10, 7))
    history_df.loc[:,['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.savefig("output_figures/kaggle/titanic/DeepLearning_ACCU.pdf")

    #Validation
    pred = DL_Model.predict(Titanic_test_x).round(0).astype(int)
    global Performance_df
    Performance_df = Performance_df.append(pd.DataFrame(
        [['Tensorflow', 'Full', accuracy_score(Titanic_test_y,pred), log_loss(Titanic_test_y,pred), roc_auc_score(Titanic_test_y, pred)]],
        columns=['Model', 'Feature Selection', 'Accuracy', 'Log Loss', 'ROC']
    ), sort=False)
    
    print('Accuracy:', accuracy_score(Titanic_test_y, pred))
    print('Log Loss:', log_loss(Titanic_test_y, pred))
    print('ROC Accuracy:', roc_auc_score(Titanic_test_y, pred))
    print('Confusion Matrix:\n', confusion_matrix(Titanic_test_y, pred))


# DeepLearning()

def DeepLearningSelected():
    early_stopping = EarlyStopping(
        min_delta=0.001,
        patience=20,
        restore_best_weights=True,
    )

    DL_Model = keras.Sequential([
        layers.Dense(256, input_shape=[3]),
        layers.Dense(1, activation='sigmoid')
    ])

    DL_Model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()]
    )

    History = DL_Model.fit(
        Titanic_train_x[SFS_Variable], Titanic_train_y,
        validation_data=(Titanic_test_x[SFS_Variable], Titanic_test_y),
        callbacks=[early_stopping],
        batch_size=100,
        epochs=10000,
        verbose=0,
    )

    history_df = pd.DataFrame(History.history)
    plt.figure(figsize=(10, 7))
    history_df.loc[:, ['loss', 'val_loss']].plot()
    plt.savefig("output_figures/kaggle/titanic/DeepLearning_Selected_LOSS.pdf")
    plt.figure(figsize=(10, 7))
    history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.savefig("output_figures/kaggle/titanic/DeepLearning_Selected_ACCU.pdf")

    pred=DL_Model.predict(Titanic_test_x[SFS_Variable]).round(0).astype(int)
    global Performance_df
    Performance_df = Performance_df.append(pd.DataFrame(
        [['Tensorflow', 'Selected', accuracy_score(Titanic_test_y,pred), log_loss(Titanic_test_y,pred),roc_auc_score(Titanic_test_y,pred)]],
        columns=['Model', 'Feature Selection', 'Accuracy', 'Log Loss', 'ROC']
        ), sort=False)
    
    print('Accuracy:', accuracy_score(Titanic_test_y, pred))
    print('Log Loss:', log_loss(Titanic_test_y, pred))
    print('ROC Accuracy:', roc_auc_score(Titanic_test_y, pred))
    print('Confusion Matrix:\n', confusion_matrix(Titanic_test_y, pred))


# DeepLearningSelected()

def Review():
    RandomForest()
    SVM()
    XGBoost()
    DeepLearning()
    DeepLearningSelected()
    global Performance_df
    print(Performance_df)

Review()
