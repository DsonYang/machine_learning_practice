#mix two blogs
##https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
##https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
from math import gamma
from scipy.special import boxcox1p
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew


from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import xgboost as xgb
import lightgbm as lgb


warnings.filterwarnings('ignore')
#Data analyzation
df_train = pd.read_csv(r'dataset/kaggle/house_price/train.csv')
df_test = pd.read_csv(r'dataset/kaggle/house_price/test.csv')

y_train = 0

train_ID = df_train['Id']
test_ID = df_test['Id']

df_train.drop("Id", axis=1, inplace=True)
df_test.drop("Id", axis=1, inplace=True)
# print(df_train.columns)
# print(df_train['SalePrice'].describe())


def AnalyseDataSubjectively():
    
    plt.figure(figsize=(10, 7))
    sns.distplot(df_train['SalePrice'])
    plt.savefig("output_figures/kaggle/house_price/Data_Analyze_SalePrice.pdf")

    #Plot the relationship between data
    data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
    plt.figure(figsize=(10, 7))
    data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
    plt.savefig("output_figures/kaggle/house_price/Relation_price_grlivarea.pdf")

    data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
    plt.figure(figsize=(10, 7))
    data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))
    plt.savefig("output_figures/kaggle/house_price/Relation_price_TotalBsmtSF.pdf")

    data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
    plt.figure(figsize=(10, 7))
    fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
    fig.axis(ymin = 0, ymax = 800000)
    plt.savefig("output_figures/kaggle/house_price/Relation_price_OverallQual.pdf")

    data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
    plt.figure(figsize=(10, 7))
    fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)
    plt.savefig("output_figures/kaggle/house_price/Relation_price_YearBuilt.pdf")


def AnalyseDataObjectively():
    #Correlation matrix(heatmap style)
    global df_train
    corrmat = df_train.corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.savefig("output_figures/kaggle/house_price/Heatmap.pdf")

    #Zoomed heatmap 
    plt.figure(figsize=(10, 7))
    k = 10
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale = 1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10},
        yticklabels=cols.values, xticklabels=cols.values)
    plt.savefig("output_figures/kaggle/house_price/Heatmap_zoomed.pdf")

    #scatterplot
    plt.figure(figsize=(20, 14))
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea',
            'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df_train[cols], size=2.5)
    plt.savefig("output_figures/kaggle/house_price/scatterplot.pdf")

    
    

    #Outliers - Univariate analysis
    saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
    low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
    high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
    # print('outer range (low) of the distribution:')
    # print(low_range)
    # print("\nouter range (hight) of the distribution:")
    # print(high_range)

    #Outliers - Bivariate analysis saleprice/grlivarea
    ##see the SalePrice/Grlivarea in fig[scatterplot.pdf], we find two points on the right art not following the crowd
    ##so we delete them
    # df_train.sort_values(by='GrLivArea', ascending=False)[:2]
    # df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    # df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

    df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000) &
                                      (df_train['SalePrice'] < 300000)].index)


def KeepAnalysis():
    #search for normality
    plt.figure(figsize=(10, 7))
    sns.distplot(df_train['SalePrice'], fit=norm)
    plt.savefig("output_figures/kaggle/house_price/saleprice_histogram.pdf")

    plt.figure(figsize=(10, 7))
    stats.probplot(df_train['SalePrice'], plot=plt)
    plt.savefig(
        "output_figures/kaggle/house_price/saleprice_normal_probability.pdf")

    #transform data to normal
    df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
    plt.figure(figsize=(10, 7))
    sns.distplot(df_train['SalePrice'], fit=norm)
    plt.savefig(
        "output_figures/kaggle/house_price/saleprice_histogram_after_transform.pdf")

    plt.figure(figsize=(10, 7))
    stats.probplot(df_train['SalePrice'], plot=plt)
    plt.savefig(
        "output_figures/kaggle/house_price/saleprice_normal_probability_after_transform.pdf")

    #same with GrLivArea
    plt.figure(figsize=(10, 7))
    sns.distplot(df_train['GrLivArea'], fit=norm)
    plt.savefig("output_figures/kaggle/house_price/grlivarea_histogram.pdf")

    plt.figure(figsize=(10, 7))
    stats.probplot(df_train['GrLivArea'], plot=plt)
    plt.savefig(
        "output_figures/kaggle/house_price/grlivarea_normal_probability.pdf")

    #transform data to normal
    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
    plt.figure(figsize=(10, 7))
    sns.distplot(df_train['GrLivArea'], fit=norm)
    plt.savefig(
        "output_figures/kaggle/house_price/grlivarea_histogram_after_transform.pdf")

    plt.figure(figsize=(10, 7))
    stats.probplot(df_train['GrLivArea'], plot=plt)
    plt.savefig(
        "output_figures/kaggle/house_price/grlivarea_normal_probability_after_transform.pdf")

    #check TotalBsmtSF
    plt.figure(figsize=(10, 7))
    sns.distplot(df_train['TotalBsmtSF'], fit=norm)
    plt.savefig("output_figures/kaggle/house_price/TotalBsmtSF_histogram.pdf")

    plt.figure(figsize=(10, 7))
    stats.probplot(df_train['TotalBsmtSF'], plot=plt)
    plt.savefig(
        "output_figures/kaggle/house_price/TotalBsmtSF_normal_probability.pdf")
    #to apply a log transformation here, we'll do a log transformation to all the non-zero observations, ignoring those with value zero
    #create a new column for new variable, if area>0 it gets 1
    # df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
    # df_train['HasBsmt'] = 0
    # df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    # #transform data
    # df_train.loc[df_train['HasBsmt'] == 1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
    
    plt.figure(figsize=(10, 7))
    sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
    plt.savefig("output_figures/kaggle/house_price/TotalBsmtSF_histogram_after_transform.pdf")

    plt.figure(figsize=(10, 7))
    stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
    plt.savefig(
        "output_figures/kaggle/house_price/TotalBsmtSF_normal_probability_after_transform.pdf")

    #check the homoscedasticity
    plt.figure(figsize=(10, 7))
    plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
    plt.savefig(
        "output_figures/kaggle/house_price/Relation_price_grlivarea_after_transform.pdf")

    plt.figure(figsize=(10, 7))
    plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'],df_train[df_train['TotalBsmtSF'] > 0]['SalePrice'])
    plt.savefig(
        "output_figures/kaggle/house_price/Relation_price_TotalBsmtSF_after_transform.pdf")




def FeaturesEngineering():
    global df_train, df_test, y_train
    y_train = df_train.SalePrice.values
    n_train = df_train.shape[0]
    n_test = df_test.shape[0]
    df_all = pd.concat((df_train, df_test)).reset_index(drop=True)
    # print("df_train size is : {}".format(df_train.shape))
    # print("df_test size is : {}".format(df_test.shape))
    df_all.drop(['SalePrice'], axis=1, inplace=True)
    # print("all_data size is : {}".format(df_all.shape))
    #Check missing data
    total = df_all.isnull().sum().sort_values(ascending=False)
    percent = (df_all.isnull().sum() / df_all.isnull().count()
               ).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1,
                             keys=['Total', 'Percent'])
    # print(missing_data.head(20))

    #deleting the missing data
    # df_train = df_train.drop(
    #     (missing_data[missing_data['Total'] > 1]).index, 1)
    # df_train = df_train.drop(
    #     df_train.loc[df_train['Electrical'].isnull()].index)
    # print(df_train.isnull().sum().max())
    
    #inputing missing value
    ##since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood, 
    ##we can fill in missing values by the median LogFrontage of the neighborhood
    df_all['LotFrontage'] = df_all.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))
    
    for col in ('FireplaceQu','Fence', 'Alley', 'MiscFeature', 'PoolQC', 'GarageType', 'GarageFinish', 
                'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
        df_all[col] = df_all[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'MasVnrArea',
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        df_all[col] = df_all[col].fillna(0)
    
    ##MSZoning: RL is the most common value, same as some features, we fill in with the most common calue
    for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):
        df_all[col] = df_all[col].fillna(df_all[col].mode()[0])

    ##Utilities: almost all records are "AllPub",except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling.
    df_all = df_all.drop(['Utilities'], axis=1)

    ##Functional: data description says NA means typical
    df_all['Functional'] = df_all['Functional'].fillna('Typ')


    #check are there any missing values
    total = df_all.isnull().sum().sort_values(ascending=False)
    percent = (df_all.isnull().sum() / df_all.isnull().count()
               ).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1,
                             keys=['Total', 'Percent'])
    # print(missing_data.head(20))

    #Transforming some numerical cariables that are really categorical
    df_all['MSSubClass'] = df_all['MSSubClass'].apply(str)
    df_all['OverallCond'] = df_all['OverallCond'].astype(str)
    df_all['YrSold'] = df_all['YrSold'].astype(str)
    df_all['MoSold'] = df_all['MoSold'].astype(str)

    #label encoding some categorical variables that may contain information in ordering set
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(df_all[c].values))
        df_all[c] = lbl.transform(list(df_all[c].values))
    # print('Shape all_data: {}'.format(df_all.shape))
    #area related features are very important to determine house prices, add new column
    df_all['TotalSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF']+df_all['2ndFlrSF']
    
    #skewed features
    numeric_feats = df_all.dtypes[df_all.dtypes != "object"].index
    ##check the skew of all numerical features
    skewed_feats = df_all[numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew':skewed_feats})
    # print(skewness.shape[0])
    # print(skewness.head(10))

    skewness = skewness[abs(skewness) > 0.75]
    
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        df_all[feat] = boxcox1p(df_all[feat], lam)
    
    df_all = pd.get_dummies(df_all)
    df_train = df_all[:n_train]
    df_test = df_all[:n_test]

    print(df_train.shape)
    print(df_test.shape)


#modelling
#basic models
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(
    alpha=0.0005, l1_ratio=0.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817,
                                n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state=7, nthread=-1)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                learning_rate=0.05, n_estimators=720,
                                max_bin=55, bagging_fraction=0.8,
                                bagging_freq=5, feature_fraction=0.2319,
                                feature_fraction_seed=9, bagging_seed=9,
                                min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
##Validation function 
###corss-calidation
def rmsle_cv(model):
    kf = KFold(5, shuffle=True,
               random_state=42).get_n_splits(df_train.values)
    
    rmse = np.sqrt(-cross_val_score(model, df_train.values, y_train,
                   scoring="neg_mean_squared_error", cv=kf))
    return(rmse)

def BasicModels():
    #models scores
    
    for score in (rmsle_cv(lasso), rmsle_cv(ENet), rmsle_cv(KRR), rmsle_cv(GBoost), rmsle_cv(model_xgb), rmsle_cv(model_lgb)):
        print("\nscore:{:.4f} ({:.4f})\n".format(score.mean(), score.std()))
       
#Stacking model
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

def AveragedModels():
    averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
    score = rmsle_cv(averaged_models)
    print("Averaged base models score:{:.4f} ({:.4f})\n".format(score.mean(), score.std()))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, 1] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

def StackingModels():
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model= lasso)
    score = rmsle_cv(stacked_averaged_models)
    print("Stacking Averaged models score: {:.4f} ({:.4f})".format(
        score.mean(), score.std()))


#Ensembling StackedRegressor, XGBoost, LightGBM
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def FinalTrainAndPrediction():
    #stacked regressor
    stacked_averaged_models = StackingAveragedModels(
        base_models=(ENet, GBoost, KRR), meta_model=lasso)
    stacked_averaged_models.fit(df_train.values, y_train)
    stacked_train_pred = stacked_averaged_models.predict(df_train.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(df_test.values))
    print(rmsle(y_train, stacked_train_pred))

    #xgboost
    model_xgb.fit(df_train, y_train)
    xgb_train_pred = model_xgb.predict(df_train)
    xgb_pred = np.expm1(model_xgb.predict(df_test))
    print(rmsle(y_train, xgb_train_pred))

    #light gbm
    model_lgb.fit(df_train, y_train)
    lgb_train_pred = model_lgb.predict(df_train)
    lgb_pred = np.expm1(model_lgb.predict(df_test.values))
    print(rmsle(y_train, lgb_train_pred))

    #RMSE on the entire Train data when averaging
    print('RMSE score on train data:')
    print(rmsle(y_train, stacked_train_pred*0.7+xgb_train_pred*0.15+lgb_train_pred*0.15))

    #Ensemble prediction
    ensemble = stacked_pred*0.7+xgb_pred*0.15+lgb_pred*0.15
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    sub.to_csv('kaggle/house_price/submission.csv', index=False)

AnalyseDataSubjectively()
AnalyseDataObjectively()
KeepAnalysis()
FeaturesEngineering()
# BasicModels()
# AveragedModels()
# StackingModels()
FinalTrainAndPrediction()

