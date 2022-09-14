import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline 
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import precision_score
from xgboost import XGBClassifier

def pipeline_normalizer(X:pd.DataFrame) -> pd.DataFrame:
    def find_outliers_IQR(X:pd.DataFrame) -> pd.DataFrame:
        q1=X.quantile(0.25)
        q3=X.quantile(0.75)
        IQR=q3-q1
        outliers = X[((X<(q1-1.5*IQR)) | (X>(q3+1.5*IQR)))]
        #Outliers are identified by the imbalance in their quantile differences
        return len(outliers)
    
    column_list = list(X.columns)
    binary_col = []
    robust_scaling_cat = []
    standard_scaling_cat = []
    for num in column_list:
        if X[num].max() == 1 and X[num].min() == 0:
            #test to see which are categorical data of 1's and 0's
            binary_col.append(num)
          

        if find_outliers_IQR(X[num])>100 and num not in binary_col:
            #arbitrarily set at 100 so we dont lose information in the noise--> Robust Scaler for these
            robust_scaling_cat.append(num)
            
        if find_outliers_IQR(X[num])<100 and num not in binary_col:
            standard_scaling_cat.append(num)           
            #StandardScaler for these. Normally distributed with fewer than a hundred outliers

    if len(robust_scaling_cat)+len(standard_scaling_cat)+len(binary_col) != len(X.columns):
        #test to see if you can proceed
        print('Something is wrong, Hold!')
    
    #make pipeline to standardize datasets with RobustScaler and Standardized Scaler where necessary
    preproc_robustscaler = make_pipeline(
        SimpleImputer(strategy = 'most_frequent'),
        #A lot of the nan's in the dataset correspon to zero which is the most frequent value
        RobustScaler())

    preproc_standardscaler = make_pipeline(
        SimpleImputer(strategy = 'most_frequent'),
        StandardScaler())

    categorical_scaler = make_pipeline(
        SimpleImputer(strategy="most_frequent"))
    #OneHotEncoder(handle_unknown="ignore")
    # Assuming I'm getting this data already encoded
    #if not unhash the line above and press tab
    preproc = make_column_transformer(
        (preproc_robustscaler, robust_scaling_cat),
        (preproc_standardscaler, standard_scaling_cat),
        (categorical_scaler, binary_col),    
        remainder="drop")
    
    return preproc, binary_col, robust_scaling_cat, standard_scaling_cat



def initialize_model(X, y, drop):
    pipe = pipeline_normalizer(X)[0]
    binary_col= pipeline_normalizer(X)[1]
    robust_scaling_cat = pipeline_normalizer(X)[2]
    standard_scaling_cat = pipeline_normalizer(X)[3]
    
    
    y = y.replace(to_replace=1, value = -1)\
    .replace(to_replace=0, value = 1).replace(to_replace=-1, value = 0)
    #based on predicting the non purchasers

    if drop: 
        threshold = int(input('Please Select Correlation Threshold Percentage for Columns to drop: '))
        num_corr_threshold = threshold/100
        corr_num = X[robust_scaling_cat].corr()
        corr_num_upper_triangle = corr_num.where(np.triu(np.ones(corr_num.shape),k=1).astype(np.bool)).abs()
        num_col_to_drop = [column for column in corr_num_upper_triangle.columns if any(corr_num_upper_triangle[column] > num_corr_threshold)]
        corr_num_s = X[standard_scaling_cat].corr()
        corr_num_upper_triangle_s = corr_num_s.where(np.triu(np.ones(corr_num_s.shape),k=1).astype(np.bool)).abs()
        num_col_to_drop_s = [column for column in corr_num_upper_triangle_s.columns if any(corr_num_upper_triangle_s[column] > num_corr_threshold)]
        
        print(f'You dropped {len(num_col_to_drop)+len(num_col_to_drop_s)} features')
        X = X.drop(columns = num_col_to_drop).drop(columns = num_col_to_drop_s)
        
        # robust_scaling_cat = [robust_scaling_cat.remove(num) for num in num_col_to_drop]
        # standard_scaling_cat = [standard_scaling_cat.remove(num) for num in num_col_to_drop_s]  
        for num in num_col_to_drop:
            robust_scaling_cat.remove(num)
        for num in num_col_to_drop_s:
            standard_scaling_cat.remove(num)
            
        preproc_robustscaler = make_pipeline(
            SimpleImputer(strategy = 'most_frequent'),
            #A lot of the nan's in the dataset correspon to zero which is the most frequent value
            RobustScaler())

        preproc_standardscaler = make_pipeline(
            SimpleImputer(strategy = 'most_frequent'),
            StandardScaler())

        categorical_scaler = make_pipeline(
            SimpleImputer(strategy="most_frequent"))        
        
        pipe = make_column_transformer(
                        (preproc_robustscaler, robust_scaling_cat),
                        (preproc_standardscaler, standard_scaling_cat),
                        (categorical_scaler, binary_col),    
                        remainder="drop")
        
        print('Initializing Model...')
        '''
        Initialize XGBoost Classifier
        '''
        #initialize model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)        
        #Split Training and Test Set
        X_train, X_eval, y_train, y_eval = train_test_split(X_train,y_train,random_state=42)
        #Split the traning set into a train test and an evaluation set
        X_train_preproc = pipe.fit_transform(X_train, y_train)
        #Instantiate The X_train and X_eval
        X_eval_preproc = pipe.transform(X_eval)        
        model_xgb = XGBClassifier(learning_rate = 0.1, max_depth = 10, n_estimators = 500)#Instantiating Model
        model_xgb.fit(X_train_preproc, y_train,
                    verbose=False,
                    eval_set=[(X_train_preproc, y_train), (X_eval_preproc, y_eval)],
                    early_stopping_rounds=30)
        #Fitting the model with an early stopping criteria of 20 epochs to stop from overfitting
        # print(model_xgb.evals_result())
        print('Testing Precision...')
        y_pred = model_xgb.predict(pipe.transform(X_test))
        #Set y_pred to test with y_test
        return precision_score(y_test, y_pred)
    
    if drop == False:
        print('Initializing Model...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)        
        #Split Training and Test Set
        X_train, X_eval, y_train, y_eval = train_test_split(X_train,y_train,random_state=42)
        #Split the traning set into a train test and an evaluation set
        X_train_preproc = pipe.fit_transform(X_train, y_train)
        #Instantiate The X_train and X_eval
        X_eval_preproc = pipe.transform(X_eval)        
        model_xgb = XGBClassifier(learning_rate = 0.1, max_depth = 10, n_estimators = 500)#Instantiating Model
        model_xgb.fit(X_train_preproc, y_train,
                    verbose=False,
                    eval_set=[(X_train_preproc, y_train), (X_eval_preproc, y_eval)],
                    early_stopping_rounds=30)
        #Fitting the model with an early stopping criteria of 20 epochs to stop from verfitting
        # print(model_xgb.evals_result())
        print('Testing Precision...')
        y_pred = model_xgb.predict(pipe.transform(X_test))
        #Set y_pred to test with y_test
        return precision_score(y_test, y_pred)
        
def initialize(X,y, drop = True):
    return initialize_model(X, y, drop)

# def evaluate_model(X:pd.DataFrame) -> pd.DataFrame:
    