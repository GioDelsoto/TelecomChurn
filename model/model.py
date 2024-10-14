import requests
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from model.config import Config
import os
from sklearn.base import BaseEstimator, TransformerMixin



class WOEMappingTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.woe_dict = {}
    
    def fit(self, X, y):
        # If X is a single column, convert to DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # If X is a DataFrame, process each column
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                self.woe_dict[col] = self._calculate_woe(X[col], y)
        else:
            # If X is a Series (single column), handle it directly
            self.woe_dict = {self._calculate_woe(X, y)}  # Store the result for single column

        return self

    def _calculate_woe(self, X_col, y):
        # Implement your WOE calculation logic here
        df = pd.DataFrame({'X': X_col, 'y': y})
        woe_dict = {}
        
        # Calculate total events and non-events
        total_events = df['y'].sum()
        total_non_events = len(df) - total_events
        
        # Calculate WOE for each category
        for category in df['X'].unique():
            subset = df[df['X'] == category]
            count_events = subset['y'].sum()
            count_non_events = len(subset) - count_events
            
            # Avoid division by zero
            if count_events == 0:
                woe = -np.inf  # or a very large negative number
            elif count_non_events == 0:
                woe = np.inf  # or a very large positive number
            else:
                woe = np.log((count_events / total_events) / (count_non_events / total_non_events))
                
            woe_dict[category] = woe
        
        return woe_dict
    
    def transform(self, X):
        # Transform X based on WOE mapping
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # If X is a DataFrame, transform each column
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in X.columns:
                X_transformed[col] = X[col].map(self.woe_dict.get(col, {})).fillna(0)  # Use 0 for unseen categories
        else:
            # If X is a Series, transform it directly
            return X.map(self.woe_dict).fillna(0)

        return X_transformed

    


class Model():
    
    def __init__(self, steps = []):
        
     
        
        self.continuous_vars = ['change_mou','change_rev','months','hnd_price','eqpdays',
                            'custcare_Mean','totmrc_Mean','rev_Mean','mou_Mean',
                            'ovrmou_Mean','ovrrev_Mean','inonemin_Mean',
                            'mou_cvce_Mean','mou_rvce_Mean','owylis_vce_Mean',
                            'mouowylisv_Mean','mou_peav_Mean','complete_Mean',
                            'totcalls','adjqty','avgrev','avgmou','avgqty','avg3qty',
                            'avg3rev','avg6mou','mou_price','ovr_price','drop_blk_percentage','mouiwylisv_Mean',
                            ]

        
        
        self.woe_features = ['crclscod', 'area']
        self.one_hot = ['asl_flag', 'dualband', 'refurb_new', 'creditcd']
        self.target = ['churn']
        self.woe_area_dict = {}        
        self.woe_crclscod_dict = {}        
        
        
        mapper = DataFrameMapper(
              [ (categorical_col, LabelBinarizer()) for categorical_col in self.one_hot  ] +
              
              [('area', WOEMappingTransformer()), 
               ('crclscod', WOEMappingTransformer())]
            
            )
        
        model_parameters = Config.model_parameters
        estimator = xgb.XGBClassifier( random_state=42, **model_parameters)
        
        
        self.pipeline = Pipeline(
                            [  ("mapper", mapper),
                               ("estimator", estimator)  ]
                            )
        return
    
    
    def fit_model(self, X_fit =[], y_fit=[], save_model = False):
        
        '''
            Fits the model using the provided data or data fetched from a database.
            
            Parameters:
            - X_fit: DataFrame containing the features for training the model.
            - y_fit: Series or DataFrame containing the target values for training the model.
            - save_model: Boolean flag indicating whether to save the trained model. If True, the model will be saved to the specified path.
            
            - The provided X_fit and y_fit will be used to train the model.
            - The model will be trained and evaluated using the provided data.
            
            The model's pipeline is fitted with the features from X_fit, and the selected features are stored in self.data_columns.
        '''
        
        
           
        if len(X_fit)==0 or len(y_fit)==0:
            print("X_fit or y_fit have no elements.")
        
        
        else:
            #Select only important features
            
            X_fit['ovr_price'] = X_fit['ovrrev_Mean']/(X_fit['ovrmou_Mean']+1)
            X_fit['mou_price'] = X_fit['rev_Mean']/(X_fit['mou_Mean']+1)  #Minimum value of mou_Mean = 0.. 0.25..         
            X_fit['drop_blk_percentage'] = (X_fit['drop_blk_Mean'])/(X_fit['attempt_Mean']+1)
            
            X_fit = X_fit[self.continuous_vars + self.woe_features + self.one_hot]

            print(X_fit)
            
            self.data_columns = X_fit.columns
            self.pipeline.fit(X_fit, y_fit)
            print("Model fitted with success!")
            
            
            if save_model == True:
                
                file_name = 'model.pkl'
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
                model_path = os.path.join(base_dir, file_name)
                
                self.save_model(model_path)
                print(f"Model saved in {model_path}")


    
    def evaluate_model(self, X_eval, y_eval): 
        
        X_eval['ovr_price'] = X_eval['ovrrev_Mean']/(X_eval['ovrmou_Mean']+1)
        X_eval['mou_price'] = X_eval['rev_Mean']/(X_eval['mou_Mean']+1)  #Minimum value of mou_Mean = 0.. 0.25..         
        X_eval['drop_blk_percentage'] = (X_eval['drop_blk_Mean'])/(X_eval['attempt_Mean']+1)
                
        #Select only important features
        X_eval = X_eval[self.continuous_vars + self.woe_features + self.one_hot]

        cv_scores = cross_val_score(self.pipeline, X_eval, y_eval, cv=5, scoring='f1')
        print(f'F1 Score (Cross-Validation): {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')
    
        y_predict = self.pipeline.predict(X_eval)
        conf_matrix = confusion_matrix(y_eval, y_predict)
        # Plota a Matriz de Confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
                
                
    
    def predict_churn(self, X_input):
        
        X_input['ovr_price'] = X_input['ovrrev_Mean']/(X_input['ovrmou_Mean']+1)
        X_input['mou_price'] = X_input['rev_Mean']/(X_input['mou_Mean']+1)  #Minimum value of mou_Mean = 0.. 0.25..         
        X_input['drop_blk_percentage'] = (X_input['drop_blk_Mean'])/(X_input['attempt_Mean']+1)
        
        X_input = X_input[self.continuous_vars + self.woe_features + self.one_hot]
        
        #Select only important features

        probs = self.pipeline.predict_proba(X_input)
        print(probs)
        return probs
    
    
    def save_model(self, filename):
        
        #os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

