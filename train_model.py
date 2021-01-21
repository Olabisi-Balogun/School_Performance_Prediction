#Load libraries
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#from IPython.display import display
import pandas as pd
import numpy as np
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error
#from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.api import VAR
import pickle
import warnings
warnings.filterwarnings("ignore")


#%matplotlib inline
#load datasets
#nyc_charter2 = pd.read_csv('nyc_charter_grp.csv')
nyc_charter = pd.read_csv('nyc_charter_format.csv')

#select useful features
nyc_charter_grp = nyc_charter[['DBN','Year','Not_Meeting_Pct','Partially_Meeting_Pct','Meeting_Pct']]

#group new dataset by DBN and Year
nyc_charter2 = nyc_charter_grp.groupby(['DBN','Year']).mean()


#store school ids and school names
school_ids = np.unique(nyc_charter['DBN'])
school_names = np.unique(nyc_charter['School_Name'])
#print(nyc_charter.head())
#print(nyc_charter2.head())
#print(nyc_charter.columns)

#function subsets dataset to dataset of individual schools
def get_school_df(school_id,df = nyc_charter2):
    '''
    INPUT: (pandas dataframe) nyc_charter,
            (str) school_id
    OUTPUT: (pandas dataframe) school_df
    TASK: subset dataframe to consist of only one school
    '''
    school_df = df.loc[school_id]
    return school_df 

#test get_school_df; useful to get column names for later use
school = get_school_df('84K125')
cols = school.columns

#function splits the data into train set and validation set
def train_valid_split(df):
    '''
    INPUT:(pandas dataframe) individual school dataframe
    OUTPUT: (pandas dataframe) train_df, valid_df
    TASK: split df into train set and validation set;
           validation set has records of most current year
    '''
    #add noise to data to add variance for VAR to work optimally
    #create noise with the same dimension as the subset data
    mu, sigma = 0,0.1
    nrow,ncol = df.shape[0],df.shape[1]
    noise = np.random.normal(mu, sigma,[nrow,ncol])
    df = df + noise
    cut_off = int(df.shape[0]*0.75)
    #split data into train set and validation set
    train_df = df[0:cut_off]
    valid_df = df[cut_off:]
    
    return train_df, valid_df

#function builds the model
#function builds the model
def build_model(train):
    '''
    INPUT: (pandas dataframe) train dataframe
    OUTPUT: model_fit
    TASK: build forecast model for each school
    '''
    train_diff = train.diff().dropna()
    #model = VAR(train)
    model = VAR(train_diff[['Not_Meeting_Pct','Partially_Meeting_Pct','Meeting_Pct']])
    model_fit = model.fit(2)
    
    return model_fit

#function makes prediction using the model
def predict(train, validation, model):
    '''
    INPUT: (pandas dataframe) validation set
           (timeseries model) model
    OUTPUT: prediction
    TASK: use model to make prediction
    '''
    train_diff = train.diff().dropna()
    lag_order = model.k_ar
    forecast_input = train_diff.values[-lag_order:]
    #prediction = model.forecast(model.y, steps=len(validation))
    prediction = model.forecast(forecast_input, steps=len(validation))
    
    #covert prediction to dataframe
    pred = pd.DataFrame(prediction,index = validation.index,columns=cols+'_1d')
    #pred.columns = ['Not_Meeting_Pct_Forecast','Partially_Meeting_Pct_Forecast','Meeting_Pct_Forecast']
    
    #forecast_orig = pred.copy()
    for col in cols:
        pred[str(col)+'_forecast'] = train[col].iloc[-1] + pred[str(col)+'_1d']
    
    pred.loc[:,['Not_Meeting_Pct_forecast','Partially_Meeting_Pct_forecast','Meeting_Pct_forecast']]
    pred = pred[['Not_Meeting_Pct_forecast','Partially_Meeting_Pct_forecast','Meeting_Pct_forecast']]
   

            
    #return forecast_orig
    return pred

#This function evaluates the model
def evaluate_model(pred, valid):
    '''
    INPUT: (pandas dataframe) prediction by model
            validation dataframe
    OUTPUT: root mean squared error(rmse) for the three target variables, mean absolute error(mae)
    TASK: evaluate the performance of the model
    '''
    mae = np.mean(np.abs(pred-valid))
    rmse = np.sqrt(np.mean((pred-valid)**2))
    
    return ({'mae':mae,'rmse':rmse})

    #function builds models for individual schools and stores them in a dictionary
def build_allmodels(sch_ids):
    '''
    INPUT: (list) the ids of the schools
    OUTPUT: (dict) a dictionary mapping the school id to the school model
    TASK: build models for all the schools in the dataset
    '''
    all_models = dict()
    for ind in sch_ids:
        school_data = get_school_df(ind)
        train_data, validation_data = train_valid_split(school_data)
        try:
            school_model = build_model(train_data)
            all_models[ind]=school_model
        except ValueError: #raised if train_data is empty
            pass
    
    return all_models

#function evaluates the models
def evaluate_allmodels(models):
    '''
    INPUT: (dict) models for schools in the dataset
    OUTPUT: (float) root mean squared error of each model
    TASK: compute the rmse for the model
    
    '''
    for key, value in models.items():
        sub_data = get_school_df(key)
        t_data, v_data = train_valid_split(sub_data)
        #get prediction for each school's model
        preds = predict(t_data,v_data,models[key])
        print('Model Accuracy for school with id {}'.format(key))
        print('Forecast Accuracy of: Not_Meeting_Pct')
        accuracy = evaluate_model(preds['Not_Meeting_Pct_forecast'].values, v_data['Not_Meeting_Pct'])
        for k,val in accuracy.items():
            print(k,': ',round(val,4))
            print(' ')    
        print('Forecast Accuracy of: Partially_Meeting_Pct')
        accurcay = evaluate_model(preds['Partially_Meeting_Pct_forecast'].values, v_data['Partially_Meeting_Pct'])
        for k,val in accuracy.items():
            print(k,': ',round(val,4))
            print(' ')
        print('Forecast Accuracy of: Meeting_Pct')
        accuracy = evaluate_model(preds['Meeting_Pct_forecast'].values, v_data['Meeting_Pct'])
        for k,val in accuracy.items():
            print(k,': ',round(val,4))

#store the models
models = build_allmodels(school_ids)
#save models
pickle.dump(models, open("forecast.p","wb"))
#returns root mean squared error of each school's model for the prediction
evaluate_allmodels(models)












  

