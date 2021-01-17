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
def build_model(train):
    '''
    INPUT: (pandas dataframe) train dataframe
    OUTPUT: model_fit
    TASK: build forecast model for each school
    '''
    #data = np.log(train).diff().dropna()
    #model = VAR(data)
    model = VAR(endog=train)
    model_fit = model.fit(1)
    
    return model_fit

#function makes prediction using the model
def predict(validation, model):
    '''
    INPUT: (pandas dataframe) validation set
           (timeseries model) model
    OUTPUT: prediction
    TASK: use model to make prediction
    '''
    prediction = model.forecast(model.y, steps=len(validation))
    
    #covert prediction to dataframe
    pred = pd.DataFrame(index = range(0,len(prediction)),columns=[cols])
    
    for i in range(0,3):
        for j in range(0, len(prediction)):
            pred.iloc[j][i] = prediction[j][i]
            
    return pred

#This function evaluates the model
def evaluate_model(pred, valid):
    '''
    INPUT: (pandas dataframe) prediction by model
            validation dataframe
    OUTPUT: root mean squared score for the three target variables
    TASK: evaluate the performance of the model
    '''
    for index,i in enumerate(valid.columns):
        print(i)
        print('rmse:{}'.format(math.sqrt(mean_squared_error(valid.iloc[:,index],pred.iloc[:,index])))) 
            

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
        preds = predict(v_data,models[key])
        print("The rmse score for school with id {}".format(key))
        evaluate_model(preds,v_data)
        print(' ')

#store the models
models = build_allmodels(school_ids)
#save models
pickle.dump(models, open("forecast.p","wb"))
#returns root mean squared error of each school's model for the prediction
evaluate_allmodels(models)












  

