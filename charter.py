#Load libraries
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
import numpy as np
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings("ignore")
import pickle
import json
from flask import Flask
from flask import render_template, request, jsonify
import plotly
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

#load the model
forecast_models = pickle.load(open("forecast.p", "rb"))
#load dataset
nyc_charter = pd.read_csv('nyc_charter_format.csv')
#select useful features
nyc_charter_grp = nyc_charter[['DBN','Year','Not_Meeting_Pct','Partially_Meeting_Pct','Meeting_Pct']]

#group new dataset by DBN and Year
nyc_charter2 = nyc_charter_grp.groupby(['DBN','Year']).mean()

#reset dataframe
nyc_reset = nyc_charter2.reset_index()

def get_school_ids(models):
    '''
        INPUT: model
        OUTPUT: (list) school ids of school with saved models
        TASK: get the ids of schools from the model dict
    '''
    school_ids = []
    for k,v in models.items():
        school_ids.append(k)

    return school_ids

def get_school_names():
    '''
    OUTPUT: (list) school names
    TASK: get unique school names
    '''
    school_ids = get_school_ids(forecast_models)
    school_names = np.unique(nyc_charter.loc[nyc_charter['DBN'].isin(school_ids)]['School_Name']).tolist()

    return school_names

def get_school_id(name):
    '''
    INPUT: (str) school name
    OUTPUT: (str) school id
    TASK: get unique school id
    '''
    #school_ids = get_school_ids(forecast_models)
    school_id = np.unique(nyc_charter.loc[nyc_charter['School_Name']==name]['DBN']).tolist()

    return school_id

#function subsets dataset to dataset of individual schools
def get_school_df(school_id,df = nyc_charter2):
    '''
    INPUT: (pandas dataframe) nyc_charter,
            (str) school_id
    OUTPUT: (pandas dataframe) school_df
    TASK: subset dataframe to consist of only one school
    '''
    school_df = df.loc[school_id]
    #add noise to data to add variance for VAR to work optimally
    #create noise with the same dimension as the subset data
    mu, sigma = 0,0.1
    nrow,ncol = school_df.shape[0],school_df.shape[1]
    noise = np.random.normal(mu, sigma,[nrow,ncol])
    school_df = school_df + noise

    return school_df 

   
@app.route('/')
@app.route('/index')
def index():
    fig4= px.bar(nyc_reset, x="Year", y="Not_Meeting_Pct",
        labels={"Not_Meeting_Pct":"Not Meeting Learning Standards"})

    fig5= px.bar(nyc_reset, x="Year", y="Partially_Meeting_Pct",
        labels={"Partially_Meeting_Pct":"Partially t Meeting Learning Standards"})

    fig6= px.bar(nyc_reset, x="Year", y="Meeting_Pct",
        labels={"Meeting_Pct":"Meeting Learning Standards"})

    #encode plotly graphs in JSON
    fig4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    fig5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    fig6JSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)



    return render_template('index.html', plot4=fig4JSON, plot5=fig5JSON,plot6=fig6JSON)

@app.route('/result',methods=['GET'])
def result():
    #get query and store
    query = request.args.get('query')
    #get the school id
    school_id = get_school_id(query)
    school_id = school_id[0]
    #get each school data
    school_data = get_school_df(school_id)
    #get colunm names
    cols = school_data.columns
    #fit model to data
    model = VAR(school_data)
    model_fit = model.fit(1)
    #forecast using model for next 3 years
    yhat = model_fit.forecast(model_fit.y, steps=4)
    #convert predictions from numpy array to dataframe
    pred_years = ['2019','2020','2021','2022']
    yhat_df = pd.DataFrame(data=yhat, index=pred_years,columns=cols)
    yhat_df.index = pd.to_datetime(yhat_df.index)

    #Visualize the treand and forecast of performance levels
    fig = px.line(school_data, x=school_data.index,y='Not_Meeting_Pct',
        title='Math Performance Trend vs Forecast',
        labels={'Not_Meeting_Pct':'Not Meeting Learning Standards(%)'})
    fig.add_trace(go.Scatter(
        x = yhat_df.index,
        y=yhat_df['Not_Meeting_Pct'],
        mode='lines',
        name='Forecast')
    )

    fig2 = px.line(school_data, x=school_data.index,y='Partially_Meeting_Pct',
        labels={'Partially_Meeting_Pct':'Partially Meeting Learning Standards(%)'})
    fig2.add_trace(go.Scatter(
        x = yhat_df.index,
        y=yhat_df['Partially_Meeting_Pct'],
        mode='lines',
        name='Forecast')
    )

    fig3 = px.line(school_data, x=school_data.index,y='Meeting_Pct',
        labels={'Meeting_Pct':'Meeting Learning Standards(%)'})
    fig3.add_trace(go.Scatter(
        x = yhat_df.index,
        y=yhat_df['Meeting_Pct'],
        mode='lines',
        name='Forecast')
    )


    #encode plotly graphs in JSON
    figJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    fig2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    fig3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    


    #return render_template('result.html',query=query,plot1=figJSON)
    return render_template('result.html',query=query,plot1=figJSON,plot2=fig2JSON,plot3=fig3JSON)

    
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()





    

#print(nyc_charter.head())


    
