# School_Performance_Prediction
### Project Motivation

The goal of the project is to design an application to predict the performance ratio of students in Math in a specific charter school in New York City. The application predicts the percentage of students that would fall into each of the performance standards - Meeting Standards, Partially Meeting Standards, and Not Meeting Standards.

### Built With:
* Python 3
* Boostrap
* Pandas
* Numpy
* Plotly
* Flask

### File Description
 #### Application File
  * charter.py #Flask file that runs the web application
#### Model
 * train_model.py #python file that trains the model for individual schools
 * forecast.p #pickle file for the saved models
#### Data Files
 * 2013-2019_Math_Test_Results_Charter_-_School #raw data from nyc open data website
 * nyc_charter_format.csv #clean version of the raw data
 * nyc_charter_grp.csv #further processed data file
 
#### Web File
* template
  |-index.html #main page of the web app
  |-result.html #the forecast and trend result page for each selected school



 
