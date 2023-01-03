import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
We have given a set of observations linking the amount of rain per year and field
productivity in a dry area somewhere in Central America.

'''

''' 

The file contains two columns: Amount of Precipitations (in mm per year)
and the Productivity Coefficient. 

'''

# Read the CSV datafile into Python numpy array 

dataframe = pd.read_csv("inputdata2.CSV")
a = np.array(dataframe['Rainfall'])
a = a.reshape((-1,1))
b = np.array(dataframe['Productivity'])

# Function for plotting the data as a two-dimensional scatter plot
def generate_Scatterplt():
    
    plt.figure(dpi = 720, figsize = (8,6))
    
    # Add the Labels and Titles foe the 2D Scatter plot
    plt.xlabel ("Amount of Precipitations (in mm per year)")
    plt.ylabel("Productivity Coefficient")
    plt.title("Amount of Precipitations (in mm per year) VS Productivity Coefficient")
    
    #Plot the scatter plot for the given data
    plt.scatter(a,b, marker="^")
    plt.legend()
    plt.tight_layout()
    plt.savefig("scatterplot.png")  
    plt.show()

# Function for creating the Linear Regression model for the input data    
def generate_linear_regression_model():
    plt.figure(dpi = 720, figsize = (8,6))
    x_rainfall = 280
    
    # Create the Linear Regression Model 
    model = LinearRegression()
    model.fit(a,b)
    score = model.score(a,b)
    
    # Predict the Productivity Coefficient for the Amount of Precipitation.
    predicted_value = model.predict([[x_rainfall]])
    intercept = model.intercept_
    coef = model.coef_
    yfit = intercept + coef*a
    
    print(predicted_value)
    
    # Add the Labels and Titles foe the 2D Scatter plot
    plt.xlabel ("Amount of Precipitations (in mm per year)")
    plt.ylabel("Productivity Coefficient")
    plt.title("Amount of Precipitations (in mm per year) VS Productivity Coefficient")
    
    # Plot the Original Data and Linear Regression model on the same plot 
    plt.plot(a,yfit,label = 'Linear Regression Model')
    plt.scatter(a, b,marker="^",label='Original Data')
    # Show the Predicted Productivity Coefficient on the plot
    plt.scatter(x_rainfall,predicted_value,label='Predicted Value',linewidths=(5))
    plt.text(287,0.096,f'{predicted_value}',color='red')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Linear Model.png")  
    plt.show()

# Call the functions to generate respective plots for the Data    
generate_Scatterplt()
generate_linear_regression_model()

