import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

# Use pd consistently as the alias for pandas
car = pd.read_csv("cleaned.csv")
model = pickle.load(open("LinearRegressionModel.pkl",'rb'))
@app.route('/')
def index():
    # Sort the unique values directly without the need for an additional 'sorted' call
    companies = car['company'].unique().tolist()

    car_models = car['name'].unique().tolist()
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique().tolist()

    return render_template('index.html', companies=sorted(companies), car_models=sorted(car_models), years=years, fuel_types=sorted(fuel_types))

@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get("car_model")
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    print(company,car_model,year,fuel_type,kms_driven)
    prediction = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    print(prediction)
    return str(np.round(prediction[0]))



if __name__ == '__main__':
    app.run(debug=True)
