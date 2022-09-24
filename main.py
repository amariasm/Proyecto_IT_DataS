from flask import Flask
#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/make_preds',methods = ['GET','POST'])
def make_preds(to_predict_list):
  
  sq_mt_built=int(to_predict_list[0])
  n_bathrooms=int(to_predict_list[1]), 
  n_rooms=int(to_predict_list[2])
  has_lift=to_predict_list[3],
  house_type_id=to_predict_list[4]

  sq_mt_built = 100
  n_bathrooms = 2
  n_rooms = 4
  has_lift = True
  house_type_id= 'HouseType 1: Pisos'  

  import pickle
  import pandas as pd

  # Load Files
  encoder_fit = pd.read_pickle("generador app/app/encoder.pickle")
  rf_reg_fit = pd.read_pickle("generador app/app/model.pickle")

 
  # Create df
  x_pred = pd.DataFrame(
    [[sq_mt_built, n_bathrooms, n_rooms, bool(has_lift), house_type_id]],
    columns = ['sq_mt_built', 'n_bathrooms', 'n_rooms', 'has_lift', 'house_type_id'])

  # One hot encoding
  encoded_data_pred = pd.DataFrame( encoder_fit.transform(x_pred['house_type_id']),columns = encoder_fit.classes_.tolist()) 

  # Build final df
  x_pred_transf = pd.concat([x_pred.reset_index(), encoded_data_pred], axis = 1).drop(['house_type_id', 'index'], axis = 1)

  pred= rf_reg_fit.predict(x_pred_transf)

  # return round(preds[0])
  return  pred[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        #try:
        to_predict_list = list(map(str,to_predict_list))
        prediction = make_preds(to_predict_list)
        #except ValueError:
         #  prediction='para no deprimirme: 123435'

        return render_template("result.html", prediction=prediction)




if __name__=="__main__":
    app.debug = True
    app.run(port=5001)
