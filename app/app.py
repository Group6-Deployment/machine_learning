#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
from joblib import load
# importing decimal module
import decimal

app = Flask(__name__)

#default page of our web-app
@app.route('/')
def Home():
    return render_template("index.html")

#To predict a winner in a soccer match
@app.route('/index',methods=['POST'])#route to winner
def predict():
 if request.method == 'POST':
    model = pickle.load(open('model.pkl', 'rb'))
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.round(np.array(int_features))]
    prediction = model.predict_proba(final_features)[:,1][0]

    #getting a winner
    #if (prediction<=0.5):
            #result="Team A wins"
    #elif (prediction>0.5):
            #result="Team B wins"
    #else:
            #result="Draw"
    # output = round(prediction[0], 2)
    # print(prediction) 
    return render_template('index.html',prediction= prediction)
    #return result


#default page for Insurance Premium
@app.route('/insurance', methods =['GET'])
def insurance():
        return render_template("insurance.html")

#To predict Insurance Premiums new prices for new customers
@app.route('/insurance',methods =['POST'])#prices for customer
def insurance_predict():
        if request.method == 'POST':
                model_insurence = pickle.load(open('insurance.pkl', 'rb'))
                data1 = request.form['a']
        data2 = request.form['b']
        data3 = request.form['c']
        data4 = request.form['d']
        data5 = request.form['e']
        arr = np.array([[data1, data2, data3, data4, data5]])
        pred = model_insurence.predict(arr)
        return render_template('insurance.html', data=pred)

#default  page for sales
@app.route('/sales', methods =['GET'])
def sales():
        return render_template("sales.html")

#To predict number of sales
@app.route('/sales',methods =['POST'])#number of sales
def sale_predict():
        if request.method == 'POST':
                model_sales = pickle.load(open('sales.pkl', 'rb'))
                data1 = request.form["MarketID"]
        data2 = request.form["MarketSize"]
        data3 = request.form["LocationID"]
        data4 = request.form["AgeOfStore"]
        data5 = request.form["Promotion"]
        data6 = request.form["week"]
        final_data = np.array([[data1,data2,data3,data4,data5,data6]])
        pred1 = model_sales.predict(final_data)
        return render_template('sales.html', sales=pred1)

@app.route('/dibetes', methods =['GET'])
def home():
    return render_template("dibetes.html")

@app.route('/dibetes',methods=['POST'])
def dibetes_predict():
    if request.method == 'POST':
        model_dibetes = pickle.load(open('dibetes.pkl', 'rb'))  
    #int_features=[ request.form.values()] 
    int_features1 = request.form['ID']
    int_features2 = request.form['Gender']
    int_features3 = request.form['Years']
    int_features4 = request.form['income']
    int_features5 = request.form['arm']
    int_features6 = request.form['sagg']
    int_features7 = request.form['grip']
    int_features8 = request.form['breast']

    final_features = np.array([[int_features1,int_features2,int_features3,int_features4,int_features5,int_features6,int_features7,int_features8]])
    final_features = np.array(final_features, dtype=float)
    predition1 = model_dibetes.predict(final_features)
 
    
    return render_template('dibetes.html', data=predition1)
    
        

if __name__ == "__main__":
    app.run(debug=True)