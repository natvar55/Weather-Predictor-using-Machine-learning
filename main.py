import pandas as pd
from flask import Flask,redirect,url_for,render_template,request,send_from_directory
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import math

#Data cleaning
weather = pd.read_csv("weather.csv")
weather.dropna(inplace=True)
weather = weather[(weather['wind'] >= 0) & (weather['precipitation'] >= 0)]
weather.drop_duplicates(inplace = True)
output_mapping = {'sun': 0, 'fog': 1, 'drizzle': 2, 'rain': 3, 'snow': 4 }
weather['weather'] = weather['weather'].map(output_mapping)

#data training
predictors = ["precipitation", "temp_max", "temp_min","wind"]   
train = weather

reg = Ridge(alpha=.1)
reg.fit(train[predictors], train["weather"])

# Create a KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train[predictors], train["weather"])

dt = DecisionTreeClassifier()
dt.fit(train[predictors], train["weather"])

def predictor(precipitation,temp_max,temp_min,wind):
    predict=np.array([[precipitation,temp_max,temp_min,wind]])
    final_predict=reg.predict(predict)+knn.predict(predict)+dt.predict(predict)
    final_predict=final_predict/3
    reverse_mapping = { 0:'sunny',  1:'fog', 2:'drizzle', 3:'rain', 4:'snow' }
    final_predict=math.floor(final_predict)
    prediction = reverse_mapping[final_predict]
    return prediction

app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('homepage.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/sunny')
def sunny():
    return render_template('sunny.html')

@app.route('/snow')
def snow():
    return render_template('snow.html')

@app.route('/fog')
def fog():
    return render_template('fog.html')

@app.route('/drizzle')
def drizzle():
    return render_template('drizzle.html')

@app.route('/rain')
def rain():
    return render_template('rain.html')



@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        precipitation=float(request.form['precipitation'])
        temp_max=float(request.form['temp-max'])
        temp_min=float(request.form['temp-min'])
        wind=float(request.form['wind'])
        res=predictor(precipitation,temp_max,temp_min,wind)
        print("///////")
        print(res)
    return redirect(url_for(res))

@app.route('/weather.jpg')
def weather():
    return send_from_directory('templates', 'weather.jpg')

@app.route('/main_img.jpg')
def main_img():
    return send_from_directory('templates', 'main_img.jpg')

@app.route('/sunny.jpg')
def sunny_img():
    return send_from_directory('templates', 'sunny.jpg')

@app.route('/snow.jpg')
def snow_img():
    return send_from_directory('templates', 'snow.jpg')

@app.route('/fog.jpg')
def fog_img():
    return send_from_directory('templates', 'fog.jpg')

@app.route('/rainy.jpg')
def rainny_img():
    return send_from_directory('templates', 'rainy.jpg')

@app.route('/drizzle.jpg')
def drizzle_img():
    return send_from_directory('templates', 'drizzle.jpg')

if __name__=='__main__':
    app.run(debug=True)