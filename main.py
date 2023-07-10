from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    c = request.args.get('c', 0, type=int)
    d = request.args.get('d', 0, type=int)
    e = request.args.get('e', 0, type=int)
    f = request.args.get('f', 0, type=int)
    g = request.args.get('g', 0, type=int)
    print("###########################################################")
    df = pd.read_csv('weatherHistory.csv')
    del df['Loud Cover']
    map_dict = {"rain": 0, "snow": 1, "nan": 2}
    reverse_map_dict = {0: "rain", 1: "snow", 2: "nan"}
    df['Precip Type'] = df['Precip Type'].map(map_dict)
    df['Precip Type'] = df['Precip Type'].fillna(0)
    df['Precip Type'] = df['Precip Type'].astype(int)
    y = df['Precip Type']
    x = df.drop(['Precip Type', 'Formatted Date', 'Summary', 'Daily Summary'], axis=1)
    from sklearn.model_selection import train_test_split
    print("###########################################################")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    ypred = classifier.predict(X_test)
    results = pd.DataFrame(ypred)
    results.columns = ['results']
    results['results'] = results['results'].map(reverse_map_dict)
    results.head()
    sample=pd.DataFrame({'Temperature (C)': [a], 'Apparent Temperature (C)':[b],'Humidity':[c],'Wind Speed (km/h)':[d],'Wind Bearing (degrees)':[e],'Visibility (km)':[f],'Pressure (millibars)':[g]})
    x = classifier.predict(sample)
    print("###########################################################")
    if x[0] ==1:
        result = "Snow"
    else:
        result = "Rain"

    return jsonify(result=result)

@app.route('/')
def index():
    return render_template('index.html')


app.run(host='0.0.0.0')
