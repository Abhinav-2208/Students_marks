from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

with open('students_marks.pkl', 'rb') as file:
    model = pickle.load(file)

df=pd.DataFrame()

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df

    input_features = [float(x) for x in request.form.values()]
    # Convert the input features to a numpy array
    input_array = np.array(input_features)

    if input_features[0] < 0 or input_features[0] > 24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24')

    # Make the prediction using the loaded model
    output = model.predict(input_array.reshape(1, -1))[0].round(2)

    df = pd.concat([df, pd.DataFrame({'Study_hours': input_array, 'predicted_output': [output]})], ignore_index=True)
    print(df)
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html', prediction_text="You will get [{}%] marks when you study [{}] hours per day".format(output, int(input_array[0])))


app.run(debug=True)
