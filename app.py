#pip install tensorflow


import os
import pandas as pd
#from flask import Flask, request, jsonify, render_template
import streamlit as st

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import optimizers
from sklearn import metrics
import tensorflow as tf
import numpy as np
import gc


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



# Path to the uploaded CSV file
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
predictions = dict()



# Streamlit app function
def streamlit_app():
     st.title('CSV File Processing')

     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

     if uploaded_file is not None:
         df = pd.read_csv(uploaded_file)
         st.write(df)

# Flask routes
@app.route('/')
def index():
    #scaleNfillna(df)
    return render_template("index.html") #return st.markdown(get_file_content_as_string("templates/index.html"), unsafe_allow_html=True)



def createModel():
    print('construction  of NN model ')
    ann = Sequential()
    ann.add(Dense(64, input_dim=255, activation='relu'))
    ann.add(Dense(32, activation='relu'))
    ann.add(BatchNormalization())
    ann.add(Dense(16, activation='relu'))
    ann.add(BatchNormalization())
    ann.add(Dense(8, activation='relu'))
    ann.add(BatchNormalization())
    ann.add(Dense(1, activation='sigmoid'))
    learning_rate=0.009 #0.01
    opt = optimizers.Adam(learning_rate = learning_rate)
    ann.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall') ])
    ann.summary()
    return ann

model = createModel()
model.load_weights('creditDef.h5')

def scaleNfillna(df):
    df.replace([np.inf, -np.inf], np.nan,inplace=True)
    df.fillna(0,inplace=True)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return(df)

def processInput(data):
    global model
    
    feats =[x for x in list(data) if x not in ['SK_ID_CURR','TARGET']]
    test = data
    test[feats] = scaleNfillna(test[feats])
    # test = test.loc[test['TARGET'].notnull(),feats].values
    print(test)
    test = test.drop(['SK_ID_CURR','TARGET'], axis  =1)
    print(test.shape)
    test = test.to_numpy(dtype='int')
    # test.reshape(-1,1)
    print('numpy', test.shape)
    preds = model.predict(test)
    print(preds)
    predictions = {}
    for idx, pred in enumerate(preds):
        if pred > 0.5:
            predictions["Person "+ str(idx+1)] = "Default"
        else:
            predictions["Person "+ str(idx+1)] = "No Default"
    return predictions


@app.route('/upload', methods=['POST'])
def upload():
    global model
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            data = pd.read_csv('./uploads/'+file.filename)
            return render_template('result.html', data=processInput(data))
        

@app.route('/process_sample', methods=['POST'])
def processSample():
    data = pd.read_csv('sampleData.csv')
    return render_template('result.html', data=processInput(data))

def get_file_content_as_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == '__main__':
    app.run(debug=True)
