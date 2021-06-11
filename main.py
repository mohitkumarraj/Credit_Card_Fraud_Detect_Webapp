from flask import Flask, request, jsonify, render_template
import pickle
import random
import pandas as pd
import json
from datetime import datetime
from flask_mail import Mail
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

with open('config.json', 'r') as c:
    params = json.load(c)["params"]

app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT='465',
    MAIL_USE_SSL=True,
    MAIL_USERNAME=params['gmail-user'],
    MAIL_PASSWORD=params['gmail-password']
)

mail = Mail(app)  # object

model = pickle.load(open('lg.pkl', 'rb'))  # Linear model classifier

model = pickle.load(open('d_model.pkl', 'rb'))  # Decision Tree classifier model classifier
data = pd.read_csv('creditcard.csv', na_values=['??', '???'])

pca = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
       'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

fraud_trans = data[data['Class'] == 1]

scaling = StandardScaler()


@app.route('/')
def home():
    return render_template('index.html', params=params)

#this is the change happened here
#this is the Second change happened here
@app.route('/result2', methods=['POST'])
def result2():
    tid = random.randint(345548616575, 785641364312)
    tid = str(tid)
    features = []
    form = [float(x) for x in request.form.values()]
    features.append(form[0])

    for i in pca:   # random value
        features.append(random.choice(data[i]))

    # =================taking a fraud transaction as input to check its performance

    # features = features + fraud_trans.iloc[0, 1:29].tolist()  # first fraud tran [time=406,amount=0]
    # features = features + fraud_trans.iloc[2,1:29].tolist()  # Second fraud trans [time=4920,239.93]
    # features = features + fraud_trans.iloc[623, 1:29].tolist()  # Second fraud trans [time=472,amount=529.00]
    # features = features + fraud_trans.iloc[279863, 1:29].tolist()  # Second fraud trans [time=169142,amount=390]
    # features = features + fraud_trans.iloc[280149, 1:29].tolist()  # Second fraud trans [time=169351,amount=77.89000]
    # features = features + fraud_trans.iloc[281144, 1:29].tolist()  # Second fraud trans [time=169966,amount=245]
    # features = features + fraud_trans.iloc[281674, 1:29].tolist()  # Second fraud trans [time=170348,amount=42.530000]

    # ========================================================================================
    features.append(form[1])
    final_feature = [np.array(features)]
    prediction = model.predict(final_feature)

    if prediction == 1:
        now = datetime.now()
        time = now.strftime("%m/%d/%Y,%H:%M:%S")
        mail.send_message('Alert message from credit card Fraud Detection System',
                          sender=params['gmail-user'],
                          recipients=[params['gmail-user']],
                          body="Dear Sir/ma'am\n\n" + params['msg'] + time + "\n" + params['with'] + tid + params[
                              'restrict'] + params['report'])

        return render_template('index.html', prediction_text=params['fraud'], params=params)
    else:
        return render_template('index.html', prediction_text=params['normal'], params=params)
        # return render_template('index.html', prediction_text='Price of car will be {}'.format(final_feature) + '\u20B9')

       # return render_template('index.html', prediction_text='Passed Features {}'.format(final_feature) + '\u20B9')

app.run(debug=True)

# print(feature)
# final_feature=[np.array(feature)]
# prediction=model.predict(final_feature)
# price=2.718281828459**prediction
# price=np.round(price,2)

# features_arra=np.array(features).reshape(1,-1)
#
# features=scaling.fit_transform(features_arra)
# print(features)


# to print the features
# list_string = map(str, final_feature)

# strlist=listToString(list_string)
# return render_template('index.html', prediction_text={}.__format__(strlist), params=params)
