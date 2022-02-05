# Put your flask application here
from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/", methods = ["GET","POST"])
def hello_world():
    data = request.form.get("abc")
    param_hotel = ""
    if data.get("hotel_type") == "city":
        param_hotel = "City Hotel"
    elif data.get("hotel_type") == "resort":
        param_hotel = "Resort Hotel"
    param_month = data.get("arrival_month")
    param_num = data.get("number_of_people")


    with open('api_app/exported_one_hot.pickle','rb') as fp:
        enc = pickle.load(fp)

    with open('api_app/exported_classifier.pickle','rb') as fp:
        classifier = pickle.load(fp)


    hotel_feature = enc.transform([[param_hotel]]).toarray()
    month_feature = (int(param_month) >=6) and (int(param_month) <=8)

    features = np.hstack([
    hotel_feature,
    np.array([[month_feature]]),
    np.array([[param_num]])])

    a = classifier.predict(features)
    return f"{data}"
    print(f"{data}")




