from model import *
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/recommended',methods = ['POST'])
def recommend():
    if request.method == "POST":
        data = request.get_json()   # Get data from request
        res = data                  # Feed data into function, then assign to res
        return jsonify(res)         # Return a json containing recommendations to front-end
    