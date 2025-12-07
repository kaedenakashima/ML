#  npm i serve -D
#  npx serve -s build
# conda deactivate
# rm -rf venv
# python3 -m venv venv
# source venv/bin/activate
# pip
# lsof -n | grep LISTEN
# lsof -i :5000
# kill -9 <PIN>
#pip install numpy pandas scikit-learn flask requests xgboost
import requests
import json
import xgboost as xgb
import pickle
from flask import Flask, request
import numpy as np
f1=pickle.load(open('classifier.pickle','rb'))
f2=pickle.load(open('sc.pickle','rb'))
app = Flask(__name__)
# res=requests.post('http://localhost:8000/model',json.dumps({'model':'knn'}))
# print(res.text)
@app.route('/1',methods=['POST'])
def hello_word1():
    req=request.get_json(force=True)
    model_name=req['model']
    return 'The prediction is {0}'.format(model_name)
@app.route('/2',methods=['POST'])
def hello_word2():
    req=request.get_json(force=True)
    age=req['age']
    salary=req['salary']
    print(age)
    print(salary)
    pred=f1.predict(f2.transform(np.array([[age,salary]])))
    return 'The prediction is {}'.format(pred)
if __name__=='__main__':
    app.run(port=8000,debug=True)