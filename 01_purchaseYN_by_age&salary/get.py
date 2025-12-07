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
import requests
import json
# res=requests.post('http://localhost:8000/1',json.dumps({'model':'knn'}))
res=requests.post('http://localhost:8000/2',json.dumps({'age':42,'salary':50000}))
print(res.text)