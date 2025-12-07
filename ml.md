#### ssh google cloud

#### remove line
    ctr + k
#### remove file
    rm sc.pickle
#### open and save file
    nano test.py
    paste code
    control + x
    y
    enter
#### download file
    sudo apt install wget
    wget tfkerras_dataset.h5
    wget https://github.com/kaedenakashima/ML/blob/867eca5462f81e6615c98bc619345cbe448a0c8d/practice01/asset/sc.pickle

#### load file
  x getting error

    import pickle
    from flask import Flask, request
    import numpy as np
    f1=pickle.load(open('classifier.pickle','rb'))
    f2=pickle.load(open('sc.pickle','rb'))
    ls 

#### create server folder
    mkdir test
    cd test
    python3 -m venv venv
    source venv/bin/activate
    pip install numpy pandas scikit-learn flask requests
    rm -f filename
    python3 test.py

#### run server with docker
    docker run -t --rmm -p 8501:8501