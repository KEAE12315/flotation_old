import pandas as pd
from flask import Flask
from flask_cors import CORS

from utils.load import LoadG

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.get('/data')
def get_data():
    dataset_path = 'dataset/Geolife Trajectories 1.3/Data/001/'
    dataset = enumerate(LoadG(dataset_path))

    dfs = pd.DataFrame()
    for i, df in dataset:
        df['trackID'] = i + 1
        dfs = pd.concat([dfs, df], ignore_index=True)
    dfs.reset_index(drop=True)

    return dfs.to_json(orient='records', date_format='iso')
