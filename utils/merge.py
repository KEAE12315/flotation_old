import pandas as pd
from load import *


def toDf(fps):
    def readOne(f, s, e):
        df = pd.read_json(f)
        df.set_index(["index"], inplace=True)
        df = df[(df.trackID >= s) & (df.trackID <= e)]
        df.datetime=pd.to_datetime(df.datetime,format='%Y-%m-%d,%H:%M:%S')
        return df

    for fse in fps:
        yield readOne(fse['fp'], fse['start'], fse['end'])



if __name__ == "__main__":
    fps = [
        {'fp': 'dataset/GTMarker/001/markerRaw/1-35.json',
         'start': 1,
         'end': 35},
        {'fp': 'dataset/GTMarker/001/markerRaw/36-71.json',
         'start': 36,
         'end': 71}
    ]

    df=pd.concat([d for d in toDf(fps)])
    df.to_json('dataset/GTMarker/001/all.json',orient='records')
