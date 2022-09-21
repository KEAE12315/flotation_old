def pie2rawIndex(pieces, df):
    for p in pieces:
        print(p, p[1]-p[0]+1)
        p = df.loc[p[0]:p[1]]
        p = p.rawIndex.tolist()
        print(p[0], p[-1], len(p))
