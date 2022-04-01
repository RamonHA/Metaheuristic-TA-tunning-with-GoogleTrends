import pandas as pd

from trading.mev.mev import mevs

def historic(inst):

    cols = inst.df.columns

    for c in cols:
        for i in range(1, 6):
            inst.df[ "{}_{}".format(c, i) ] = inst.df[c].pct_change( periods = i )

    return inst

def google_trends(inst):

    df = inst.google_trends(from_ = "db")

    col = df.columns
    for c in col:
        for i in range( 1, 5 ):
            df[ "{}_{}".format( c, i ) ] = df[ c ].shift(i)
            df[ "{}_{}_change".format( c, i ) ] = df[ c ].pct_change(i)
    
    inst.df = pd.concat([ inst.df, df ], axis = 1)

    return inst

def mevs_f(inst):

    df = mevs("all", "1m")

    col = df.columns
    for c in col:
        for i in range( 1, 5 ):
            df[ "{}_{}".format( c, i ) ] = df[ c ].shift(i)

    inst.df = pd.concat([ inst.df, df ], axis = 1)

    return inst

