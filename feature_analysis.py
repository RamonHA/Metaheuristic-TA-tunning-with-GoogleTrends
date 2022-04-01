import warnings
warnings.filterwarnings("ignore")
from trading import Asset
from datetime import date, datetime

from trading.func_aux import get_assets, min_max

import json
import pandas as pd
from copy import deepcopy, copy
import json

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score

from trading.predictions import MyGridSearch

from functions import *

regr = {
    "rf":{
        "regr":RandomForestRegressor(),
        "paremeters":{
            "n_estimators":[10, 20, 50, 100, 200],
            "criterion":["squared_error", "absolute_error"]
        },
        "error":mean_squared_error
    },

    # "regr_lstm":{
        
    # },
}

clf = {
    "clf_rf":{
        "regr":RandomForestClassifier(),
        "paremeters":{
            "n_estimators":[10, 20, 50, 100, 200],
            "criterion":["gini", "entropy"]
        },
        "error":precision_score
    },

    "clf_svm":{
        "regr":SVC(),
        "parameters":{
            "C":[0.01, 0.1, 1.0, 10],
            "kernel":["linear", "poly", "rbf", "sigmoid"],
        },
        "error":precision_score
    }
}

def grid_search(df, params, **kwargs):
    m = MyGridSearch(
        df,
        regr = params["regr"],
        parameters=params["paremeters"],
        train_test_split= kwargs.get("train_test_split", 0.8),
        target = kwargs.get("target", "target"),
        error = params["error"],
        error_ascending=kwargs.get("error_ascending", True)
    )

    m.test()

    r = m.cache

    try:
        return r.to_dict()
    except:
        return {}

def model_regr(df, target, **kwargs):
    best = {}
    for i, v in regr.items():

        best[i] = grid_search(
            df,
            v,
            target = target,
            **kwargs
        )
    
    return best


#Feature analysis

# Features

def ta(inst):

    for i in [ 4,7,10,14,21 ]:
        inst.df["rsi_{}".format(i)] = inst.rsi(i)    
        inst.df["cci_{}".format(i)] = inst.cci(i)

    for i in [ 3, 6, 12 ]:
        inst.df[ "sma_{}".format(i) ] = inst.sma(i)
        inst.df[ "dema_{}".format(i) ] = inst.dema(i)
        inst.df[ "wma_{}".format(i) ] = inst.wma(i)
        inst.df[ "ema_{}".format(i) ] = inst.ema(i)

    for c in ["sma", "dema", "wma", "ema"]:
        for i in [3, 6, 12]:
            for s in [1,2,3,4]:
                inst.df[ "{}_{}_{}".format(c, i, s) ] = inst.df[ "{}_{}".format(c, i) ].pct_change(s)

    inst.df["adx_7"], *_ = inst.adx(7)
    inst.df["adx_14"], *_ = inst.adx(14)
    inst.df["adx_21"], *_ = inst.adx(21)

    inst.df["bbh_20"], inst.df["bb_20"], inst.df["bbl_20"] = inst.bollinger_bands(20)
    inst.df["bbh_12"], inst.df["bb_12"], inst.df["bbl_12"] = inst.bollinger_bands(12)

    inst.df["dpo_10"] = inst.dpo(10)
    inst.df["dpo_50"] = inst.dpo(50)

    inst.df["force_index_10"] = inst.force_index(10)
    inst.df["force_index_30"] = inst.force_index(30)

    inst.df["momentum_4"] = inst.momentum(4)
    inst.df["momentum_7"] = inst.momentum(7)
    inst.df["momentum_14"] = inst.momentum(14)

    inst.df["obv"] = inst.obv()

    inst.df["roc_7"] = inst.roc(7)
    inst.df["roc_14"] = inst.roc(14)

    inst.df["stoch_6"], _ = inst.stoch(6, 3)
    inst.df["stoch_10"], _ = inst.stoch(10, 3)
    inst.df["stoch_12"], _ = inst.stoch(12, 3)

    inst.df["tsi_12"] = inst.tsi(12, 6)
    inst.df["tsi_25"] = inst.tsi(25, 13)

    inst.df["vwap_14"] = inst.vwap(14)

    inst.df["vpt"] = inst.vpt()
    inst.df["vpt_3"] = inst.df["vpt"].pct_change(3)

    inst.df["william_15"] = inst.william(15)

    return inst

# 

def target(inst):
    cols = []
    for i in range(1, 6):
        c = "target_{}".format( i )
        inst.df[ c ] = inst.df["close"].pct_change( periods = i )
        cols.append( c )

    return inst, cols

def causality(inst, targets):
    cols = inst.causality( targets= targets )
    return cols

def corr(inst, targets):
    cols = inst.corr(targets = targets)
    cols = abs(cols)
    for c in cols: cols[ c ] = cols[ c ].apply( lambda x : 1 if x > 0.7 else 0 )
    return cols

def mFCBF(inst, targets):
    df = pd.DataFrame()
    for t in targets:
        instt = deepcopy(inst)
        wt = list( set(targets) - set([t]))
        cols = list( set(inst.df.columns) - set(wt) )

        corrs = abs(instt.df[ cols ].corr())

        cols = list( set(corrs[ corrs[t] > 0.2 ].index.tolist()) - set([t]))

        if len(cols) == 0:
            corrs[ t ] = 0
            df = pd.concat( [df, corrs[t].drop(index = t)], axis = 1 )
            continue

        cols = instt.redundancy( 
            df = abs(instt.df[ cols + [t] ].corr()),
            targets = [t],
            threshold = 0.8, 
            above = False 
        )

        corrs.loc[ cols, t ] = 1
        corrs.loc[  list( set(corrs.index) - set(cols) ) , t ] = 0

        df = pd.concat( [df, corrs[t].drop(index = t)], axis = 1 )
    
    return df

def main(inst, fs = True):

    inst = historic(inst)

    inst = google_trends(inst)

    inst = mevs_f(inst)

    inst = ta(inst)

    inst, targets = target(inst)

    inst.df.dropna(inplace = True)

    r = {}
    if fs:
        for i in ["causality", "corr", "mFCBF"]:
            print(i)
            inst_ = deepcopy(inst)

            try:
                r[i] = {
                    "causality":causality,
                    "corr":corr,
                    "mFCBF":mFCBF
                }[i](inst_, targets).to_json()
            except Exception as e:
                r[i] = {}
                print("Exception: ", e)
    else:
        r = inst.df.columns.to_list()


    return r

def test():
    inst = Asset(
        symbol = "AAL",
        broker = "gbm",
        fiat = "mx",
        start = date(2000,1,1),
        end = date(2022,3,1),
        frequency = "1m",
        from_ = "db"
    )

    inst.df.dropna(inplace = True)

    r = main(inst)

def regr_pred(inst):

    targets = [ t for t in inst.df.columns if "target" in t]

    X = inst.df.drop(columns = targets)

    y = inst.df[targets]

    r = {}
    
    for t in targets:
        r[t] = model_regr( 
            pd.concat([ X, y[t] ], axis = 1),
            target=t
        )
    
    return r


if __name__ == "__main__":



    assets = get_assets()[ "gbm" ]

    results = {}

    for i in list(assets.keys())[ :30 ] :
        print(i)
        inst = Asset(
            symbol=i,
            broker="gbm",
            fiat = "mx",
            start = date(2000,1,1),
            end = date(2022,2,1),
            frequency="1m",
            from_ = "db"
        )

        inst.df.dropna(inplace = True)

        if inst.df is None or len(inst.df) == 0:
            continue

        results[i] = {}

        results[i]["features"] = main(inst, fs = True)

        # targets = [ t for t in inst.df.columns if "target" in t]

        # results[i]["regr"] = regr_pred(inst)


    with open( "results.json", "w" ) as fp:
        json.dump( results, fp )
