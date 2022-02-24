
from trading.predictions import MyGridSearch
from trading.func_aux import get_assets
from trading import Asset

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import *

from datetime import date

# Feature relecvance

def causality(inst, targets, target):
    cols = list(inst.df.columns).remove( targets.remove(target) )
    lag = int( target.split("_")[-1] )
    cols = inst.causality( df = inst.df[cols], targets = [target],  lag = lag)

    cols = cols[ cols[target] == 1 ].index.tolist()

    inst.df = inst.df[ cols + [target] ]

    return inst

# Regressions and Classifications

regr = {
    "regr_rf":{
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

    return m.cache

def model_clf(inst, target = ["target"], fr = "causality", **kwargs):

    best = {}

    for i, v in clf.items():
        best[i] = {}
        for j in target:

            df = {
                "causality":causality
            }[ fr ]( inst, target, j )

            best[i][j] = { 
                grid_search(
                    df,
                    v,
                    target = j,
                    error_ascending = False,
                    **kwargs
                )
            }

    return best

def main(func, target = "close", target_periods = 5, fr = ["causality"]):

    targets = [ "{}_{}".format( target, i ) for i in range(1, target_periods+1) ]

    results = {}

    assets = get_assets()[ "gbm" ]

    for i in list(assets.keys())[ :10 ] :

        results[ i ] = {}

        inst = Asset(
            symbol=i,
            broker="gbm",
            fiat = "mx",
            start = date(2000,1,1),
            end = date(2022,2,1),
            frequency="1m"
        )

        for f in fr:

            results[i]["regr"] = {}
            results[i]["regr"][f] = model_regr(
                func(inst, type = "regr"),
                target = targets,
                fr = f
            )

            results[i]["clf"] = {}
            results[i]["clf"][f] = model_clf(
                func(inst, type = "clf"),
                target=targets,
                fr = f
            )
    
    return results

def main_(func, targets = 5 , type = "regr", fr = "causality", **kwargs):

    targets = [ "{}_{}".format( "target", i ) for i in range(1, targets+1) ]

    assets = get_assets()[ "gbm" ]
    results = {}

    for i in list(assets.keys())[ :10 ] :
        results[i] = {}

        inst = Asset(
            symbol=i,
            broker="gbm",
            fiat = "mx",
            start = date(2000,1,1),
            end = date(2022,2,1),
            frequency="1m",
        )

        inst = func( inst, type = type )

        for target in targets:
            inst = {
                "causality":causality
            }[fr](inst, targets, target)

            results[i][fr] = {}

            if type == "regr":
                results[i][fr]["regr"] = model_regr( inst, target=target, **kwargs )
            else:
                pass
    
    return results
                
