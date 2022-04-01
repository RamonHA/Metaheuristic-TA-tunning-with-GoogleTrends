from calendar import month
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

from trading.processes import Simulation
from trading.predictions import MyGridSearch
from trading import Asset

from functions import *


def ta(inst):

    for i in [ 4,7,10 ]:
        inst.df["rsi_{}".format(i)] = inst.rsi(i)    
        inst.df["cci_{}".format(i)] = inst.cci(i)

    for i in [ 3, 6 ]:
        inst.df[ "sma_{}".format(i) ] = inst.sma(i)
        inst.df[ "dema_{}".format(i) ] = inst.dema(i)
        inst.df[ "wma_{}".format(i) ] = inst.wma(i)
        inst.df[ "ema_{}".format(i) ] = inst.ema(i)

    for c in ["sma", "dema", "wma", "ema"]:
        for i in [3, 6]:
            for s in [1,2,3,4]:
                inst.df[ "{}_{}_{}".format(c, i, s) ] = inst.df[ "{}_{}".format(c, i) ].pct_change(s)

    inst.df["adx_7"], *_ = inst.adx(7)
    inst.df["adx_14"], *_ = inst.adx(14)
    inst.df["adx_21"], *_ = inst.adx(21)

    inst.df["bbh_20"], inst.df["bb_20"], inst.df["bbl_20"] = inst.bollinger_bands(20)
    inst.df["bbh_12"], inst.df["bb_12"], inst.df["bbl_12"] = inst.bollinger_bands(12)

    inst.df["dpo_10"] = inst.dpo(10)
    inst.df["dpo_5"] = inst.dpo(5)

    inst.df["force_index_12"] = inst.force_index(12)
    inst.df["force_index_6"] = inst.force_index(6)

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

def Grid(inst):
    m = MyGridSearch(
        inst.df,
        regr = RandomForestRegressor(),
        parameters={
            "n_estimators":[10, 20, 50, 100, 200],
            "criterion":["squared_error", "absolute_error"]
        },
        train_test_split=0.8,
        target =  "target",
        error = mean_squared_error,
        error_ascending=True
    )

    m.test()

    return m.predict() if m.best is not None else None

def func(inst):
    print("Google Trends for: ", inst)

    inst = historic(inst)

    inst = google_trends(inst)

    inst = mevs_f(inst)

    inst = ta(inst)

    inst.df = inst.df.replace( [np.inf, -np.inf] , np.nan).dropna()

    if len(inst.df) == 0: return None

    inst.df["target"] = inst.df["close"].pct_change(1).shift(-1)

    return Grid(inst)

def test():
    inst = Asset(
        symbol="aal",
        broker = "gbm",
        fiat = "mx",
        end = date(2017,12,1),
        start = date(2017,12,1) - relativedelta(months = 256),
        frequency = "1m",
        from_ = "db"
    )


if __name__ == "__main__":

    st = time.time()
    portfolio_value = 100000

    s = Simulation(
        broker = "gbm", # change if different project name in configuration
        fiat = "mx",
        commission=0, # If wanted to test with brokers commisions
        assets=None, # If you didnt add assets in configuration step, you can add the dictionary in this step
                    # just change the broker name to default
        end = date(2021,4,1), # If more recent data, simulation end time can be extended
        simulations=40, # Amount of simulations to run (based on the analysis frequency period)
        realistic=1,
        verbose = 2,
        parallel = True
    )


    s.analyze(
        frequency="1m",
        test_time=1,
        analysis={
            "RF-MyGrid":{
                "type":"prediction",
                "time":252,
                "function":func
            }
        },
        run = False
    )

    for bt in [12, 24, 48]:
        # for r, o in [ ("efficientfrontier", "minvol"), ("efficientsemivariance", "minsemivariance"), ("efficientcvar", "mincvar"), ("efficientcdar", "mincdar") ]:
        for r in ["efficientfrontier", "efficientsemivariance", "efficientcvar" , "efficientcdar"]:
            for t in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2 ]:

                print( bt, r, t )

                s.optimize(
                    balance_time = bt,
                    value = portfolio_value,
                    exp_return = True,
                    risk = r,
                    objective = "efficientreturn",
                    target_return = t,
                    run = True
                )


    results = s.results_compilation()

    print(results)
    
    df = s.behaviour( results.loc[ 0, "route" ] )

    df[ "acc" ].plot()
    plt.show()

    print("Execution time: ", time.time() - st)
