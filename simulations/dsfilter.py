import collections
from trading.processes import Simulation
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.algorithms.so_de import DE
from pymoo.optimize import minimize

from trading.processes import Simulation
from trading.metaheuristics.ta_tunning import TATunningProblem
from trading.strategy import Strategy

import sys
sys.path.append("../")
from functions import *

from copy import copy

def dsfilter(asset):
    nasset = copy(asset)

    st = Strategy( asset=asset, tas = "trend_oneparam" )

    st.TREND_FIRST_PARAM = ( 3, 30, 3 )

    r = st.value( target=[ 4 ], verbose = 0 )

    r = r[ r["result"] < 0.05 ]

    if r.empty:
        return None
    
    r.sort_values( by = "result", ascending=True , inplace = True)

    l = 10 if len(r) > 10 else int(len(r) / 2)

    r = r.iloc[ :10 ]

    v = [ ]
    last_v = st.asset.df.iloc[-1].to_dict()

    for i, row in r.iterrows():
        v.append( row["range_down"][0] < last_v[ row["col"] ] < row["range_down"][1] )
    
    counter = collections.Counter(v)

    return (counter[False] > counter[True]) and (counter[False] > len( v ) // 2)

GEN = 50

class GTTATunning(TATunningProblem):

    def _update_ta(self, vector):

        self.asset.df["ema"] = self.asset.ema( vector[0] ).pct_change(1)

        self.asset.df["wma"] = self.asset.wma( vector[1] ).pct_change(1)

        self.asset.df["dema"] = self.asset.ema( vector[2] ).pct_change(1)

        self.asset.df["cci"] = self.asset.cci( vector[3] )

        self.asset.df["sma"] = self.asset.sma( vector[4] ).pct_change(1)

        self.asset.df["stoch"], _ = self.asset.stoch( vector[5], 3 )

        self.asset.df["williams"] = self.asset.williams( vector[6] )

        self.asset.df["force_index"] = self.asset.force_index( vector[7] )

def metaheuristic(inst):
    
    inst.df["vpt"] = inst.vpt()

    inst.df.dropna(inplace = True)

    if len(inst.df) == 0: return None

    algorithm = DE(
        pop_size = 100,
        variant="DE/best/2/bin",
        CR = 0.8,
        F = 0.1,
        dither = "scalar",
    )

    problem = GTTATunning(
        asset = inst,
        regr = RandomForestRegressor(),
        xl = 3,
        xu = 24,
        verbose = 0,
        n_var = 8
    )

    # try:
    res = minimize(
        problem,
        algorithm,
        ("n_gen", GEN),
        seed = 1,
        verbose = False
    )
    # except Exception as e:
    #     print("{} with exception {}".format( inst, e ))
    #     return None

    problem.update_ta( res.X.astype(int) )

    predict = problem.predict(for_real = True)

    aux = predict[-1] if predict is not None else None

    return aux

def func(inst):
    print("Google Trends for: ", inst)

    inst = historic(inst)

    inst = google_trends(inst)

    inst = mevs_f(inst)
    
    pred = metaheuristic(inst)

    print(pred)

    return pred

if __name__ == "__main__":

    st = time.time()

    portfolio_value = 100000

    s = Simulation(
        broker = "gbm", # change if different project name in configuration
        fiat = "mx",
        commission=0, # If wanted to test with brokers commisions
        assets=None, # If you didnt add assets in configuration step, you can add the dictionary in this step
                    # just change the broker name to default
        end = date(2020,12,1), # If more recent data, simulation end time can be extended
        simulations=36, # Amount of simulations to run (based on the analysis frequency period)
        realistic=1,
        verbose = 2,
        parallel = True
    )


    s.analyze(
        frequency="1m",
        test_time=1,
        analysis={
            "DSFilter":{
                "type":"filter",
                "frequency":"1w",
                "time":300,
                "function":dsfilter
            },
            "DE":{
                "type":"prediction",
                "time":252,
                "function":func
            }
        },
        run = True
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


    # results = s.results_compilation()

    # print(results)
    
    # df = s.behaviour( results.loc[ 0, "route" ] )

    # df[ "acc" ].plot()
    # plt.show()

    # print("Execution time: ", time.time() - st)