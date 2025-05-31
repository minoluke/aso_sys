#!/usr/bin/env python3
import os, sys, itertools, time, json
from datetime import timedelta

sys.path.insert(0, os.path.abspath('..'))
from modules import *

with open(os.path.join("data", "data_streams.json"), "r", encoding="utf-8") as f:
    data_streams_dict = json.load(f)

month = timedelta(days=365.25/12)
n_jobs = 6
observation_m = ObservationData()

start_period = '2010-01-01'
end_period = '2022-12-31'

overlap = 0.99
classifier = 'DT'
#all_classifiers = ['DT','XGBoost','LightGBM','CatBoost']

def one_train_test(od,lb,lf,cv):
    start_time = time.time()

    GPU_AVAILABLE = is_gpu_available()
    if GPU_AVAILABLE:
        print("GPU available")
    else:
        print("GPU not available")

    te = observation_m.tes[int(cv)]

    data_streams = data_streams_dict.get(od)
    if data_streams is None:
        raise ValueError("Invalid value for 'od'")
    
    train_m = TrainModel(ti=start_period, tf=end_period, look_backward=float(lb), overlap=overlap, look_forward=float(lf), data_streams=data_streams, od=od, cv=cv)
    train_m.train(cv=cv, ti=start_period, tf=end_period, retrain=True, exclude_dates=[[te-6*month,te+6*month],], n_jobs=n_jobs, classifier=classifier) 

    test_m = TestModel(ti=start_period, tf=end_period, look_backward=float(lb), overlap=overlap, look_forward=float(lf), data_streams=data_streams, od=od, cv=cv)
    test_m.test(cv=cv, ti=start_period, tf=end_period, recalculate=True, n_jobs=n_jobs, classifier=classifier)  

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")

def overlap_train_test(od, min_window, delta_window, grid_number):
    """
    Function to perform grid search on one_train_test.

    Parameters:
    - od: Observation data parameter
    - min_window: Minimum window size
    - delta_window: Increment of window size
    - grid_number: Number of grid divisions
    """

    # generate frid 
    look_backward_values = [min_window + delta_window * i for i in range(grid_number)]
    look_forward_values = [min_window + delta_window * i for i in range(grid_number)]
    
    cv_values = range(observation_m.eruption_number)
    
    for lb_val, lf_val, cv_val in itertools.product(look_backward_values, look_forward_values, cv_values):
        print(f"Running train_test with look_backward={lb_val}, look_forward={lf_val}, cv={cv_val}")
        one_train_test(od, str(lb_val), str(lf_val), str(cv_val))

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('od', type=str, help='The observation data parameter')
    parser.add_argument('lb', type=str, help='The look backward parameter')
    parser.add_argument('lf', type=str, help='The look forward parameter')
    parser.add_argument('cv', type=str, help='The count volcanic eruption parameter')

    args = parser.parse_args()
    
    one_train_test(args.od,args.lb, args.lf,args.cv)
    """
    # Set grid search parameters
    observation_data = 'yudamari' 
    min_window = 30           
    delta_window = 15    
    grid_number = 11            

    overlap_train_test(observation_data, min_window, delta_window, grid_number)

    #observation_data = 'kakou'

    #overlap_train_test(observation_data, min_window, delta_window, grid_number)
    
