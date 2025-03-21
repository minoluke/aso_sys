from .FeatureExtractionModel import FeatureExtractionModel
from multiprocessing import Pool
from functools import partial
from glob import glob
import gc

import os, joblib
import pandas as pd
import numpy as np
from .helper import get_classifier, datetimeify

makedir = lambda name: os.makedirs(name, exist_ok=True)

class TestModel(FeatureExtractionModel):
    """
    Methods:
        forecast
            Use classifier models to forecast eruption likelihood.
    """
    
    def test(self,cv=0, ti=None, tf=None, recalculate=False, n_jobs=6, classifier='GRU'):
        """ Use classifier models to forecast eruption likelihood.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of forecast period (default is beginning of model analysis period).
            tf : str, datetime.datetime
                End of forecast period (default is end of model analysis period).
            recalculate : bool
                Flag indicating forecast should be recalculated, otherwise forecast will be
                loaded from previous save file (if it exists).
            n_jobs : int
                Number of cores to use.

            Returns:
            --------
            consensus : pd.DataFrame
                The model consensus, indexed by window date.
        """
        self.classifier = classifier
        makedir(self.preddir)

        # 
        self.ti_forecast = self.ti_model if ti is None else datetimeify(ti)
        self.tf_forecast = self.tf_model if tf is None else datetimeify(tf)
        if self.tf_forecast > self.data.tf:
            self.tf_forecast = self.data.tf
        if self.ti_forecast - self.dtw < self.data.ti:
            self.ti_forecast = self.data.ti+self.dtw

        loadFeatureMatrix = True

        model_path = self.modeldir + os.sep            
        model,classifier = get_classifier(self.classifier)

        # logic to determine which models need to be run and which to be read from disk
        pref = type(model).__name__
        fls = glob('{:s}/{:s}_*.pkl'.format(model_path, pref))
        load_predictions = []
        run_predictions = []
        if recalculate:
            run_predictions = fls
        else:
            for fl in fls:
                num = fl.split(os.sep)[-1].split('_')[-1].split('.')[0]
                flp = '{:s}/{:s}_{:s}.csv'.format(self.preddir, pref, num)
                if not os.path.isfile(flp):
                    run_predictions.append(flp)
                else:
                    load_predictions.append(flp)

        ys = []            
        # load existing predictions
        for fl in load_predictions:
            y = pd.read_csv(fl, index_col=0, parse_dates=['time'], infer_datetime_format=True)
            ys.append(y)

        # generate new predictions
        if len(run_predictions)>0:
            run_predictions = [(rp, rp.replace(model_path, self.preddir+os.sep).replace('.pkl','.csv')) for rp in run_predictions]

            # load feature matrix
            fM,_ = self._extract_features(self.ti_forecast, self.tf_forecast)

            # setup predictor
            if self.n_jobs > 1:
                p = Pool(self.n_jobs)
                mapper = p.imap
            else:
                mapper = map
            f = partial(test_one_model, fM, model_path, pref)

            # train models with glorious progress bar
            for i, y in enumerate(mapper(f, run_predictions)):
                cf = (i+1)/len(run_predictions)
                print(f'forecasting: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
                ys.append(y)
            
            if self.n_jobs > 1:
                p.close()
                p.join()
        
        # condense data frames and write output
        ys = pd.concat(ys, axis=1, sort=False)
        consensus = np.mean([ys[col].values for col in ys.columns if 'pred' in col], axis=0)
        forecast = pd.DataFrame(consensus, columns=['consensus'], index=ys.index)

        ob_folder = os.path.join(self.consensusdir, self.od)
        wl_lfl_folder = os.path.join(ob_folder, f"{self.look_backward}_{self.look_forward}")
        makedir(wl_lfl_folder)
        consensus_file = os.path.join(wl_lfl_folder, f"{cv}_consensus.csv")
        forecast.to_csv(consensus_file, index=True, index_label='time')
        
        # memory management
        if len(run_predictions)>0:
            del fM
            gc.collect()

        return forecast

def test_one_model(fM, model_path, pref, flp):
    flp,fl = flp
    num = flp.split(os.sep)[-1].split('.')[0].split('_')[-1]
    model = joblib.load(flp)
    with open(model_path+'{:s}.fts'.format(num)) as fp:
        lns = fp.readlines()
    fts = [' '.join(ln.rstrip().split()[1:]) for ln in lns]            
    
    # simulate predicton period
    yp = model.predict(fM[fts])
    
    # save prediction
    ypdf = pd.DataFrame(yp, columns=['pred{:s}'.format(num)], index=fM.index)
    ypdf.to_csv(fl, index=True, index_label='time')
    return ypdf
