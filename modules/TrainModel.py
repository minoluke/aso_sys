from .FeatureExtractionModel import FeatureExtractionModel

import os
import gc
import shutil
from glob import glob
import pandas as pd
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from .helper import datetimeify, get_classifier

import joblib
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from tsfresh.transformers import FeatureSelector
from sklearn.model_selection import GridSearchCV, ShuffleSplit

makedir = lambda name: os.makedirs(name, exist_ok=True)


class TrainModel(FeatureExtractionModel):
    """
    Methods:
        _load_data
            Load feature matrix and label vector.
        _exclude_dates
            Drop rows from feature matrix and label vector.
        _collect_features
            Aggregate features used to train classifiers by frequency.
        train
            Construct classifier models.
    """

    def _load_data(self, ti, tf):
        """ Load feature matrix and label vector.

            Parameters:
            -----------
            ti : str, datetime
                Beginning of period to load features.
            tf : str, datetime
                End of period to load features.

            Returns:
            --------
            fM : pd.DataFrame
                Feature matrix.
            ys : pd.DataFrame
                Label vector.
        """
        # return pre loaded
        try:
            if ti == self.ti_prev and tf == self.tf_prev:
                return self.fM, self.ys
        except AttributeError:
            pass

        # read from CSV file
        try:
            t = pd.to_datetime(pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], usecols=['time']).index.values)
            if (t[0] <= ti) and (t[-1] >= tf):
                self.ti_prev = ti
                self.tf_prev = tf
                fM,ys = self._extract_features(ti,tf)
                self.fM = fM
                self.ys = ys
                return fM,ys
        except FileNotFoundError:
            pass

        # range checking
        if tf > self.data.tf:
            raise ValueError("Model end date '{:s}' beyond data range '{:s}'".format(tf, self.data.tf))
        if ti < self.data.ti:
            raise ValueError("Model start date '{:s}' predates data range '{:s}'".format(ti, self.data.ti))
        
        # divide training period into years
        ts = [datetime(*[yr, 1, 1, 0, 0, 0]) for yr in list(range(ti.year+1, tf.year+1))]
        if ti - self.dtw < self.data.ti:
            ti = self.data.ti + self.dtw
        ts.insert(0,ti)
        ts.append(tf)
    
        for t0,t1 in zip(ts[:-1], ts[1:]):
            print('period from {:s} to {:s}'.format(t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d')))
            fM,ys = self._extract_features(ti,t1)

        self.ti_prev = ti
        self.tf_prev = tf
        self.fM = fM
        self.ys = ys
        return fM, ys
    
    def _exclude_dates(self, X, y, exclude_dates):
        """ Drop rows from feature matrix and label vector.

            Parameters:
            -----------
            X : pd.DataFrame
                Matrix to drop columns.
            y : pd.DataFrame
                Label vector.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.

            Returns:
            --------
            Xr : pd.DataFrame
                Reduced matrix.
            yr : pd.DataFrame
                Reduced label vector.
        """
        self.exclude_dates = exclude_dates
        if len(self.exclude_dates) != 0:
            for exclude_date_range in self.exclude_dates:
                t0,t1 = [datetimeify(dt) for dt in exclude_date_range]
                inds = (y.index<t0)|(y.index>=t1)
                X = X.loc[inds]
                y = y.loc[inds]
        return X,y
    
    def _collect_features(self, save=None):
        """ Aggregate features used to train classifiers by frequency.

            Parameters:
            -----------
            save : None or str
                If given, name of file to save feature frequencies. Defaults to all.fts
                if model directory.

            Returns:
            --------
            labels : list
                Feature names.
            freqs : list
                Features appearance in classifier models.
        """
        makedir(self.modeldir)
        if save is None:
            save = '{:s}/all.fts'.format(self.modeldir)
        
        feats = []
        fls = glob('{:s}/*.fts'.format(self.modeldir))
        for i,fl in enumerate(fls):
            if fl.split(os.sep)[-1].split('.')[0] in ['all','ranked']: continue
            with open(fl) as fp:
                lns = fp.readlines()
            feats += [' '.join(ln.rstrip().split()[1:]) for ln in lns]               

        labels = list(set(feats))
        freqs = [feats.count(label) for label in labels]
        labels = [label for _,label in sorted(zip(freqs,labels))][::-1]
        freqs = sorted(freqs)[::-1]
        # write out feature frequencies
        with open(save, 'w') as fp:
            _ = [fp.write('{:d},{:s}\n'.format(freq,ft)) for freq,ft in zip(freqs,labels)]
        return labels, freqs

    def train(self, cv=0, ti=None, tf=None, Nfts=20, Ncl=100, retrain=False, classifier="DT", random_seed=0,
             n_jobs=6, exclude_dates=[]):
        """ Construct classifier models.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of training period (default is beginning model analysis period).
            tf : str, datetime.datetime
                End of training period (default is end of model analysis period).
            Nfts : int
                Number of most-significant features to use in classifier.
            Ncl : int
                Number of classifier models to train.
            retrain : boolean
                Use saved models (False) or train new ones.
            classifier : str, list
                String or list of strings denoting which classifiers to train (see options below.)
            random_seed : int
                Set the seed for the undersampler, for repeatability.
            n_jobs : int
                CPUs to use when training classifiers in parallel.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.

            Classifier options:
            -------------------
            DT - Decision Tree
            XGBoost - XGBoost
            LightGBM - LightGBM
            CatBoost - CatBoost
        """
        self.classifier = classifier
        self.n_jobs = n_jobs
        makedir(self.modeldir)

        # initialise training interval
        self.ti_train = self.ti_model if ti is None else datetimeify(ti)
        self.tf_train = self.tf_model if tf is None else datetimeify(tf)
        if self.ti_train - self.dtw < self.data.ti:
            self.ti_train = self.data.ti+self.dtw
        
        # check if any model training required
        if not retrain:
            run_models = False
            pref = type(get_classifier(self.classifier)[0]).__name__ 
            for i in range(Ncl):         
                if not os.path.isfile('{:s}/{:s}_{:04d}.pkl'.format(self.modeldir, pref, i)):
                    run_models = True
            if not run_models:
                return # not training required
        else:
            # delete old model files
            _ = [os.remove(fl) for fl in  glob('{:s}/*'.format(self.modeldir))]

        # get feature matrix and label vector
        fM, ys = self._load_data(self.ti_train, self.tf_train)

        # manually drop windows (rows)
        fM, ys = self._exclude_dates(fM, ys, exclude_dates)
        if ys.shape[0] != fM.shape[0]:
            raise ValueError("dimensions of feature matrix and label vector do not match")
        
        # select training subset
        inds = (ys.index>=self.ti_train)&(ys.index<self.tf_train)
        fM = fM.loc[inds]
        ys = ys['label'].loc[inds]

        # set up model training
        if self.n_jobs > 1:
            p = Pool(self.n_jobs)
            mapper = p.imap
        else:
            mapper = map
        f = partial(train_one_model, fM, ys, Nfts, self.modeldir, self.classifier, retrain, random_seed)

        # train models with glorious progress bar
        for i, _ in enumerate(mapper(f, range(Ncl))):
            cf = (i+1)/Ncl
            print(f'building models: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
        if self.n_jobs > 1:
            p.close()
            p.join()

        # free memory
        del fM
        gc.collect()
        self._collect_features()

        all_fts_path = os.path.join(self.modeldir, 'all.fts')
    
        ob_folder = os.path.join(self.consensusdir, self.od)
        wl_lfl_folder = os.path.join(ob_folder, f"{self.look_backward}_{self.look_forward}")
        makedir(wl_lfl_folder)
        
        new_all_fts_path = os.path.join(wl_lfl_folder, f"{cv}_all.fts")
        
        if os.path.exists(all_fts_path):
            shutil.copy(all_fts_path, new_all_fts_path)
 
def train_one_model(fM, ys, Nfts, modeldir, classifier, retrain, random_seed, random_state):
    # undersample data
    rus = RandomUnderSampler(sampling_strategy=0.75, random_state=random_state+random_seed)
    fMt,yst = rus.fit_resample(fM,ys)
        
    # find significant features
    select = FeatureSelector(n_jobs=0, ml_task='classification')
    select.fit_transform(fMt,yst)
    fts = select.features[:Nfts]
    pvs = select.p_values[:Nfts]

    fMt = fMt[fts]
    with open('{:s}/{:04d}.fts'.format(modeldir, random_state),'w') as fp:
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))

    # get sklearn training objects
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=random_state+random_seed)
    model, grid = get_classifier(classifier)            
        
    # check if model has already been trained
    pref = type(model).__name__
    fl = '{:s}/{:s}_{:04d}.pkl'.format(modeldir, pref, random_state)
    if os.path.isfile(fl) and not retrain:
        return
    
    model_cv = GridSearchCV(model, grid, cv=ss, scoring="balanced_accuracy", error_score=np.nan)
    model_cv.fit(fMt, yst)
    _ = joblib.dump(model_cv.best_estimator_, fl, compress=3)