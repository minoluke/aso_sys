from .BaseModel import BaseModel

import os
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute

from .helper import sanitize_feature_names

makedir = lambda name: os.makedirs(name, exist_ok=True)

class FeatureExtractionModel(BaseModel):
    """
    Methods:
        _construct_windows
            Create overlapping data look_backward windows for feature extraction.
        _extract_features
            Extract features from windowed data.
        _get_label
            Compute label vector.
    """
    def _construct_windows(self, Nw, ti, i0=0, i1=None):
        """ Create overlapping data windows for feature extraction.

            Parameters:
            -----------
            Nw : int
                Number of windows to create.
            ti : datetime.datetime
                End of first window.
            i0 : int
                Skip i0 initial windows.
            i1 : int
                Skip i1 final windows.

            Returns:
            --------
            df : pandas.DataFrame
                Dataframe of windowed data, with 'id' column denoting individual windows.
            window_dates : list
                Datetime objects corresponding to the beginning of each data window.
        """
        if i1 is None:
            i1 = Nw

        # get data for windowing period
        df = self.data.get_data(ti-self.dtw, ti+(Nw-1)*self.dto)[self.data_streams]

        # create windows
        dfs = []
        for i in range(i0, i1):
            dfi = df[:].iloc[i*(self.iw-self.io):i*(self.iw-self.io)+self.iw]
            if len(dfi) != self.iw:
                print("not equal")
            try:
                dfi['id'] = pd.Series(np.ones(self.iw, dtype=int)*i, index=dfi.index)
            except ValueError:
                print('hi')
            dfs.append(dfi)
        df = pd.concat(dfs)
        window_dates = [ti + i*self.dto for i in range(Nw)]
        return df, window_dates[i0:i1]
    
    def _extract_features(self, ti, tf):
        """ Extract features from windowed data.

            Parameters:
            -----------
            ti : datetime.datetime
                End of first window.
            tf : datetime.datetime
                End of last window.

            Returns:
            --------
            fm : pandas.Dataframe
                tsfresh feature matrix extracted from data windows.
            ys : pandas.Dataframe
                Label vector corresponding to data windows

            Notes:
            ------
            Saves feature matrix to $rootdir/features/$root_features.csv to avoid recalculation.
        """
        makedir(self.featdir)

        # number of windows in feature request
        Nw = int(np.floor(((tf-ti)/self.dt)/(self.iw-self.io)))

        # features to compute
        cfp = ComprehensiveFCParameters()

        # check if feature matrix already exists and what it contains
        if os.path.isfile(self.featfile):
            t = pd.to_datetime(pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], usecols=['time']).index.values)
            ti0,tf0 = t[0],t[-1]
            Nw0 = len(t)
            hds = pd.read_csv(self.featfile, index_col=0, nrows=1)
            hds = list(set([hd.split('__')[1] for hd in hds]))

            # option 1, expand rows
            pad_left = int((ti0-ti)/self.dto)# if ti < ti0 else 0
            pad_right = int(((ti+(Nw-1)*self.dto)-tf0)/self.dto)# if tf > tf0 else 0
            i0 = abs(pad_left) if pad_left<0 else 0
            i1 = Nw0 + max([pad_left,0]) + pad_right
            
            # option 2, expand columns
            existing_cols = set(hds)        # these features already calculated, in file
            new_cols = set(cfp.keys()) - existing_cols     # these features to be added
            more_cols = bool(new_cols)
            all_cols = existing_cols|new_cols
            cfp = ComprehensiveFCParameters()
            cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in all_cols])

            # option 3, expand both
            if any([more_cols, pad_left > 0, pad_right > 0]) and self.update_feature_matrix:
                fm = pd.read_csv(self.featfile, index_col=0, parse_dates=['time'])
                if more_cols:
                    # expand columns now
                    df0, wd = self._construct_windows(Nw0, ti0)
                    cfp0 = ComprehensiveFCParameters()
                    cfp0 = dict([(k, cfp0[k]) for k in cfp0.keys() if k in new_cols])
                    fm2 = extract_features(df0, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp0, impute_function=impute)
                    fm2.index = pd.Series(wd)
                    
                    fm = pd.concat([fm,fm2], axis=1, sort=False)

                # check if updates required because training period expanded
                    # expanded earlier
                if pad_left > 0:
                    df, wd = self._construct_windows(Nw, ti, i1=pad_left)
                    fm2 = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp, impute_function=impute)
                    fm2.index = pd.Series(wd)
                    fm = pd.concat([fm2,fm], sort=False)
                    # expanded later
                if pad_right > 0:
                    df, wd = self._construct_windows(Nw, ti, i0=Nw - pad_right)
                    fm2 = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp, impute_function=impute)
                    fm2.index = pd.Series(wd)
                    fm = pd.concat([fm,fm2], sort=False)
                
                # write updated file output
                fm.to_csv(self.featfile, index=True, index_label='time')
                # trim output
                fm = fm.iloc[i0:i1]    
            else:
                # read relevant part of matrix
                fm = pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], header=0, skiprows=range(1,i0+1), nrows=i1-i0)
        else:
            # create feature matrix from scratch   
            df, wd = self._construct_windows(Nw, ti)
            fm = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp, impute_function=impute)
            fm.index = pd.Series(wd)
            fm.to_csv(self.featfile, index=True, index_label='time')

        fm.columns = sanitize_feature_names(fm.columns)  
        ys = pd.DataFrame(self._get_label(fm.index.values), columns=['label'], index=fm.index)
        return fm, ys
    
    def _get_label(self, ts):
        """ Compute label vector.

            Parameters:
            -----------
            t : datetime like
                List of dates to inspect look-forward for eruption.

            Returns:
            --------
            ys : list
                Label vector.
        """
        return [self.data._is_eruption_in(days=self.look_forward, from_time=t) for t in pd.to_datetime(ts)]
  