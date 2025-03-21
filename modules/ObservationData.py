import os
from datetime import datetime
from inspect import getfile, currentframe
import pandas as pd

from .helper import datetimeify


class ObservationData(object):
    """ Object to manage acquisition and processing of seismic data.
        
        Attributes:
        -----------
        df : pandas.DataFrame
            Time series of observation data and transforms.
        t0 : datetime.datetime
            Beginning of data range.
        t1 : datetime.datetime
            End of data range.

        Methods:
        --------
        get_data
            Return observation data in requested date range.
    """
    def __init__(self):
        self.file = os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data','observation_data.dat'])
        self._assess()
    def __repr__(self):
        if self.exists:
            tm = [self.ti.year, self.ti.month, self.ti.day]
            tm += [self.tf.year, self.tf.month, self.tf.day]
            return 'ObservationData:{:d}/{:02d}/{:02d} to {:d}/{:02d}/{:02d}'.format(*tm)
        else:
            return 'no data'

    def _assess(self):
        """ Load existing file and check date range of data.
        """
        # get eruptions
        with open(os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data','eruptive_periods.txt']),'r') as fp:
            self.tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        self.eruption_number = len(self.tes)
        # check if data file exists
        self.exists = os.path.isfile(self.file)
        # check date of latest data in file
        self.df = pd.read_csv(self.file, index_col=0, parse_dates=[0,])
        self.ti = self.df.index[0]
        self.tf = self.df.index[-1]

    def _is_eruption_in(self, days, from_time):
        """ Binary classification of eruption imminence.

            Parameters:
            -----------
            days : float
                Length of look-forward.
            from_time : datetime.datetime
                Beginning of look-forward period.

            Returns:
            --------
            label : int
                1 if eruption occurs in look-forward, 0 otherwise
            
        """
        for te in self.tes:
            if 0 < (te-from_time).total_seconds()/(3600*24) < days:
                return 1.
        return 0.

    def get_data(self, ti=None, tf=None):
        """ Return observation data in requested date range.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Date of first data point (default is earliest data).
            tf : str, datetime.datetime
                Date of final data point (default is latest data).

            Returns:
            --------
            df : pandas.DataFrame
                Data object truncated to requsted date range.
        """
        # set date range defaults
        if ti is None:
            ti = self.ti
        if tf is None:
            tf = self.tf

        # convert datetime format
        ti = datetimeify(ti)
        tf = datetimeify(tf)

        # subset data
        inds = (self.df.index>=ti)&(self.df.index<tf)
        return self.df.loc[inds]
    