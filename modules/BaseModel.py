import os
from datetime import timedelta
from inspect import getfile, currentframe
from .ObservationData import ObservationData
from .helper import datetimeify
 
class BaseModel(object):
    """ Object for train and running forecast models.
        
        Constructor arguments:
        ----------------------
        look_backward : float
            Length of data look_backward window in days.
        overlap : float
            Fraction of overlap between adjacent windows. Set this to 1. for overlap of entire window minus 1 data point.
        look_forward : float
            Length of look-forward in days.
        ti : str, datetime.datetime
            Beginning of analysis period. If not given, will default to beginning of observation data.
        tf : str, datetime.datetime
            End of analysis period. If not given, will default to end of observation data.
        data_streams : list
            Data streams and transforms from which to extract features. 
        root : str 
            Naming convention for forecast files. If not given, will default to 'fm_*Tw*wndw_*eta*ovlp_*Tlf*lkfd_*ds*' where
            Tw is the look_backward window length, eta is overlap fraction, Tlf is look-forward and ds are data streams.

        Attributes:
        -----------
        data : ObservationData
            Object containing observation data.
        dtw : datetime.timedelta
            Length of look_backward window.
        dtf : datetime.timedelta
            Length of look-forward.
        dt : datetime.timedelta
            Length between data samples.
        dto : datetime.timedelta
            Length of non-overlapping section of look_backward window.
        iw : int
            Number of samples in look_backward window.
        io : int
            Number of samples in overlapping section of look_backward window.
        ti_model : datetime.datetime
            Beginning of model analysis period.
        tf_model : datetime.datetime
            End of model analysis period.
        ti_train : datetime.datetime
            Beginning of model training period.
        tf_train : datetime.datetime
            End of model training period.
        ti_forecast : datetime.datetime
            Beginning of model forecast period.
        tf_forecast : datetime.datetime
            End of model forecast period.
        exclude_dates : list
            List of time look_backward windows to exclude during training. Facilitates dropping of eruption 
            look_backward windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
            ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.
        n_jobs : int
            Number of CPUs to use for parallel tasks.
        rootdir : str
            Repository location on file system.
        plotdir : str
            Directory to save forecast plots.
        modeldir : str
            Directory to save forecast models (pickled sklearn objects).
        featdir : str
            Directory to save feature matrices.
        featfile : str
            File to save feature matrix to.
        preddir : str
            Directory to save forecast model predictions.
    """
    def __init__(self, look_backward, look_forward,data_streams, overlap=0.99, ti=None, tf=None, root=None, od=None, cv=None):
        self.look_backward = look_backward
        self.overlap = overlap
        self.look_forward = look_forward
        self.data_streams = data_streams
        self.data = ObservationData()
        #if any([d not in self.data.df.columns for d in self.data_streams]):
        #    raise ValueError("data restricted to any of {}".format(self.data.df.columns))
        self.eruption_number = self.data.eruption_number
        if ti is None: ti = self.data.ti
        if tf is None: tf = self.data.tf
        self.ti_model = datetimeify(ti)
        self.tf_model = datetimeify(tf)
        if self.tf_model > self.data.tf:
            raise ValueError("Model end date '{:s}' beyond data range '{:s}'".format(self.tf_model, self.data.tf))
        if self.ti_model < self.data.ti:
            raise ValueError("Model start date '{:s}' predates data range '{:s}'".format(self.ti_model, self.data.ti))
        self.dtw = timedelta(days=self.look_backward)
        self.dtf = timedelta(days=self.look_forward)
        self.dt = timedelta(days=1.)
        self.dto = (1.-self.overlap)*self.dtw
        self.iw = int(self.look_backward)   
        self.od = od    
        self.cv = str(cv)    
        self.io = int(self.overlap*self.iw)      
        if self.io == self.iw: self.io -= 1

        self.look_backward = self.iw*1.
        self.dtw = timedelta(days=self.look_backward)
        if self.ti_model - self.dtw < self.data.ti:
            self.ti_model = self.data.ti+self.dtw
        self.overlap = self.io*1./self.iw
        self.dto = (1.-self.overlap)*self.dtw
        
        self.exclude_dates = []
        self.update_feature_matrix = True
        self.n_jobs = 6
        self.classifier = "DT"

        # naming convention and file system attributes
        if root is None:
            self.root = '{}_{}wndw_{}lkfd_{:3.2f}ovlp'.format(self.od, int(self.look_backward), int(self.look_forward), self.overlap)
        else:
            self.root = root
        self.rootdir = os.sep.join(getfile(currentframe()).split(os.sep)[:-2])
        self.plotdir = r'{:s}/save/figures/{:s}'.format(self.rootdir, self.root)
        self.modeldir = r'{:s}/save/rawdata/models/{:s}cv_{:s}'.format(self.rootdir, self.cv, self.root)
        self.featdir = r'{:s}/save/rawdata/features'.format(self.rootdir, self.root)
        self.featfile = r'{:s}/{:s}_features.csv'.format(self.featdir, self.root)
        self.preddir = r'{:s}/save/rawdata/pred_each/{:s}cv_{:s}'.format(self.rootdir, self.cv, self.root)
        self.consensusdir = r'{:s}/save/rawdata/consensus/'.format(self.rootdir)
        self.aucmapdir = r'{:s}/save/rawdata/aucmap/{:s}'.format(self.rootdir,self.od)
    