from .FeatureExtractionModel import FeatureExtractionModel
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter, label
#from skimage.measure import regionprops
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import  MinMaxScaler
import json

makedir = lambda name: os.makedirs(name, exist_ok=True)

class PlotModel(FeatureExtractionModel):
    """
    Description:
        Class for plotting model results and features.
        This class inherits from FeatureExtractionModel.
        (FeatureExtractionModel also inherits from BaseModel)

    Attributes:
        eruptive_periods_path : str
            Path to eruptive periods file
        data_path : str
            Path to observation data file


    Methods:
        _load_eruptive_periods
            Load eruption periods from file
        plot_feature_histogram
            Plot histogram of feature
        _eruption_within_days
            Check if eruption is within days
        _split_train_test
            Split data into train and test sets
        _preprocess_data
            Preprocess data for plotting
        plot_learning_curve
            Plot learning curve
        plot_AUC_colormap
            Plot AUC colormap
        _get_alarm_periods
            Get alarm periods
        _smooth_data    
            Smooth data
        _scale_data
            Scale data
        calculate_metrics
            Calculate metrics
        _filter_post_eruption
            Filter post eruption
        plot_time_series_with_alarm
            Plot time series with alarm
        _calculate_pearson_correlation
            Calculate Pearson correlation
        _find_consensus_paths
            Find consensus paths
        
    """

    def __init__(self, look_backward, overlap, look_forward, data_streams, ti=None, tf=None, root=None, od=None, cv=None):
        super().__init__(look_backward, overlap, look_forward, data_streams, ti, tf, root, od, cv)
        self.eruptive_periods_path = r'{:s}/data/eruptive_periods.txt'.format(self.rootdir)
        self.data_path = r'{:s}/data/observation_data.dat'.format(self.rootdir)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.timeseries_dir = r'{:s}/save/figures/timeseries/{:s}'.format(self.rootdir,timestamp)
        self.feature_histogram_dir = r'{:s}/save/figures/feature_histogram/{:s}'.format(self.rootdir, self.od)
        self.learning_curve_dir = r'{:s}/save/figures/learning_curve'.format(self.rootdir)
        self.auc_map_dir = r'{:s}/save/figures/AUC_matrix'.format(self.rootdir)
        self.auc_data_dir = r'{:s}/save/rawdata/AUC_colormap'.format(self.rootdir)
        self.pearson_correlation_matrix_dir = r'{:s}/save/figures/pearson_correlation_matrix'.format(self.rootdir)
        self.optimized_time_scale_figure_dir = r'{:s}/save/figures/optimized_time_scale'.format(self.rootdir)
        self.optimized_time_scale_data_dir = r'{:s}/save/rawdata/optimized_time_scale'.format(self.rootdir)
           
    def _load_eruptive_periods(self, file_path=None):
        if file_path is None:
            file_path = self.eruptive_periods_path
        eruptive_periods = []
        with open(file_path, 'r') as file:
            for line in file:
                eruptive_periods.append(datetime.strptime(line.strip(), '%Y %m %d %H %M %S'))
        return eruptive_periods

    def plot_feature_histogram(self, feature_name, x_min, x_max, xlabel, months, log_scale=False):
        eruptive_periods = self._load_eruptive_periods()
        file_path = self.featfile
        data = pd.read_csv(file_path)
        time_axis = pd.to_datetime(data['time'])

    
        feature_data = np.array(data[feature_name])

        bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), 30) if log_scale else np.linspace(x_min, x_max, 30)

        fig, ax = plt.subplots(figsize=(18, 6))  

        n, bins, patches = ax.hist(feature_data, bins=bin_edges, alpha=0.5, label='All Period')
        vertical_offset = max(n) * 0.9
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(xlabel, fontsize=25)

        ymax = ax.get_ylim()[1]


        pre_eruption_all_features = []

        for eruption_date in eruptive_periods:
            months = months
            pre_eruption_start = eruption_date - pd.DateOffset(months=months)


            pre_eruption_mask = (time_axis >= pre_eruption_start) & (time_axis <= eruption_date)
            pre_eruption_features = np.array(feature_data)[pre_eruption_mask]
            pre_eruption_times = time_axis[pre_eruption_mask]

            indices_to_plot = np.linspace(0, len(pre_eruption_features) - 1, min(5, len(pre_eruption_features)), dtype=int)
            pre_eruption_features = pre_eruption_features[indices_to_plot]
            pre_eruption_times = pre_eruption_times.iloc[indices_to_plot]

            days_until_eruption = (eruption_date - pre_eruption_times).dt.days
            sizes = ymax * 0.9 - 0.02 * ymax * days_until_eruption / months  

            ax.scatter(pre_eruption_features, [vertical_offset] * len(pre_eruption_features),
                    label=f'{months} month before {eruption_date.strftime("%Y-%m-%d")}',
                    marker='x', s=sizes, linewidths=5)

            
            pre_eruption_all_features.extend(pre_eruption_features)

            vertical_offset -= ymax * 0.19


        stat, p_value = mannwhitneyu(feature_data, pre_eruption_all_features, alternative='two-sided')

        ax.text(0.05, 0.3, f'p = {p_value:.1e}', fontsize=24, color='black', style='italic', transform=ax.transAxes)

        ax.set_yticks([]) 
        ax.tick_params(axis='x', labelsize=14)


        if log_scale:
            ax.set_xscale('log')  

  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

 
        ax.legend(loc='upper left', fontsize=14, frameon=False, bbox_to_anchor=(0.02, 0.98), borderaxespad=0)

        plot_dir = self.feature_histogram_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(f'{plot_dir}/{feature_name}.png')
        #plt.show()

    def _eruption_within_days(self,date, eruptive_periods, days):
        for eruption_date in eruptive_periods:
            if date <= eruption_date <= date + timedelta(days=days):
                return True
        return False

    def _split_train_test(self, df, eruptive_periods, start_test, end_test, exclusion_index):
        df['is_test_period'] = df['time'].apply(lambda x: start_test <= x <= end_test and x < eruptive_periods[exclusion_index])
        test_df = df[df['is_test_period']]
        train_df = df[~df['is_test_period']]
        return test_df, train_df

    def _preprocess_data(self, file_path, eruptive_periods, lookforward_days, start_test, end_test, exclusion_index, smoothed=False, smoothed_window_size=1):
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        if smoothed:
            df = self._smooth_data(df, 1)

            df['consensus'] = df['smoothed']
            df['eruption_within_days'] = df['time'].apply(lambda x: self._eruption_within_days(x, eruptive_periods, smoothed_window_size))
        else:
            #df = self._smooth_data(df, 1)

            #df['consensus'] = df['smoothed']
            df['eruption_within_days'] = df['time'].apply(lambda x: self._eruption_within_days(x, eruptive_periods, lookforward_days))

        test_df, train_df = self._split_train_test(df, eruptive_periods, start_test, end_test, exclusion_index)
        return test_df, train_df

    def plot_learning_curve(self,max_models=10, metrics='AUC', observation_data_name=None):
        pred_path = self.preddir
        observation_data = self.od
        eruptive_periods = self._load_eruptive_periods()
        metric_test_scores = []
        metric_train_scores = []
        model_range = range(1, max_models + 1)
        lookforward_days = self.look_forward

        exclusion_index = int(self.cv)
        start_test = eruptive_periods[exclusion_index] - timedelta(days=180) - timedelta(days=lookforward_days)
        end_test = eruptive_periods[exclusion_index] 

        for num_models in model_range:
            all_y_true_test = []
            all_y_scores_test = []
            all_y_true_train = []
            all_y_scores_train = []

            for i in range(num_models):
                file_path = f'{pred_path}/DecisionTreeClassifier_{i:04d}.csv'
                test_df, train_df = self._preprocess_data(file_path, eruptive_periods, lookforward_days, start_test, end_test, exclusion_index)
                if test_df is None:
                    continue

                # Test data
                all_y_true_test.extend(test_df['eruption_within_days'].astype(int))
                all_y_scores_test.extend(test_df[f'pred{i:04d}'])

                # Train data
                all_y_true_train.extend(train_df['eruption_within_days'].astype(int))
                all_y_scores_train.extend(train_df[f'pred{i:04d}'])

            if metrics == 'AP':
                metric_test = average_precision_score(all_y_true_test, all_y_scores_test) if all_y_true_test else 0
                metric_train = average_precision_score(all_y_true_train, all_y_scores_train) if all_y_true_train else 0
            elif metrics == 'AUC':
                metric_test = roc_auc_score(all_y_true_test, all_y_scores_test) if all_y_true_test else 0
                metric_train = roc_auc_score(all_y_true_train, all_y_scores_train) if all_y_true_train else 0
            else:
                raise ValueError("metrics must be 'AP' or 'AUC'")

            metric_test_scores.append(metric_test)
            metric_train_scores.append(metric_train)

        if observation_data_name is not None:
            observation_data = observation_data_name
        # Plotting AP for test and train data against the number of models
        
        # Plotting AP for test and train data against the number of models
        plt.figure(figsize=(10, 6))
        plt.plot(model_range, metric_test_scores, label=f'Test Data {metrics}', marker='o')
        plt.plot(model_range, metric_train_scores, label=f'Train Data {metrics}', marker='x')


        plt.xlabel('Number of Models', fontsize=18)
        plt.ylabel(metrics, fontsize=18)

  
        plt.title(f'AUC trends for {observation_data}', fontsize=20)

        plt.tick_params(axis='x', labelsize=14)  
        plt.tick_params(axis='y', labelsize=14)  

        plt.legend(fontsize=14)

        plt.grid()

        plot_dir = self.learning_curve_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(f'{plot_dir}/{self.root}.png')

    def _calculate_auc(self, file_path, eruptive_periods, lookforward_days, exclusion_index, smoothed):
        exclusion_index = int(self.cv)
        start_test = eruptive_periods[exclusion_index] - timedelta(days=180) - timedelta(days=lookforward_days)
        end_test = eruptive_periods[exclusion_index]
        test_df, train_df = self._preprocess_data(file_path, eruptive_periods, lookforward_days, start_test, end_test, exclusion_index, smoothed)


        y_true = test_df['eruption_within_days'].astype(int)
        y_scores = test_df['consensus']
        if len(np.unique(y_true)) < 2:
            print("y_trueには1つのクラスしか存在しないため、AUCの計算は行われません。")
            return 0.5
        else:
            return roc_auc_score(y_true, y_scores)
            
    def plot_AUC_colormap(self, min_window, delta_window, grid_number, observation_data_name=None, smoothed=False):
        eruption_number = self.eruption_number
        eruptive_periods = self._load_eruptive_periods()
        observation_data = self.od

            
     
        look_backward_values = [min_window + delta_window * i for i in range(grid_number)]
        look_forward_values = [min_window + delta_window * i for i in range(grid_number)]
        
        if self.od == 'tremor10min':
            look_backward_values = [0.5, 1, 2, 3, 4]
            look_forward_values = [0.5, 1, 2, 3, 4]
         
        
        full_auc_matrix = np.zeros((grid_number, grid_number))
        exclude_auc_matrix = np.zeros((grid_number, grid_number,eruption_number))
        for i, lb_val in enumerate(look_backward_values):
            for j, lf_val in enumerate(look_forward_values):
                full_auc_values = 0
                exclude_auc_values = np.zeros(eruption_number)
                for cv_val in range(eruption_number):
                    self.look_backward = lb_val
                    self.look_forward = lf_val
                    self.cv = cv_val
                    ob_folder = os.path.join(self.consensusdir, self.od)
                    wl_lfl_folder = os.path.join(ob_folder, f"{lb_val:.1f}_{lf_val:.1f}")
                    consensus_file = os.path.join(wl_lfl_folder, f"{cv_val}_consensus.csv")
                    auc_value = self._calculate_auc(consensus_file, eruptive_periods, lf_val, cv_val, smoothed=smoothed)
                    full_auc_values += auc_value
                    exclude_auc_values = [exclude_auc_values[k] + auc_value if k != cv_val else exclude_auc_values[k] for k in range(eruption_number)]
                    
                full_auc_matrix[i, j] = full_auc_values / eruption_number
                exclude_auc_matrix[i, j,:] = np.array(exclude_auc_values) / (eruption_number - 1)


      
        if smoothed == False:
            auc_data_dir = r'{:s}/{:s}'.format(self.auc_data_dir, observation_data)
        else:
            auc_data_dir = r'{:s}_smoothed/{:s}'.format(self.auc_data_dir, observation_data)

        if not os.path.exists(auc_data_dir):
            os.makedirs(auc_data_dir)
        np.savetxt(f'{auc_data_dir}/full_auc_matrix.csv', full_auc_matrix, delimiter=',')
        for cv_val in range(eruption_number):
            np.savetxt(f'{auc_data_dir}/exclude_auc_matrix_{cv_val}.csv', exclude_auc_matrix[:,:,cv_val], delimiter=',')

        if observation_data_name is not None:
            observation_data = observation_data_name

        plt.rcParams.update({
            'font.size': 18,         
            'axes.titlesize': 22,   
            'axes.labelsize': 20,    
            'xtick.labelsize': 17,  
            'ytick.labelsize': 17,   
            'legend.fontsize': 16   
        })
        
        plt.figure(figsize=(10, 8))
        plt.imshow(full_auc_matrix, cmap='viridis', interpolation='none', aspect='auto', vmin=0.5, vmax=1.0)
        plt.colorbar(label='Mean AUC')
        plt.xticks(ticks=np.arange(len(look_forward_values)), labels=look_forward_values)
        plt.yticks(ticks=np.arange(len(look_backward_values)), labels=look_backward_values)
        plt.xlabel('Look Forward Length (days)')
        plt.ylabel('Look Backward Length (days)')
        plt.title(f'{observation_data}')
        plt.gca().invert_yaxis()

        if smoothed == False:
            auc_dir = r'{:s}/{:s}'.format(self.auc_map_dir, observation_data)
        else:
            auc_dir = r'{:s}_smoothed/{:s}'.format(self.auc_map_dir, observation_data)

        if not os.path.exists(auc_dir):
            os.makedirs(auc_dir)
        plt.savefig(f'{auc_dir}/Mean_AUC_{observation_data}.png')

        for cv_val in range(eruption_number):
            plt.figure(figsize=(10, 8))
            plt.imshow(exclude_auc_matrix[:,:,cv_val], cmap='viridis', interpolation='none', aspect='auto', vmin=0.5, vmax=1.0)
            plt.colorbar(label='Mean AUC')
            plt.xticks(ticks=np.arange(len(look_forward_values)), labels=look_forward_values)
            plt.yticks(ticks=np.arange(len(look_backward_values)), labels=look_backward_values)
            plt.xlabel('Look Forward Length (days)')
            plt.ylabel('Look Backward Length (days)')
            plt.title(f'{observation_data} (Excluding CV={cv_val})')
            plt.gca().invert_yaxis()
            if not os.path.exists(auc_dir):
                os.makedirs(auc_dir)
            plt.savefig(f'{auc_dir}/Mean_AUC_{observation_data}_exclude_{cv_val}.png')

        #plt.show()

    def _get_alarm_periods(self, model_data, eruptive_periods, start_time, end_time, threshold, m_threshold):
        alarm_periods = []
        is_alarm = False
        last_alert_time = None
        alarm_end_time = None
        tp, fp, tn, fn = 0, 0, 0, 0

        duration = 1

        eruption_dates = [pd.to_datetime(e) for e in eruptive_periods if start_time <= pd.to_datetime(e) <= end_time]

        times = model_data[next(iter(model_data))]['time']

        """
        for t in times:
            count_above_threshold = 0
            for model in model_data.values():
                filtered = model[model['time'] == t]
                if not filtered.empty and filtered['smoothed'].values[0] > threshold:
                    count_above_threshold += 1

            model_threshold =  m_threshold

            if count_above_threshold >= model_threshold:
                if not is_alarm:
                    last_alert_time = t
                    alarm_end_time = t + pd.Timedelta(days=duration)
                    is_alarm = True
                else:
                    alarm_end_time = max(alarm_end_time, t + pd.Timedelta(days=duration))"""
        for t in times:
            smoothed_values = []
            for model in model_data.values():
                filtered = model[model['time'] == t]
                if not filtered.empty:
                    smoothed_values.append(filtered['smoothed'].values[0])
            
            if smoothed_values:
                mean_smoothed = sum(smoothed_values) / len(smoothed_values)
            else:
                mean_smoothed = 0
        
        if mean_smoothed >= threshold:
            if not is_alarm:
                last_alert_time = t
                alarm_end_time = t + pd.Timedelta(days=duration)
                is_alarm = True
            else:
                alarm_end_time = max(alarm_end_time, t + pd.Timedelta(days=duration))

            
            if is_alarm and t > alarm_end_time:
                
                eruption_in_alarm = any(last_alert_time <= ed <= alarm_end_time for ed in eruption_dates)
                if eruption_in_alarm:
                    tp += 1 
                else:
                    fp += 1  
                alarm_periods.append((last_alert_time, alarm_end_time))
                is_alarm = False

        
        if is_alarm:
            eruption_in_alarm = any(last_alert_time <= ed <= alarm_end_time for ed in eruption_dates)
            if eruption_in_alarm:
                tp += 1
            else:
                fp += 1
            alarm_periods.append((last_alert_time, alarm_end_time))

        
        last_end = start_time
        for start, end in alarm_periods:
           
            eruption_in_non_alarm = any(last_end <= ed <= start for ed in eruption_dates)
            if eruption_in_non_alarm:
                fn += 1  
            else:
                tn += 1  
            last_end = end

        
        if last_end < end_time:
            eruption_in_non_alarm = any(last_end <= ed <= end_time for ed in eruption_dates)
            if eruption_in_non_alarm:
                fn += 1
            else:
                tn += 1

        return alarm_periods, tp, fp, tn, fn
        
    def _smooth_data(self, df, window_size, DayData=True):
        
        df['smoothed'] = df['consensus'].rolling(window=window_size, min_periods=1).mean()

      
        if DayData:
            df = df.set_index('time').resample('D').asfreq()  
        df['smoothed'] = df['smoothed'].interpolate(method='linear')
        
        
        df = df.reset_index()

        return df

    def _scale_data(self, df):
        scaler = MinMaxScaler()
        df[['smoothed']] = scaler.fit_transform(df[['smoothed']])
        return df

    def _calculate_metrics(self, tp, fp, tn, fn):
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0


        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        mcc = numerator / denominator if denominator > 0 else 0

        return mcc, precision, recall

    def _filter_post_eruption(self, df, eruptive_periods, cv, months=6):
        if cv <= -1 or cv > len(eruptive_periods):
            raise ValueError("n must be between 1 and the length of eruptive_periods")

        eruption_date = pd.to_datetime(eruptive_periods[cv])  
        start_date = eruption_date - pd.DateOffset(months=months)

        df = df[(df['time'] >= start_date) & (df['time'] <= (eruption_date + pd.Timedelta(days=1)))]
        return df

    def _plot_time_series_with_alarm(self, model_paths, smooth_window_sizes, cv, threshold, m_threshold):
        model_data = {}
        eruptive_periods = self._load_eruptive_periods()

       
        for (name, path), window_size in zip(model_paths.items(), smooth_window_sizes):
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])

            if name == 'tremor10min':
                df = self._smooth_data(df, window_size, DayData=False)
            else:           
                df = self._smooth_data(df, window_size)

            df =  self._filter_post_eruption(df, eruptive_periods, cv)

          
            df = self._scale_data(df)

            model_data[name] = df

        start_time = max([data['time'].min() for data in model_data.values()])
        end_time = min([data['time'].max() for data in model_data.values()]) + pd.Timedelta(days=60)

        alarm_periods, tp, fp, tn, fn = self._get_alarm_periods(model_data, eruptive_periods, start_time, end_time, threshold=threshold, m_threshold=m_threshold)

        mcc, precision, recall = self._calculate_metrics(tp, fp, tn, fn)
        plt.figure(figsize=(15, 3))
        
        lines = []
        for name, df in model_data.items():
            label_name = name.replace('gas', 'volcanic gas').replace('magnetic', 'magnetic force').replace('kakou', 'crater wall').replace('yudamari', 'hot spring').replace('tremor', 'tremor 1 day').replace('short', 'tremor 10 min')
            line, = plt.plot(df['time'], df['smoothed'], label=f'{label_name} model')
            lines.append(line)

        for alarm_start, alarm_end in alarm_periods:
            plt.axvspan(alarm_start, alarm_end, color='lightcoral', alpha=0.3)

        for eruption in eruptive_periods:
            eruption_date = pd.to_datetime(eruption)
            if start_time <= eruption_date <= end_time:
                plt.axvline(x=eruption_date, color='red', linestyle='--')

        plt.xlim(start_time, end_time)

        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Prediction Index', fontsize=14)
        plt.title('Prediction Index Time Series with Alarm Periods', fontsize=16)

        red_patch = plt.Line2D([0], [0], color='lightcoral', lw=4, label='alarm period')
        plt.legend(handles=[red_patch] + lines, loc='upper right', fontsize=12)

        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        #plt.show()

        timeseries_dir = self.timeseries_dir
        if not os.path.exists(timeseries_dir):
            os.makedirs(timeseries_dir)
        plt.savefig(f'{timeseries_dir}/timeseries_{cv}cv.png')

        return tp, fp, tn, fn, mcc, precision, recall
  
    def plot_time_series_with_alarm_all(self, threshold = 0.65, m_threshold=3):
        basepath = self.consensusdir
        time_scale_data_dir = self.optimized_time_scale_data_dir
        time_scale_dic = json.load(open(f'{time_scale_data_dir}/optimized_time_scales.json', 'r'))

        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
        for cv in range(self.eruption_number):
            model_paths = {
                obs: f'{basepath}/{obs}/{scale[0]}.0_{scale[1]}.0/{cv}_consensus.csv'
                for obs, scale in time_scale_dic[str(cv)].items()
            }
            smooth_window_sizes = [scale[1] for scale in time_scale_dic[str(cv)].values()]

            tp, fp, tn, fn, mcc, precision, recall = self._plot_time_series_with_alarm(model_paths, smooth_window_sizes, cv, threshold=threshold, m_threshold=m_threshold)

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

        total_tn = total_tn - total_tp
        total_mcc, total_precision, total_recall = self._calculate_metrics(total_tp, total_fp, total_tn, total_fn)

        print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total TN: {total_tn}, Total FN: {total_fn}")
        print(f"Total MCC: {total_mcc}, Total Precision: {total_precision}, Total Recall: {total_recall}")

    def _calculate_pearson_correlation(self, df1, df2, cv, eruption_definition_date=10):
        merged = pd.merge(df1, df2, on='time', suffixes=('_1', '_2'))

        eruption_periods = self._load_eruptive_periods()
        eruption_date = pd.to_datetime(eruption_periods[cv])
        start_date = eruption_date - pd.DateOffset(days=eruption_definition_date)
        pos = merged[(merged['time'] >= start_date) & (merged['time'] <= eruption_date)]
        neg = merged[(merged['time'] < start_date) | (merged['time'] > eruption_date)]
        pos_1_hist, _ = np.histogram(pos['consensus_1'], bins=np.arange(0, 1.01, 0.01))
        pos_2_hist, _ = np.histogram(pos['consensus_2'], bins=np.arange(0, 1.01, 0.01))
        neg_1_hist, _ = np.histogram(neg['consensus_1'], bins=np.arange(0, 1.01, 0.01))
        neg_2_hist, _ = np.histogram(neg['consensus_2'], bins=np.arange(0, 1.01, 0.01))
        
        D_1 = pos_1_hist / len(pos) - neg_1_hist / len(neg)
        D_2 = pos_2_hist / len(pos) - neg_2_hist / len(neg)

        correlation = np.corrcoef(D_1, D_2)[0, 1]

        return correlation
    
    def _find_consensus_paths(self, window_params, cv):
        consensus_paths = {}
        for name, params in window_params.items():
            window = f'{params[0]}/{params[1]:.1f}_{params[2]:.1f}'
            consensus_paths[name] = f'{self.consensusdir}/{window}/{cv}_consensus.csv'
        return consensus_paths
    
    def plot_pearson_correlation_matrix(self, window_params):
        average_correlation_matrix = np.zeros((len(window_params), len(window_params)))
        for cv in range(self.eruption_number-1):
            consensus_paths = self._find_consensus_paths(window_params, cv)
            df_dict = {}
            for name, path in consensus_paths.items():
                df = pd.read_csv(path)
                df['time'] = pd.to_datetime(df['time'])
                df_dict[name] = df

            for name, df in df_dict.items():
                df = df.set_index('time').resample('D').asfreq()
                df = df.interpolate(method='linear')
                df = df.reset_index()
                df_dict[name] = df

            for name, df in df_dict.items():
                df_dict[name] = self._filter_post_eruption(df, self._load_eruptive_periods(), cv)
            

            correlation_matrix = np.zeros((len(df_dict), len(df_dict)))
            for i, (name1, df1) in enumerate(df_dict.items()):
                for j, (name2, df2) in enumerate(df_dict.items()):
                    correlation_matrix[i, j] = self._calculate_pearson_correlation(df1, df2, cv)
            
            abs_correlation_matrix = np.abs(correlation_matrix)
            average_correlation_matrix += abs_correlation_matrix

            plt.figure(figsize=(10, 8))
            plt.imshow(abs_correlation_matrix, cmap='viridis', interpolation='none', aspect='auto', vmin=0, vmax=1)
            plt.colorbar(label='Pearson Correlation Coefficient')

            plt.xticks(ticks=np.arange(len(df_dict)), labels=df_dict.keys())
            plt.yticks(ticks=np.arange(len(df_dict)), labels=df_dict.keys())
            plt.xlabel('Observation Data')
            plt.ylabel('Observation Data')
            plt.title(f'Pearson Correlation Coefficient Matrix {cv} of eruption number')

            for i in range(correlation_matrix.shape[0]):
                for j in range(correlation_matrix.shape[1]):
                    plt.text(j, i, f"{abs_correlation_matrix[i, j]:.2f}",
                            ha='center', va='center', color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')


            plt.gca().invert_yaxis()
            plt.xticks(rotation=45)
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
            if not os.path.exists(self.pearson_correlation_matrix_dir):
                os.makedirs(self.pearson_correlation_matrix_dir)
            plt.savefig(f'{self.pearson_correlation_matrix_dir}/{timestamp}_cv{cv}.png')
            #plt.show()

        average_correlation_matrix /= (self.eruption_number-1)


        plt.figure(figsize=(10, 8))
        plt.imshow(average_correlation_matrix, cmap='viridis', interpolation='none', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Pearson Correlation Coefficient')
        plt.xticks(ticks=np.arange(len(df_dict)), labels=df_dict.keys())
        plt.yticks(ticks=np.arange(len(df_dict)), labels=df_dict.keys())
        plt.xlabel('Observation Data')
        plt.ylabel('Observation Data')
        plt.title('Average Pearson Correlation Coefficient Matrix')

        for i in range(average_correlation_matrix.shape[0]):
            for j in range(average_correlation_matrix.shape[1]):
                plt.text(j, i, f"{average_correlation_matrix[i, j]:.2f}",
                        ha='center', va='center', color='white' if abs(average_correlation_matrix[i, j]) > 0.5 else 'black')
        plt.gca().invert_yaxis()
        plt.xticks(rotation=45)
        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        plt.savefig(f'{self.pearson_correlation_matrix_dir}/{timestamp}_average.png')
        #plt.show()

    def plot_optimized_time_scale(self, observation_names, min_window, delta_window, grid_number, smoothing_sigma=1, persentile=85):
        look_backward_values = [min_window + delta_window * i for i in range(grid_number)]
        look_forward_values = [min_window + delta_window * i for i in range(grid_number)]
        optimized_time_scales = {
            cv: {observation_name: {} for observation_name in observation_names} 
            for cv in range(self.eruption_number)
        }

        for observation_name in observation_names:
            for cv in range(self.eruption_number):
                auc_data_dir = r'{:s}/{:s}'.format(self.auc_data_dir, observation_name)
                exclude_auc_matrix = np.loadtxt(f'{auc_data_dir}/exclude_auc_matrix_{cv}.csv', delimiter=',')
                exclude_auc_matrix = exclude_auc_matrix[:grid_number, :grid_number]
                optimal_range_coords, max_auc_coords = self._find_optimal_range(exclude_auc_matrix, smoothing_sigma=smoothing_sigma, percentile=persentile)
                plt.figure(figsize=(10, 8))
                plt.imshow(exclude_auc_matrix, cmap='viridis', interpolation='none', aspect='auto', vmin=0.5, vmax=1.0)
                plt.colorbar(label='Mean AUC')
                plt.xticks(ticks=np.arange(len(look_forward_values)), labels=look_forward_values)
                plt.yticks(ticks=np.arange(len(look_backward_values)), labels=look_backward_values)
                plt.xlabel('Look Forward Length (days)')
                plt.ylabel('Look Backward Length (days)')
                plt.title(f'Optimal range for {observation_name} (Cross-validation: {cv})')
                plt.gca().invert_yaxis()
                if optimal_range_coords.size > 0: 
                    optimal_rows, optimal_cols = zip(*optimal_range_coords) 
                    plt.scatter(optimal_cols, optimal_rows, color='red', s=100, label='Optimal Range')
                    plt.scatter(max_auc_coords[1], max_auc_coords[0], color='blue', s=150, marker='x', label='optimal point')

                plt.legend()
                #plt.show()
                optimized_time_scales[cv][observation_name] = [look_backward_values[max_auc_coords[0]], look_forward_values[max_auc_coords[1]]]
                plot_dir = self.optimized_time_scale_figure_dir
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(f'{plot_dir}/Optimal_Range_{observation_name}_cv{cv}.png')
                plt.close()

        print(optimized_time_scales)
        optimized_time_scale_dir = self.optimized_time_scale_data_dir
        if not os.path.exists(optimized_time_scale_dir):
            os.makedirs(optimized_time_scale_dir)
        with open(f'{optimized_time_scale_dir}/optimized_time_scales.json', 'w') as f:
            json.dump(optimized_time_scales, f, indent=4)
                
    def _find_optimal_range(self, auc_data, smoothing_sigma=1, percentile=85):
        padding = 3
        padded_auc_data = np.pad(auc_data, pad_width=padding, mode='constant', constant_values=0)
        smoothed_data = gaussian_filter(padded_auc_data, sigma=smoothing_sigma)[padding:-padding, padding:-padding]  # パディングを除去

        threshold = np.percentile(smoothed_data, percentile)
        binary_map = smoothed_data > threshold

        labeled_array, num_features = label(binary_map)

        regions = regionprops(labeled_array)
        largest_region = max(regions, key=lambda r: r.area) 
        optimal_range_coords = largest_region.coords 

        max_auc_value = smoothed_data[largest_region.coords[:, 0], largest_region.coords[:, 1]].max()
        max_auc_coords = largest_region.coords[np.argmax(auc_data[largest_region.coords[:, 0], largest_region.coords[:, 1]])]

        return optimal_range_coords, max_auc_coords

