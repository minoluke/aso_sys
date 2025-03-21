from modules import *

data_streams_dict = {
        'tremor': ['rsam', 'mf', 'hf', 'dsar'],
        'gas': ['gas_max', 'gas_min', 'gas_mean', 'gas_number'],
        'magnetic': ['magnetic'],
        'kakou': ['kakouwall_temp'],
        'tilt': ['tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW'],
        'yudamari': ['yudamari_number', 'yudamari_temp'],
        'tremor10min': ['tremor10min'],
        'all': ['rsam', 'mf', 'hf', 'dsar', 'gas_max', 'gas_min', 'gas_mean', 'gas_number','magnetic', 'kakouwall_temp', 'tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW','yudamari_number', 'yudamari_temp']
    }

data_name_dict = {
        'gas': 'volcanic gas',
        'kakou': 'crater wall',
        'magnetic': 'magnetic force',
        'tilt': 'tilt',
        'yudamari': 'thermal pool',
        'tremor': 'tremor 1day',
        }

start_period = '2010-01-01'
end_period = '2022-12-31'

min_window = 30             
delta_window = 15            
grid_number = 11


'''       
 'gas': f'{basepath}/gas/60.0_30.0/{n}_consensus.csv',
        'kakou': f'{basepath}/kakou/120.0_150.0/{n}_consensus.csv',
        'magnetic': f'{basepath}/magnetic/150.0_90.0/{n}_consensus.csv',
        'tilt': f'{basepath}/tilt/90.0_150.0/{n}_consensus.csv',
        'yudamari': f'{basepath}/yudamari/90.0_60.0/{n}_consensus.csv',
        'tremor': f'{basepath}/tremor/90.0_30.0/{n}_consensus.csv',
        '''

overlap = 0.85
look_backward = 165
look_forward = 165
cv=1
od='kakou'

data_streams = data_streams_dict.get


plotmodel = PlotModel(ti=start_period, tf=end_period, look_backward=look_backward, overlap=overlap, look_forward=look_forward, data_streams=data_streams, od=od, cv=cv)

plotmodel.plot_AUC_colormap(min_window, delta_window, grid_number, observation_data_name='crater wall temperature') 
#plotmodel.plot_learning_curve(max_models=100, metrics='AUC',observation_data_name='tilt')
'''
for key, value in data_name_dict.items():
        od=key
        plotmodel = PlotModel(ti=start_period, tf=end_period, look_backward=look_backward, overlap=overlap, look_forward=look_forward, data_streams=data_streams, od=od, cv=cv)
        #plotmodel.plot_AUC_colormap(min_window, delta_window, grid_number, observation_data_name=value, smoothed=True)
        plotmodel.plot_AUC_colormap(min_window, delta_window, grid_number, observation_data_name=value)
        '''
#feature_name = 'yudamari_number__fft_coefficient__attr_"real"__coeff_13'

#plotmodel.plot_feature_histogram(feature_name, x_min = -1, x_max = 1, xlabel= "Gas_max Linear Trend (slope)", months = 1, log_scale=False)



observation_names = ['magnetic', 'tilt', 'gas', 'kakou', 'yudamari', 'tremor']
#plotmodel.plot_optimized_time_scale(observation_names, min_window, delta_window, grid_number, smoothing_sigma=0.5, persentile=90)
#plotmodel.plot_time_series_with_alarm_all(threshold = 0.65, m_threshold=3)


window_params = {
        'vlocanic gas': ['gas',30,60],
        'creator wall': ['kakou',180,165],
        'magnetic force': ['magnetic',45,165],
        'tilt': ['tilt',180,165],
        'thermal pool': ['yudamari',165,165],
        'tremor': ['tremor',45,45],
        
}



#plotmodel.plot_pearson_correlation_matrix(window_params)

