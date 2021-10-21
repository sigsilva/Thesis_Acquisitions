import biosignalsnotebooks as bsnb
import os
import numpy as np
import json
import glob
import itertools
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row
from aux_functions_1 import create_android_sync_header, re_sample_data, load, synchronise_signals, \
    generate_sync_txt_file,synchronize_Plux_Android


config_sync = {'JR': [147,152,158,163,146,151,157,162], 'MIH' : [112,117,123,126,111,116,121,124], 'FG' : [161,164,171,175,160,163,170,174], 'SM' : [122,128,133,136,122,128,132,135],
               'CC' : [108,113,118,124,106,111,116,122], 'MAH' : [123, 126,132,134,121,124,131,133], 'IM' : [131,136,142,146,129,134,140,144], 'PP': [115,118,124,126,113,116,122,124],
               'MD': [115,119,126,129,113,117,122,125], 'BB' : [102,105,116,120,100,103,115,119], 'NM' : [139,144,153,157,139,144,151,155], 'CQ': [152,155,157,161,150,153,156,160]}


#Synchronising data from multiple Android sensor files into one file
# set file path
path = '/Users/sarasilva/Desktop/Thesis_Acquisitions/Original_DATA/'

# get a list with all the folders within that folder
subdir_list = os.listdir(path)
#print(subdir_list)

# make full path for each folder
subdir_list= [path + subdir for subdir in subdir_list if subdir != '.DS_Store']
print('list with path folders')
'''
# get the current path
save_path = os.path.abspath('/Users/sarasilva/Desktop/Thesis_Acquisitions/Synchronised_DATA/android_sync_wear_sync/')

for path_dir in subdir_list:
    # list for holding the resampled data
    re_sampled_data_Android = []
    re_sampled_data_Wear = []

    # list for holding the time axes of each sensor
    re_sampled_time_Android = []
    re_sampled_time_Wear = []
    # save the subdirectory name in a variable
    name_subdir = path_dir.split(os.sep)[6]
    #    print(split_path)
    # get a list with all the files within that folder
    file_list = os.listdir(path_dir)
    # make full path for each file from smartphone and for each file from smartwatch
    file_list_Android = [path_dir + '/' + file for file in file_list
                 if (file != '.DS_Store') if ('ANDROID' in file)]
    file_list_Android.sort()

    file_list_Wear = [path_dir + '/' + file for file in file_list
                         if (file != '.DS_Store') if ('WEAR' in file)]
    file_list_Wear.sort()

    sensor_data_Android, report_Android = bsnb.load_android_data(file_list_Android)
    sensor_data_Wear, report_Wear = bsnb.load_android_data(file_list_Wear)
    print('data loaded')

    #out_file = open(name_subdir + "_report.json", "w")

    #json.dump(report_Android, out_file, indent=11)
    #json.dump(report_Wear, out_file, indent=11)

    #out_file.close()

    # plotting the smartphone and smartwatch sensor acquisition timeline
    bsnb.plot_android_sensor_timeline(sensor_data_Android, report_Android, line_thickness=1.5)
    bsnb.plot_android_sensor_timeline(sensor_data_Wear, report_Wear, line_thickness=1.5)

    # padding all the signals to the same length
    # the sensor that started latest is chosen for indicate when the synchronization starts
    # the sensor that stopped earliest is chosen for indicate when the synchronization stops
    padded_sensor_data_Android = bsnb.pad_android_data(sensor_data_Android, report_Android, padding_type='same')
    print('data padded')

    ## Resampling all signals to the same sampling rate
    # cycle over the signal
    for data in padded_sensor_data_Android:
        re_time, re_data, sampling_rate_Android= re_sample_data(data[:, 0], data[:, 1:], shift_time_axis=True,
                                                                  sampling_rate=100, kind_interp='previous')
        print('data re-sampled')
        # add the the time and data to the lists
        re_sampled_time_Android.append(re_time)
        re_sampled_data_Android.append(re_data)

    # padding all the signals to the same length
    # the sensor that started latest is chosen for indicate when the synchronization starts
    # the sensor that stopped earliest is chosen for indicate when the synchronization stops
    padded_sensor_data_Wear = bsnb.pad_android_data(sensor_data_Wear, report_Wear, end_with= 'WearAcc', padding_type='same')

    ## Resampling all signals to the same sampling rate
    # cycle over the signal
    for data in padded_sensor_data_Wear:
        re_time, re_data, sampling_rate_Wear = re_sample_data(data[:, 0], data[:, 1:], shift_time_axis=True,
                                                                  sampling_rate=100, kind_interp='previous')
        print('data re-sampled')
        # add the the time and data to the lists
        re_sampled_time_Wear.append(re_time)
        re_sampled_data_Wear.append(re_data)

    # create header
    header_Android = create_android_sync_header(file_list_Android, sampling_rate_Android)
    print('header was created')
    header_Wear = create_android_sync_header(file_list_Wear, sampling_rate_Wear)
    print('header was created')

    # save the synchronised data
    bsnb.save_synchronised_android_data(re_sampled_time_Android[0], re_sampled_data_Android, header_Android, save_path,
                                        file_name=name_subdir + '_Android' + '_synchronised')
    print('file with synchronised data was saved')
    bsnb.save_synchronised_android_data(re_sampled_time_Wear[0], re_sampled_data_Wear, header_Wear, save_path,
                                        file_name=name_subdir + '_Wear' + '_synchronised')
    print('file with synchronised data was saved')


# Before we start synchronising both files we will take a look at both
# accelerometer channels that we are going to use for synchronisation
# The y-axis of the accelerometer is the ("CH1") and
# the x-axis of the accelerometer is ("CH0") for the Android
# and the Wear sensor files, respectively.
# load the data

path_synchronised_data = '/Users/sarasilva/Desktop/Thesis_Acquisitions/Synchronised_DATA/android_sync_wear_sync/'
save_path = '/Users/sarasilva/Desktop/Thesis_Acquisitions/Synchronised_DATA/android_wear_sync/'
file_synchronised_list = glob.glob(path_synchronised_data + '*')
# sort list
# essential for grouping
file_synchronised_list.sort()




# group similar files name
res = [list(i) for j, i in itertools.groupby(file_synchronised_list, lambda file: file.split('_')[7])]

print(res)

for files in res:
    android_data = load(files[0])
    android = list(android_data.keys())
    android_data = android_data[android[0]]['CH1']
    wear_data = load(files[1])
    wear = list(wear_data.keys())
    wear_data = wear_data[wear[0]]['CH0']
    #android_time = np.loadtxt(files[0])[:, 0]
    #wear_time = np.loadtxt(files[1])[:, 0]
    # plot the data of both files
    # check the delay between the two time series ie the time difference
    # between the start of smartwatch and smartphone acquisition
    subject = files[0].split('_')[7]
    print(subject)
    # read the dictionary and look up the range of values for synchronization
    array_intervals = config_sync[subject]
    #samples_delay,_,_ = synchronise_signals(android_data,wear_data, array_intervals, time_interval = 300, fs = 100)
    # set path, including new file name
    print('Set PATH where synchronised file will be saved')
    new_path = os.path.join(save_path, subject +'_android_wear_sync.txt')
    print('Start Synchronization')
    # synchronize
    generate_sync_txt_file(files, array_intervals, channels=('CH1', 'CH0'), new_path=new_path)



'''
# SYNCHRONISING ANDROID AND WEAR WITH PLUX SENSORS
## Get metadata through the read of file header
# set file path

path_android_wear_sync = '/Users/sarasilva/Desktop/Thesis_Acquisitions/Synchronised_DATA/android_wear_sync/'
file_list_android_wear_sync = os.listdir(path_android_wear_sync)

for path_dir in subdir_list:
    # get a list with all the files within that folder
    file_list = os.listdir(path_dir)
    name = path_dir.split('_')[4]
    print(name)
    # make full path for BIOSIGNALSPLUX
    file_list_PLUX = [path_dir + '/' + file for file in file_list
                         if (file != '.DS_Store') if ('0007803B4638' in file)]

    android_wear_file = [path_android_wear_sync + file for file in file_list_android_wear_sync
                         if (file != '.DS_Store') if (name in file)]


    # get the header of the file
    with open(file_list_PLUX[0], encoding='latin-1') as opened_file:
    # read the information from the header lines (omitting the begin and end tags of the header)
        header_biosignalsPLUX = opened_file.readlines()[1][2:]  # omit "# " at the beginning of the sensor information

    # convert the header to a dictionary (for easier access of the information that we need)
        header_biosignalsPLUX = json.loads(header_biosignalsPLUX)

    # get the header of the file
    with open(android_wear_file[0], encoding='latin-1') as opened_file:
        # read the information from the header lines (omitting the begin and end tags of the header)
        header_Android_Wear = opened_file.readlines()[1][2:]  # omit "# " at the beginning of the sensor information

        # convert the header to a dictionary (for easier access of the information that we need)
        header_Android_Wear = json.loads(header_Android_Wear)

    ## Access/store relevant metadata about the sensor, device and acquisition
    # get the key first key of the dictionary (all information can be accessed using that key)
    dict_key = list(header_biosignalsPLUX.keys())[0]

    # get the values for unit conversion
    device = header_biosignalsPLUX[dict_key]['device']
    resolution = header_biosignalsPLUX[dict_key]['resolution'][0]

    # get the sampling rate of the device
    sampling_rate = header_biosignalsPLUX[dict_key]['sampling rate']

    # set the sensor name and the option
    sensor_name = 'ACC'
    option = 'm/s^2'

    # get the data from the file (as an numpy array)
    data_biosignalsPLUX = np.loadtxt(file_list_PLUX[0])
    android_wear_data = np.loadtxt(android_wear_file[0])

    # convert the sampling axis to a time axis
    data_biosignalsPLUX[:, 0] = bsnb.generate_time(data_biosignalsPLUX[:, 5], sampling_rate)


    #accelerometer and EMG channel data needs to be converted to G and milliVolts units respectively

    # convert the data of the ACC channel (in our case the ACC data is the last column in the file. This can be accessed using -1)
    # in order to make things simple we are directly overwriting that channel in the data array
    for i in range(4, 7):
        data_biosignalsPLUX[:,i] = bsnb.raw_to_phy(sensor_name,device,data_biosignalsPLUX[:,i], resolution, option)

    data_biosignalsPLUX[:, 2] = bsnb.raw_to_phy('EMG', device, data_biosignalsPLUX[:, 2], resolution, 'mV')
    data_biosignalsPLUX[:, 3] = bsnb.raw_to_phy('EMG', device, data_biosignalsPLUX[:, 3], resolution, 'mV')


    # set file name for the file in which the converted data is going to be saved
    file_name = name + '_PLUX_synchronised_converted.txt'

    # crete full file path
    save_path_Plux = os.path.join('/Users/sarasilva/Desktop/Thesis_Acquisitions/Synchronised_DATA/PLUX_converted/', file_name)

    # set path, including new file name
    save_path_Android_Wear = os.path.join(
        '/Users/sarasilva/Desktop/Thesis_Acquisitions/Synchronised_DATA/android_wear_biosignalsPLUX_sync/',
        name + '_new_android_bitalino_sync.txt')

    synchronize_Plux_Android(data_biosignalsPLUX, android_wear_data, 100, header_biosignalsPLUX, header_Android_Wear, save_path_Plux,
                             save_path_Android_Wear)



# Before we start synchronising both files we will take a look at both
# accelerometer channels that we are going to use for synchronisation
# The y-axis of the accelerometer is the ("CH6") and
# the y-axis of the accelerometer is second channel ("CH1") for the BITalino
# and the Android sensor files, respectively.



    # load the data
    #bitalino_data = bsnb.load(bitalino_file, get_header=False)
    #android_wear_data = bsnb.load(android_wear_file, get_header=False)

    # plot data
    #bitalino_time = np.loadtxt(bitalino_file)[:, 0]
    #android_wear_time = np.loadtxt(android_wear_file)[:, 0]

    # plot the data of both files
    #bsnb.plot([bitalino_time, android_time], [bitalino_data['CH6'], android_wear_data['CH1']], legend_label=["bitalino CH6", "android CH1"], y_axis_label=["Accelerometer", "Accelerometer"], x_axis_label="Time (s)")





