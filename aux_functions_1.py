import json
import numpy as np
import math
import scipy as scp
import scipy.interpolate as scpi
import biosignalsnotebooks as bsnb
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
from aux_functions_2 import gravitational_filter
from bokeh.plotting import figure, show, output_file

def _round_sampling_rate(sampling_rate):
    """
    Function for round the sampling rate to the nearest tens digit. Sampling rates below 5 Hz are set to 1 Hz
    Parameters
    ----------
    sampling_rate: A sampling rate
    Returns
    -------
    rounded_sampling rate: the sampling rounded to the next tens digit
    """
    # check if sampling rate is below 5 Hz in that case always round to one
    if sampling_rate < 5:

        # set sampling rate to 1
        rounded_sampling_rate = 1

    else:

        # round to the nearest 10 digit
        rounded_sampling_rate = round(sampling_rate/10) * 10

    return rounded_sampling_rate

def _calc_avg_sampling_rate(time_axis, unit='seconds', round=True):
    # function to calculate the average sampling rate of signals recorded with an android sensor. The sampling rate is
    # rounded to the next tens digit if specified(i.e 34.4 Hz = 30 Hz | 87.3 Hz = 90 Hz).
    # sampling rates below 5 Hz are set to 1 Hz.

    # Parameters
    # ----------
    # time_axis (N array_like): The time axis of the sensor
    # unit (string, optional): the unit of the time_axis. Either 'seconds' or 'nanoseconds' can be used.
    #                        If not specified 'seconds' is used
    # round (boolean, true): Boolean to indicate whether the sampling rate should be rounded to the next tens digit

    # Returns
    # -------
    # avg_sampling_rate: the average sampling rate of the sensor

    # check the input for unit and set the dividend accordingly
    if (unit == 'seconds'):

        dividend = 1

    elif (unit == 'nanoseconds'):

        dividend = 1e9

    else:  # invalid input

        raise IOError('The value for unit is not valid. Use either seconds or nanoseconds')

    # calculate the distance between sampling points
    # data[:,0] is the time axis
    sample_dist = np.diff(time_axis)

    # calculate the mean distance
    mean_dist = np.mean(sample_dist)

    # calculate the sampling rate and add it to the list
    # 1e9 is used because the time axis is in nanoseconds
    avg_sampling_rate = dividend / mean_dist

    # round the sampling rate if specified
    if (round):
        avg_sampling_rate = _round_sampling_rate(avg_sampling_rate)

    return avg_sampling_rate


def _truncate_time(number, decimals=0):
    """
    Truncates a time to the specified decimal places. This function is needed when a time axis is generated for PLUX
    devices, because otherwise there will be approximation "errors" that come from the binary internal representation
    of fractions with floats. See: https://docs.python.org/3/tutorial/floatingpoint.html
    Parameters
    ----------
    number: float
            The float number that is supposed to be truncated
    decimals: int
            The number of decimal places that are supposed to stay preserved after the truncation
    Returns
    -------
    A truncated float number
    """

    # check for validity of inputs
    if not isinstance(decimals, int):
        raise TypeError("The parameter \'decimal\' must be an integer value.")
    elif decimals < 0:
        raise ValueError("The parameter \'decimal\' has to be >= 0.")
    elif decimals == 0:
        return math.trunc(number)

    # calculate the shift factor for truncation
    shift_factor = 10 ** decimals

    # calculate the truncated number (shift --> truncate --> shift back)
    return math.trunc(number * shift_factor) / shift_factor


def _calc_time_precision(sampling_rate):
    """
    Calculates the number of needed digits after the decimal point based on the given sampling rate.
    This is done by finding the periodicity within the fraction. The precision is the number of digits of a non-repeated
    sequence AFTER the decimal point + 1.
    Parameters
    ----------
    sampling_rate: integer
        Sampling frequency of acquisition.
    Returns
    -------
    the precision needed for the given sqmpling rate
    """

    # convert sampling rate to int
    sampling_rate = int(sampling_rate)

    # calculate the first remainder
    remainder = 1 % sampling_rate

    # list for holding all remainders that have been calculated
    seen_remainders = []

    # calculate all other remainders until the modulo operation yields either zero or the remainder has already appeared
    # before (this is when the periodicity starts)
    while (remainder != 0) and (remainder not in seen_remainders):
        # append the current remainder to the seen remainder list
        seen_remainders.append(remainder)

        # multiply the remainder by ten (this is basically the process of converting a fraction to a decimal number
        # using long division
        remainder = remainder * 10

        # calculate the next remainder
        remainder = remainder % sampling_rate

    return len(seen_remainders) + 1

def create_android_sync_header(in_path, sampling_rate):

    '''
    function in order to creating a new header for a synchronised android sensor file
    (i.e. multiple android sensor files into one single file)
    Parameters
    ----------
    in_path (list of strings): list containing the paths to the files that are supposed to be synchronised
    sampling_rate(int): The sampling rate to which the signals are going to be synchronised

    Returns
    -------
    header (string): the new header as a string
    '''

    # variable for the header
    header = None

    # cycle through the file list
    for i, file in enumerate(in_path):

        # check if it is the first file entry
        if (i == 0):

            # open the file
            with open(file, encoding='latin-1') as opened_file:
                # read the information from the header lines (omitting the begin and end tags of the header)
                header_string = opened_file.readlines()[1][2:]  # omit "# " at the beginning of the sensor information

                # convert the header into a dict
                header = json.loads(header_string)

                dict_key = list(header.keys())[0]

                if "WEAR" in file:
                    # get the last channel from the channel field
                    first_num_channels = header[dict_key]['channels'][-1]
                    first_num_labels = header[dict_key]['label']

                    # adjust the channel number
                    first_new_channels = [ch + (first_num_channels + 1) for ch in range(len(first_num_labels)-1)]

                    header[dict_key]['channels'].extend(first_new_channels)
                    header[dict_key]['column'].extend(header[dict_key]['label'][1:])




        else:

            # open the file
            with open(file, encoding='latin-1') as opened_file:
                header_string = opened_file.readlines()[1][2:]

                # convert header into a dict
                curr_header = json.loads(header_string)

                # get the key 'internal sensor' or 'sensores internos'

                dict_key_alt = list(curr_header.keys())[0]

                if "ANDROID" in file:
                # add the fields from the header of the current file
                    print(header[dict_key]['sensor'])
                    header[dict_key]['sensor'].extend(curr_header[dict_key_alt]['sensor'])  # sensor field
                    header[dict_key]['column'].extend(
                    curr_header[dict_key_alt]['column'][1:])  # column field

                    # get the last channel from the channel field
                    num_channels = header[dict_key]['channels'][-1]

                    # get the channels from the current sensor
                    new_channels = curr_header[dict_key_alt]['channels']


                    # adjust the channel number
                    new_channels = [ch + (num_channels + 1) for ch in new_channels]


                    header[dict_key]['channels'].extend(new_channels)  # channels field
                    header[dict_key]['label'].extend(
                    curr_header[dict_key_alt]['label'])  # label field
                    header[dict_key]['resolution'].extend(
                    curr_header[dict_key_alt]['resolution'])  # resolution field
                    header[dict_key]['special'].extend(
                    curr_header[dict_key_alt]['special'])  # special field
                    header[dict_key]['sleeve color'].extend(
                    curr_header[dict_key_alt]['sleeve color'])  # sleeve color field
                    header[dict_key]['sampling rate'] = sampling_rate # changing sampling rate to the one specified by the
                                                                          # user
                if "WEAR" in file:
                    # add the fields from the header of the current file
                    header[dict_key]['sensor'].extend(curr_header[dict_key_alt]['sensor'])  # sensor field
                    header[dict_key]['column'].extend(curr_header[dict_key_alt]['label'])  # column field

                    # get the last channel from the channel field
                    num_channels = header[dict_key]['channels'][-1]


                    # get the channels from the current sensor
                    new_channels = curr_header[dict_key_alt]['channels']
                    new_labels = curr_header[dict_key_alt]['label']

                    # adjust the channel number
                    new_channels = [ch + (num_channels + 1) for ch in range(len(new_labels))]



                    header[dict_key]['channels'].extend(new_channels)  # channels field
                    header[dict_key]['label'].extend(curr_header[dict_key_alt]['label'])  # label field
                    header[dict_key]['resolution'].extend(curr_header[dict_key_alt]['resolution'])  # resolution field
                    header[dict_key]['special'] = ""  # special field
                    header[dict_key]['sleeve color'].extend(
                            curr_header[dict_key_alt]['sleeve color'])  # sleeve color field
                    header[dict_key]['sampling rate'] = sampling_rate  # changing sampling rate to the one specified by the
                                    # user

    # create new header string
    header_string = "# OpenSignals Text File Format\n# " + json.dumps(header) + '\n# EndOfHeader\n'

    return header_string


def re_sample_data(time_axis, data, start=0, stop=-1, shift_time_axis=False, sampling_rate=None,
                            kind_interp='linear'):
    # function to re-sample android sensor data from a non-equidistant sampling to an equidistant sampling
    # Parameters
    # ----------
    # time_axis (N, array_like): A 1D array containing the original time axis of the data

    # data (...,N,..., array_like): A N-D array containing data columns that are supposed to be interpolated.
    #                              The length of data along the interpolation axis has to be the same size as time.

    # start (int, optional): The sample from which the interpolation should be started. When not specified the
    #                       interpolation starts at 0. When specified the signal will be cropped to this value.

    # stop (int, optional): The sample at which the interpolation should be stopped. When not specified the interpolation
    #                      stops at the last value. When specified the signal will be cropped to this value.

    # shift_time_axis (bool, optional): If true the time axis will be shifted to start at zero and will be converted to seconds.

    # sampling_rate (int, optional): The sampling rate in Hz to which the signal should be re-sampled.
    #                               The value should be > 0.
    #                               If not specified the signal will be re-sampled to the next tens digit with respect to
    #                               the approximate sampling rate of the signal (i.e. approx. sampling of 99.59 Hz will
    #                               be re-sampled to 100 Hz).

    # kind_interp (string, optional): Specifies the kind of interpolation method to be used as string.
    #                                If not specified, 'linear' interpolation will be used.
    #                                Available options are: ‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
    #                                ‘previous’, ‘next’.

    # Returns
    # -------

    # the new time_axis, the interpolated data, and the sampling rate

    # crop the data and time to specified start and stop values
    if start != 0 or stop != -1:
        time_axis = time_axis[start:stop]

        # check for dimensionality of the data
        if data.ndim == 1:  # 1D array

            data = data[start:stop]

        else:  # multidimensional array

            data = data[start:stop, :]

    # get the original time origin
    time_origin = time_axis[0]

    # shift time axis (shifting is done in order to simplify the calculations)
    time_axis = time_axis - time_origin
    time_axis = time_axis * 1e-9

    # calculate the approximate sampling rate and round it to the next tens digit
    if sampling_rate is None:
        # get the average sampling rate
        sampling_rate = _calc_avg_sampling_rate(time_axis)

    # create new time axis
    time_inter = np.arange(time_axis[0], time_axis[-1], 1 / sampling_rate)

    # check for the dimensionality of the data array.
    if data.ndim == 1:  # 1D array

        # create the interpolation function
        inter_func = scpi.interp1d(time_axis, data, kind=kind_interp)

        # calculate the interpolated column and save it to the correct column of the data_inter array
        data_inter = inter_func(time_inter)

        # truncate the interpolated data
        data_inter = np.array([value for value in data_inter])

    else:  # multidimensional array

        # create dummy array
        data_inter = np.zeros([time_inter.shape[0], data.shape[1]])

        # cycle over the columns of data
        for col in range(data.shape[1]):
            # create the interpolation function
            inter_func = scpi.interp1d(time_axis, data[:, col], kind=kind_interp)

            # calculate the interpolated data
            di = inter_func(time_inter)

            # truncate the interpolated data and save the data to the correct column of the dat_inter array
            data_inter[:, col] = np.array([value for value in di])

    # check if time is not supposed to be shifted
    if not shift_time_axis:
        # shift back
        time_inter = time_inter * 1e9
        time_inter = time_inter + time_origin
    else:

        # calculate the precision needed
        precision = _calc_time_precision(sampling_rate)

        # truncate the time axis to the needed precision
        time_inter = [_truncate_time(value, precision) for value in time_inter]

    # return the interpolated time axis and data
    return time_inter, data_inter, sampling_rate

def synchronise_signals(in_signal_1, in_signal_2, time_interval = -1, fs = 100):
    """
    -----
    Brief
    -----
    This function synchronises the input signals using the full cross correlation function between the signals.
    -----------
    Description
    -----------
    Signals acquired with two devices may be dephased. It is possible to synchronise the two signals by multiple
    methods. Here, it is implemented a method that uses the calculus of the cross-correlation between those signals and
    identifies the correct instant of synchrony.
    This function synchronises the two input signals and returns the dephasing between them, and the resulting
    synchronised signals.
    ----------
    Parameters
    ----------
    in_signal_1 : list or numpy.array
        One of the input signals.
    in_signal_2 : list or numpy.array
        The other input signal.
    Returns
    -------
    phase : int
        The dephasing between signals in data points.
    result_signal_1: list or numpy.array
        The first signal synchronised.
    result_signal_2: list or numpy.array
        The second signal synchronised.
    """

    # signal segmentation
    in_signal_1 = in_signal_1[:time_interval*fs]
    in_signal_2 = in_signal_2[:time_interval*fs]

    #in_signal_2 = in_signal_2 - gravitational_filter(in_signal_2, fs)
    in_signal_1 = in_signal_1 * (-1)

    #in_signal_1[time_array[0] * fs:time_array[1] * fs] = in_signal_1[time_array[0] * fs:time_array[1] * fs] + 200
    #in_signal_2[time_array[4] * fs:time_array[5] * fs] = in_signal_2[time_array[4] * fs:time_array[5] * fs] + 200
    #in_signal_1[time_array[2] * fs:time_array[3] * fs] = in_signal_1[time_array[2] * fs:time_array[3] * fs] + 200
    #in_signal_2[time_array[6] * fs:time_array[7] * fs] = in_signal_2[time_array[6] * fs:time_array[7] * fs] + 200


    # signal normalisation
    mean_1, std_1, mean_2, std_2 = [np.mean(in_signal_1), np.std(in_signal_1), np.mean(in_signal_2),
                                    np.std(in_signal_2)]
    signal_1 = in_signal_1 - mean_1
    signal_1 /= std_1
    signal_2 = in_signal_2 - mean_2
    signal_2 /= std_2


    # zero padding signals so that they are of same length, this facilitates the calculation because
    # then the delay between both signals can be directly calculated
    # zero padding only if needed
    #if (len(signal_1) != len(signal_2)):

        # check which signal has to be zero padded
    #    if (len(signal_1) < len(signal_2)):

            # pad first signal
    #        signal_1 = np.append(signal_1, np.zeros(len(signal_2) - len(signal_1)))

    #    else:

            # pad second signal
    #        signal_2 = np.append(signal_2, np.zeros(len(signal_1) - len(signal_2)))


    N = len(signal_1) + len(signal_2) - 1
    # Calculate the cross-correlation between the two signals.
    #correlation = np.correlate(signal_1, signal_2, 'full')
    f1 = fft(signal_1, N)
    f2 = np.conj(fft(signal_2, N))
    correlation = np.real(ifft(f1 * f2))
    #correlation = fftshift(cc)


    # calculate tau / shift between both signals
    #tau = int(np.argmax(correlation) - (len(correlation)) / 2)
    tau = np.argmax(correlation)
    print(tau)
    if tau > len(correlation) // 2:
        tau = np.argmax(correlation) - len(correlation)
    print(tau)

    # crop signals to original length (removing zero padding)
    #signal_1 = signal_1[:len(in_signal_1)]
    #signal_2 = signal_2[:len(in_signal_2)]


    # check which signal has to be sliced
    if (tau < 0):
        # tau negative --> second signal lags
        signal_2 = signal_2[np.abs(tau):]

    elif (tau > 0):
        # tau positive ---> firs signal lags
        signal_1 = signal_1[np.abs(tau):]


    # revert signals to orignal scale
    result_signal_1 = signal_1 * std_1 + mean_1
    result_signal_2 = signal_2 * std_2 + mean_2

    return tau, result_signal_1, result_signal_2

def generate_sync_txt_file(in_path, time_array, channels=("CH1", "CH1"), new_path='sync_file.txt'):
    """
    -----
    Brief
    -----
    This function allows to generate a text file with synchronised signals from the input file(s).
    -----------
    Description
    -----------
    OpenSignals files follow a specific structure that allows to analyse all files in the same way. Furthermore, it
    allows those files to be opened and analysed in the OpenSignals software without the need of programming.
    This functions takes one or two files, synchronises the signals in channels and generates a new file in the new
    path.
    ----------
    Parameters
    ----------
    in_path : str or list
        If the input is a string, it is assumed that the two signals are in the same file, else, if the input is a list,
        it is assumed that the two signals are in different file (the list should contain the paths to the two files).
    channels : list
        List with the strings identifying the channels of each signal. (default: ("CH1", "CH1"))
    new_path : str
        The path to create the new file. (default: 'sync_file.txt')
    """
    if type(in_path) is str:
        _create_txt_from_str(in_path, channels, new_path)
    elif type(in_path) is list:
        print('1')
        _create_txt_from_list(in_path, time_array, channels, new_path)
        print('2')
    else:
        raise TypeError('The path should be a list of str or a str.')

def _shape_array(array1, array2, dephase):
    """
    Function that equalises the input arrays by padding the shortest one using padding type 'same', i.e. replicating
    the last row.
    ----------
    Parameters
    ----------
    array1: list or numpy.array
        Array
    array2: list or numpy.array
        Array
    Return
    ------
    arrays: numpy.array
        Array containing the equal-length arrays.
    """

    # check if the data arrays are of different size
    if (len(array1) != len(array2)):

        # check which array needs to be padded
        if (len(array1) < len(array2)):

            # get the length of the padding
            pad_length = len(array2) - len(array1)

            # pad the first array
            array1 = _pad_data(array1, pad_length)

        else:

            # get the length of the padding
            pad_length = len(array1) - len(array2)

            # pad the second array
            array2 = _pad_data(array2, pad_length)


    # hstack both arrays / concatenate both array horizontally
    arrays = np.hstack([array1, array2])

    return arrays

def _create_txt_from_str(in_path, channels, new_path):
    """
    This function allows to generate a text file with synchronised signals from the input file.
    ----------
    Parameters
    ----------
    in_path : str
        Path to the file containing the two signals that will be synchronised.
    channels : list
        List with the strings identifying the channels of each signal.
    new_path : str
        The path to create the new file.
    """
    header = ["# OpenSignals Text File Format"]
    files = [bsnb.load(in_path)]
    with open(in_path, encoding="latin-1") as opened_p:
        header.append(opened_p.readlines()[1])
    header.append("# EndOfHeader")

    data = []
    nr_channels = []
    is_integer_data = True
    for file in files:
        for i, device in enumerate(file.keys()):
            nr_channels.append(len(list(file[device])))
            data.append(file[device][channels[i]])

    dephase, s1, s2 = synchronise_signals(data[0], data[1])

    if data[0] is not int or data[1] is not int:
        is_integer_data = False

    # Avoid change in float precision if we are working with float numbers.
    if not is_integer_data:
        round_data_0 = [float('%.2f' % (value)) for value in data[0]]
        round_data_1 = [float('%.2f' % (value)) for value in data[1]]
        round_s_1 = [float('%.2f' % (value)) for value in s1]
        round_s_2 = [float('%.2f' % (value)) for value in s2]
    else:
        round_data_0 = data[0]
        round_data_1 = data[1]
        round_s_1 = s1
        round_s_2 = s2

    # Check which array is aligned.
    old_columns = np.loadtxt(in_path)
    if np.array_equal(round_s_1, round_data_0):
        # Change the second device
        aux = 3 * nr_channels[0]
        columns = old_columns[dephase:, aux:]
        new_file = _shape_array(old_columns[:, :aux], columns)
    elif np.array_equal(round_s_2, round_data_1):
        # Change the first device
        aux = 3 * nr_channels[1]
        columns = old_columns[dephase:, :aux]
        new_file = _shape_array(columns, old_columns[:, aux:])
    else:
        print("The devices are synchronised.")
        return

    # write header to file
    new_header = [h.replace("\n", "") for h in header]
    sync_file = open(new_path, 'w')
    sync_file.write(' \n'.join(new_header) + '\n')

    # write data to file
    for line in new_file:
        if is_integer_data:
            sync_file.write('\t'.join(str(int(i)) for i in line) + '\t\n')
        else:
            sync_file.write('\t'.join(str(i) for i in line) + '\t\n')
    sync_file.close()

def _available_channels(devices, header):
    """
    Function used for the determination of the available channels in each device.

    ----------
    Parameters
    ----------
    devices : list ["mac_address_1" <str>, "mac_address_2" <str>...]
        List of devices selected by the user.

    header: dict
        Dictionary that contains auxiliary data of the acquisition.

    Returns
    -------
    out : dict
        Returns a dictionary where each device defines a key and the respective value will be a list
        of the available channels for the device.

    """

    # ------------------------ Definition of constants and variables ------------------------------
    chn_dict = {}

    # %%%%%%%%%%%%%%%%%%%%%% Access to the relevant data in the header %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for dev in devices:
        chn_dict[dev] = header[dev]["column labels"].keys()

    return chn_dict

def _load_txt(file, devices, channels, header):
    """
    Function used for reading .txt files generated by OpenSignals.

    ----------
    Parameters
    ----------
    file : file, str, or pathlib.Path
        File, filename, or generator to read.  If the filename extension is
        ``.gz`` or ``.bz2``, the file is first decompressed. Note that
        generators should return byte strings for Python 3k.

    devices : list ["mac_address_1" <str>, "mac_address_2" <str>...]
        List of devices selected by the user.

    channels : list [[mac_address_1_channel_1 <int>, mac_address_1_channel_2 <int>...],
                    [mac_address_2_channel_1 <int>...]...]
        From which channels will the data be loaded.

    header : dict
        File header with relevant metadata for identifying which columns may be read.

    **kwargs : list of variable keyword arguments. The valid keywords are those used by
               numpy.loadtxt function.

    Returns
    -------
    out_dict : dict
        Data read from the text file.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%% Columns of the selected channels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out_dict = {}
    for dev_nbr, device in enumerate(devices):
        out_dict[device] = {}
        columns = []
        for chn in channels[device]:
            columns.append(header[device]["column labels"][chn])
            # header[device]["column labels"] contains the column of .txt file where the data of
            # channel "chn" is located.
            out_dict[device]["CH" + str(chn)] = np.loadtxt(fname=file, usecols=header[device]["column labels"][chn])

    return out_dict


def load(file):

    header = bsnb.read_header(file)
    dev_list = list(header.keys())
    chn_dict = _available_channels(dev_list, header)
    data = _load_txt(file, dev_list , chn_dict, header)

    return data


def _create_txt_from_list(in_path, time_array, channels, new_path):
    """
    This function allows to generate a text file with synchronised signals from the input files.
    ----------
    Parameters
    ----------
    in_path : list
        Paths to the files containing the two signals that will be synchronised.
    channels : list
        List with the strings identifying the channels of each signal.
    new_path : str
        Path to create the new file.
    """

    header = ["# OpenSignals Text File Format"]
    files = [load(p) for p in in_path]
    with open(in_path[0], encoding="latin-1") as opened_p:
        with open(in_path[1], encoding="latin-1") as opened_p_1:
            # append both headers
            header.append(opened_p.readlines()[1][:-2] + ', ' + opened_p_1.readlines()[1][3:])

    header.append("# EndOfHeader")

    print('2')
    # lists for holding the read data
    data = []
    nr_channels = []

    # read the data
    for i, file in enumerate(files):
        device = list(file.keys())
        nr_channels.append(len(list(file[device[0]])))
        data.append(file[device[0]][channels[i]])
    print('3')
    # calculate the delay between both signals
    dephase, _, _ = synchronise_signals(data[0], data[1],time_array, time_interval = 300, fs = 100)
    print('4')
    # load original data
    data_1 = np.loadtxt(in_path[0])
    data_2 = np.loadtxt(in_path[1])

    # Check which device lags
    if dephase < 0:

        # second device lags
        # slice the data
        data_2 = data_2[np.abs(dephase):]

    elif dephase > 0:

        # first device lags
        # slice the data
        data_1 = data_1[np.abs(dephase):]

    else:
        # dephase == 0 ---> devices were already syncronised
        print("The devices were already synchronised.")

    print(len(data_1))
    print(len(data_2))

    # pad data so that both devices are of the same length
    # in case that phase = 0 the data will only be concatenated horizontally
    print('5')
    new_file = _shape_array(data_1, data_2, dephase)
    print('6')

    # write header to file
    new_header = [h.replace("\n", "") for h in header]
    sync_file = open(new_path, 'w')
    sync_file.write('\n'.join(new_header) + '\n')

    # writing synchronised data to file
    for line in new_file:
        sync_file.write('\t'.join(str(i) for i in line) + '\t\n')

    # close the file
    sync_file.close()

def _pad_data(data, pad_length, padding_type='same'):
    """
    Function for padding data. The function uses padding type 'same', i.e. it replicates the last row of the data, as default
    ----------
    Parameters
    ----------
    data (numpy.array): the data that is supposed to be padded
    pad_length (int): the length of the padding that is supposed to be applied to data
    padding_type (string, optional): The type of padding applied, either: 'same' or 'zero'. Default: 'same'
    Return
    ------
    padded_data (numpy.array): The data with the padding
    """

    # get the sampling period (or distance between sampling points, for PLUX devices this is always 1)
    # it is assumed that the signals are equidistantly sampled therefore only the distance between to sampling points
    # is needed to calculate the sampling period
    T = data[:, 0][1] - data[:, 0][0]

    if padding_type == 'same':

        # create the 'same' padding array
        padding = np.tile(data[-1, 1:], (pad_length, 1))

    elif padding_type == 'zero':

        # get the number of columns for the zero padding
        num_cols = data.shape[1] - 1  # ignoring the time/sample column

        # create the zero padding array
        padding = np.zeros((pad_length, num_cols))

    else:

        IOError('The padding type you chose is not defined. Use either \'same\ or \'zero\'.')

    # create the time / sample axis that needs to be padded
    start = data[:, 0][-1] + T
    stop = start + (T * pad_length)
    time_pad = np.arange(start, stop, T)
    time_pad = time_pad[:pad_length]  # crop the array if there are to many values

    # expand dimension for hstack operation
    time_pad = np.expand_dims(time_pad, axis=1)

    # hstack the time_pad and the zero_pad to get the final padding array
    pad_array = np.hstack((time_pad, padding))

    # vstack the pad_array and the new_array
    padded_data = np.vstack([data, pad_array])

    return padded_data


def synchronize_Plux_Android(data_Plux, data, fs ,header_Plux, header_Android_Wear, save_path_Plux, save_path_Android_Wear):
    """
        This function allows to generate a text file with synchronised signals from the input files.
        ----------
        Parameters
        ----------
        in_path : list
            Paths to the files containing the two signals that will be synchronised.
        channels : list
            List with the strings identifying the channels of each signal.
        new_path : str
            Path to create the new file.
        """

    #Calculates the delay between the biosignalsPlux signal undersampled at 100Hz and
    # the Android signal sampled at 100Hz for the first 5 seconds of acquisition.
    dephase, _, _ = synchronise_signals(data_Plux[::10,5], data[:,2], time_interval = 300, fs = 100)
    # Check which device lags
    if dephase < 0:

        # second device lags
        # slice the data
        data = data[np.abs(dephase):]
        # creates a new time axis starting at zero seconds
        data[:, 0] = np.arange(0, len(data)*(1/fs), 1 / fs)
        header_string = "# OpenSignals Text File Format\n# " + json.dumps(header_Android_Wear) + '\n# EndOfHeader\n'

        # open a new file at the path location
        sync_file = open(save_path_Android_Wear, 'w')

        # write the header to the file
        sync_file.write(header_string)

        #  write the data to the file. The values in each line are written tab separated
        for row in data:
            sync_file.write('\t'.join(str(value) for value in row) + '\t\n')

        # close the file
        sync_file.close()

        header_string_P = "# OpenSignals Text File Format\n# " + json.dumps(header_Plux) + '\n# EndOfHeader\n'

        # open a new file at the path location
        sync_file = open(save_path_Plux, 'w')

        # write the header to the file
        sync_file.write(header_string_P)

        #  write the data to the file. The values in each line are written tab separated
        for row in data_Plux:
            sync_file.write('\t'.join(str(value) for value in row) + '\t\n')

        # close the file
        sync_file.close()

    elif dephase > 0:

        # first device lags
        # slice the data
        #as the lag was calculated for the two signals sampled at 100Hz to have the biosignals signals synchronized
        # with the remaining devices they are cut by tau*10 since this signal is sampled at 1000 Hz
        data_Plux = data_Plux[np.abs(dephase)*10:]
        data_Plux[:, 0] = np.arange(0, len(data_Plux)*(1/(fs*10)), 1 / (fs * 10))
        # convert header to string
        header_string = "# OpenSignals Text File Format\n# " + json.dumps(header_Plux) + '\n# EndOfHeader\n'

        # open a new file at the path location
        sync_file = open(save_path_Plux, 'w')

        # write the header to the file
        sync_file.write(header_string)

        #  write the data to the file. The values in each line are written tab separated
        for row in data_Plux:
            sync_file.write('\t'.join(str(value) for value in row) + '\t\n')

        # close the file
        sync_file.close()

    else:
        # dephase == 0 ---> devices were already syncronised
        print("The devices were already synchronised.")


