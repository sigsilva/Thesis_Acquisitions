from scipy.signal import filtfilt, sosfilt, medfilt, butter, lfilter
from numpy.lib.stride_tricks import as_strided as ast
import matplotlib.pyplot as plt
import math
import fathon
from fathon import fathonUtils as fu
import numpy as np
from   itertools import accumulate


'''
Functions for pre-processing the data.
Contains:
1. functions to apply filters
2. functions to plot data before and after filtering
'''



def gravitational_filter(acc_data, fs):
    """
    Function to filter out the gravitational component of ACC signals using a 3rd order butterworth lowpass filter with
    a cuttoff frequency of 0.3 Hz
    The implementation is based on: https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2013-84.pdf

    :param acc_data: a 1-D or (MxN) array, where where M is the signal length in samples and
                 N is the number of signals / channels.
    :param fs: the sampling frequency of the acc data.
    :return: the gravitational component of each signal/channel contained in acc_data
    """

    # define the filter
    order = 3
    f_c = 0.3
    filt = butter(order, f_c, fs=fs, output='sos')

    # copy the array
    gravity_data = acc_data.copy()

    # check the dimensionality of the input
    if gravity_data.ndim > 1: # (MxN) array

        # cycle of the channels contained in data
        for channel in range(gravity_data.shape[1]):

            # get the channel
            sig = acc_data[:, channel]

            # apply butterworth filter
            gravity_data[:, channel] = sosfilt(filt, sig)

    else: # 1-D array

        gravity_data = sosfilt(filt, acc_data)

    return gravity_data



def inertial_data_filter(acc_data, fs, medfilt_window_length=21):
    """
    function to filter the inertial and magnetic data. First a median filter is applied and then a 2nd order butterworth lowpass
    filter with a cutoff frequency of 10 Hz is applied.
    The filtering scheme is based on:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8567275
    :param acc_data: a 1-D or (MxN) array, where where M is the signal length in samples and
                 N is the number of signals / channels.
    :param fs: the sampling frequency of the acc data.
    :param medfilt_window_length: the length of the median filter. Has to be odd.
    :return: the filtered data
    """

    # define the filter
    order = 1
    f_c = 6
    b, a = butter(order, f_c, fs=fs, output='ba')

    # copy the array
    filtered_data = acc_data.copy()

    # check the dimensionality of the input
    if filtered_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(filtered_data.shape[1]):
            # get the channel
            sig = acc_data[:, channel]

            # apply the median filter
            #sig = medfilt(sig, medfilt_window_length)

            # apply butterworth filter
            #filtered_data[:, channel] = sosfilt(filt, sig)
            filtered_data[:, channel] = filtfilt(b, a, sig)

    else:  # 1-D array

        # apply median filter
        #med_filt = medfilt(acc_data, medfilt_window_length)

        # apply butterworth filter
        filtered_data = filtfilt(b,a, acc_data)

    return filtered_data


def bandpass(s, f1, f2, order=2, fs=1000.0, use_filtfilt=False):
    """
    -----
    Brief
    -----
    For a given signal s passes the frequencies within a certain range (between f1 and f2) and rejects (attenuates) the
    frequencies outside that range by applying a Butterworth digital filter.
    -----------
    Description
    -----------
    Signals may have frequency components of multiple bands. If our interest is to have an idea about the behaviour
    of a specific frequency band, we should apply a band pass filter, which would attenuate all the remaining
    frequencies of the signal. The degree of attenuation is controlled by the parameter "order", that as it increases,
    allows to better attenuate frequencies closer to the cutoff frequency. Notwithstanding, the higher the order, the
    higher the computational complexity and the higher the instability of the filter that may compromise the results.
    This function allows to apply a band pass Butterworth digital filter and returns the filtered signal.
    ----------
    Parameters
    ----------
    s: array-like
        signal
    f1: int
        the lower cutoff frequency
    f2: int
        the upper cutoff frequency
    order: int
        Butterworth filter order
    fs: float
        sampling frequency
    use_filtfilt: boolean
        If True, the signal will be filtered once forward and then backwards. The result will have zero phase and twice
        the order chosen.
    Returns
    -------
    signal: array-like
        filtered signal
    """
    b, a = butter(order, [f1 * 2 / fs, f2 * 2 / fs], btype='bandpass')

    # copy the array
    filtered_data = s.copy()

    # check the dimensionality of the input
    if filtered_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(filtered_data.shape[1]):
            # get the channel
            sig = s[:, channel]

            if use_filtfilt:
                # apply butterworth filter
                filtered_data[:, channel] = filtfilt(b, a, sig)

            filtered_data[:, channel] = lfilter(b, a, sig)
    else:  # 1-D array

        if use_filtfilt:
            # apply butterworth filter
            filtered_data = filtfilt(b, a, s)

        filtered_data = lfilter(b, a, s)

    return filtered_data

def RMS(signal, windowsize):
    q = windowsize
    # copy the array
    filtered_data = signal.copy()
    # check the dimensionality of the input
    if filtered_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(filtered_data.shape[1]):
            a2 = np.power(signal[:,channel], 2)
            window = np.ones(q) / float(q)
            filtered_data[:,channel] = np.sqrt(np.convolve(a2, window, "same"))
            '''
            a = signal[:,channel]
            c = [None] * len(signal[:,channel])

            for j in range(len(signal[:,channel])):
                b = 0
                for i in range(j - int(q/2),
                                j + int(q/2) + 1):
                    if(i < 0 or i > len(signal[:,channel])):
                        a[i] = 0
                    if(i >= len(signal[:,channel])):
                        a = np.append(a, 0)
                    b = b + (a[i])**2

                b = b/(q+1)
                b = math.sqrt(b)
                c[j] = b
            filtered_data[:,channel] = c
            '''

    else:  # 1-D array
        a2 = np.power(signal, 2)
        window = np.ones(q) / float(q)
        filtered_data = np.sqrt(np.convolve(a2, window, "same"))
        '''
        a = signal
        c = [None] * len(signal)
        for j in range(0, len(signal)):
            b = 0
            for i in range(j - int(q / 2),
                           j + int(q / 2) + 1):
                if (i < 0 or i > len(signal)):
                    a[i] = 0
                if (i >= len(signal)):
                    a = np.append(a, 0)
                b = b + (a[i]) ** 2

            b = b / (q + 1)
            b = math.sqrt(b)
            c[j] = b
        filtered_data = c
        '''

    return filtered_data

def moving_average(signal, window_size):
    i = 0
    moving_averages = []
    while i < len(signal) - window_size + 1:
        this_window = signal[i : i + window_size]


        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1

    return moving_averages

def apdf(signal, nbins):
    # Amplitude probability distribution function (APDF) #
    count, bins_count = np.histogram(signal, bins=nbins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)


    return count, bins_count, pdf, cdf

def angles_histogram(signal):

    array_counts = []
    array_bins = []
    for column in range(signal.shape[1]-1):
         hist, bin_edges = np.histogram(signal[:,column])
         array_counts.append(hist)
         array_bins.append(bin_edges)

    array_counts = np.vstack(array_counts)
    array_bins = np.vstack(array_bins)
    return array_counts, array_bins



def smooth(y, win_size):
    # copy the array
    smooth_data = y.copy()
    window = np.ones(win_size) / win_size

    # check the dimensionality of the input
    if smooth_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(smooth_data.shape[1]):
            smooth_data[:, channel] = np.convolve(y[:, channel], window, mode='same')

    else:
        smooth_data = np.convolve(y, window, mode='same')

    return smooth_data


def quaternation_to_euler_angles(signal, Mode=False):
    orientation_angles = signal.copy()
    # cycle of the rows contained in data
    for row in range(signal.shape[0]):
        if Mode == False:
            x = signal[row, 0]
            y = signal[row, 1]
            z = signal[row, 2]
            w = signal[row, 3]
        else:
            x = signal[row, 1]
            y = signal[row, 2]
            z = signal[row, 3]
            w = signal[row, 0]

        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2  # singularities from euler angles
        t2 = -1.0 if t2 < -1.0 else t2  # singularities from euler angles
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        orientation_angles[row, 0] = X
        orientation_angles[row, 1] = Y
        orientation_angles[row, 2] = Z

    return orientation_angles


def getRotationMatrixFromVector(rotation_vector):
    orientation_angles = rotation_vector.copy()

    # cycle of the rows contained in data
    for row in range(rotation_vector.shape[0]):
        R = np.zeros((3, 3))
        q0 = rotation_vector[row, 0]
        q1 = rotation_vector[row, 1]
        q2 = rotation_vector[row, 2]
        q3 = rotation_vector[row, 3]

        sq_q1 = 2 * q1 * q1
        sq_q2 = 2 * q2 * q2
        sq_q3 = 2 * q3 * q3
        q1_q2 = 2 * q1 * q2
        q3_q0 = 2 * q3 * q0
        q1_q3 = 2 * q1 * q3
        q2_q0 = 2 * q2 * q0
        q2_q3 = 2 * q2 * q3
        q1_q0 = 2 * q1 * q0

        R[0][0] = 1 - sq_q2 - sq_q3
        R[0][1] = q1_q2 - q3_q0
        R[0][2] = q1_q3 + q2_q0

        R[1][0] = q1_q2 + q3_q0
        R[1][1] = 1 - sq_q1 - sq_q3
        R[1][2] = q2_q3 - q1_q0

        R[2][0] = q1_q3 - q2_q0
        R[2][1] = q2_q3 + q1_q0
        R[2][2] = 1 - sq_q1 - sq_q2

        X = math.degrees(math.atan2(R[0][1], R[1][1]))
        Y = math.degrees(math.asin(-R[2][1]))
        Z = math.degrees(math.atan2(-R[2][0], R[2][2]))

        orientation_angles[row, 0] = X
        orientation_angles[row, 1] = Y
        orientation_angles[row, 2] = Z

    return orientation_angles

def chunk_data(data,window_size,overlap_size=0,flatten_inside_window=True):
   """
   Gives a matrix with all the windows of the signal separated by window size and overlap size.
   :param data:
   :param window_size:
   :param overlap_size:
   :param flatten_inside_window:
   :return: matrix with signal windowed based on window_size and overlap_size
   """
   WinRange = window_size // 2



   data = np.r_[data[WinRange:0:-1], data, data[-1:len(data) - WinRange:-1]]

   assert data.ndim == 1 or data.ndim == 2
   if data.ndim == 1:
       data = data.reshape((-1,1))

   # get the number of overlapping windows that fit into the data
   num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
   overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

   # if there's overhang, need an extra window and a zero pad on the data
   # (numpy 1.7 has a nice pad function I'm not using here)
   if overhang != 0:
       num_windows += 1
       newdata = np.zeros((num_windows*window_size - (num_windows-1)*overlap_size,data.shape[1]))
       newdata[:data.shape[0]] = data
       data = newdata

   sz = data.dtype.itemsize
   ret = ast(
       data,
       shape=(num_windows,window_size*data.shape[1]),
       strides=((window_size-overlap_size)*data.shape[1]*sz, sz)
   )

   if flatten_inside_window:
       return ret

   else:
       return ret.reshape((num_windows,-1,data.shape[1]))

def DFA(b):

    array = []
    for window in b:
        # zero-mean cumulative sum
        mod = fu.toAggregated(window)

        # initialize dfa object
        pydfa = fathon.DFA(mod)

        # compute fluctuation function and Hurst exponent
        lowW = 16
        HighW = len(window) / 4  # len(time series)/4
        wins = fu.linRangeByStep(lowW, HighW)
        n, F = pydfa.computeFlucVec(wins, revSeg=True, polOrd=2)
        H, H_intercept = pydfa.fitFlucVec()
        #plt.scatter(np.log(F), np.log(n), color='k')
        tex = round(H, 2)
        array.append(tex)
        #tex = str(tex)
        #plt.text(4.0, 4.0, s='Hurst = ' + tex, fontsize=16, fontweight='bold')
        #plt.show()
        #print(H)
    return array

def calcultate_variation_coeff(b):

    mean= np.mean(b, axis=1)
    standard_deviation = np.std(b, axis=1)
    cv = (standard_deviation / mean) * 100

    return cv




def plot_data(data_array1, data_array2, file_name):

    fig1, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
    ax1.plot(data_array1[:, 0], data_array1[:, 1],
             data_array1[:, 0], data_array1[:, 2],
             data_array1[:, 0], data_array1[:, 3])
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Acceleration (m/s^2)')
    ax1.legend(['Phone - acc x-axis', 'Phone - acc y-axis', 'Phone - acc z-axis'], bbox_to_anchor=(1.05, 1.0))
    ax2.plot(data_array1[:, 0], data_array1[:, 4],
             data_array1[:, 0], data_array1[:, 5],
             data_array1[:, 0], data_array1[:, 6])
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.legend(['Phone - gyr x-axis', 'Phone - gyr y-axis', 'Phone - gyr z-axis'],bbox_to_anchor=(1.05, 1.0))
    ax3.plot(data_array1[:, 0], data_array1[:, 7],
             data_array1[:, 0], data_array1[:, 8],
             data_array1[:, 0], data_array1[:, 9])
    ax3.set_xlabel('Time(s)')
    ax3.set_ylabel('Magnetic Field (¬µT)')
    ax3.legend(['Phone - mag x-axis', 'Phone - mag y-axis', 'Phone - mag z-axis'],bbox_to_anchor=(1.05, 1.0))
    ax4.plot(data_array1[:, 0], data_array1[:, 11],
             data_array1[:, 0], data_array1[:, 12],
             data_array1[:, 0], data_array1[:, 13],
             data_array1[:, 0], data_array1[:, 14])
    ax4.set_xlabel('Time(s)')
    ax4.set_ylabel('Rotation Vector')
    ax4.legend(['Phone - Rot x-axis', 'Phone - Rot y-axis', 'Phone - Rot z-axis', 'Phone - Scalar Rot'],bbox_to_anchor=(1.05, 1.0))
    ax5.plot(data_array1[:, 0], data_array1[:, 16],
             data_array1[:, 0], data_array1[:, 17],
             data_array1[:, 0], data_array1[:, 18])
    ax5.set_xlabel('Time(s)')
    ax5.set_ylabel('Acceleration (m/s^2)')
    ax5.legend(['Watch - acc x-axis ', 'Watch - acc y-axis ', 'Watch - acc z-axis '],bbox_to_anchor=(1.05, 1.0))
    ax6.plot(data_array1[:, 0], data_array1[:, 19],
             data_array1[:, 0], data_array1[:, 20],
             data_array1[:, 0], data_array1[:, 21])
    ax6.set_xlabel('Time(s)')
    ax6.set_ylabel('Angular Velocity (rad/s)')
    ax6.legend(['Watch - gyr x-axis ', 'Watch - gyr y-axis ', 'Watch - gyr z-axis '],bbox_to_anchor=(1.05, 1.0))
    ax7.plot(data_array1[:, 0], data_array1[:, 22],
             data_array1[:, 0], data_array1[:, 23],
             data_array1[:, 0], data_array1[:, 24])
    ax7.set_xlabel('Time(s)')
    ax7.set_ylabel('Magnetic Field (¬µT)')
    ax7.legend(['Watch - mag x-axis', 'Watch - mag y-axis', 'Watch - mag z-axis'],bbox_to_anchor=(1.05, 1.0))
    ax8.plot(data_array1[:, 0], data_array1[:, 25],
             data_array1[:, 0], data_array1[:, 26],
             data_array1[:, 0], data_array1[:, 27],
             data_array1[:, 0], data_array1[:, 28])
    ax8.set_xlabel('Time(s)')
    ax8.set_ylabel('Rotation Vector')
    ax8.legend(['Watch - Rot x-axis', 'Watch - Rot y-axis', 'Watch - Rot z-axis', 'Watch - Scalar Rot'],bbox_to_anchor=(1.05, 1.0))

    fig2, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(data_array2[:, 0], data_array2[:, 4],
             data_array2[:, 0], data_array2[:, 5],
             data_array2[:, 0], data_array2[:, 6])
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Acceleration (m/s^2)')
    ax1.legend(['Plux - acc x-axis', 'Plux - acc y-axis', 'Plux - acc z-axis'], bbox_to_anchor=(1.05, 1.0))
    ax2.plot(data_array2[:, 0], data_array2[:, 2],
             data_array2[:, 0], data_array2[:, 3])
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Eletric Tension (mv)')
    ax2.legend(['EMG - right', 'EMG - left'], bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()
    #return fig1.savefig(file_name + '1'+ '.png', dpi = 1200, bbox_inches="tight"), fig2.savefig(file_name + '2' + '.png', dpi = 1200, bbox_inches="tight")
    return plt.show()


