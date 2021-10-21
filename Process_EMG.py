import numpy as np
import glob
import matplotlib
from aux_functions_2 import bandpass, RMS, moving_average,apdf, chunk_data, smooth
from scipy import stats
import matplotlib.pyplot as plt
import statistics

path_plux_sync = '/Users/sarasilva/Desktop/Thesis_Acquisitions/Synchronised_DATA/PLUX_converted/'

file_list_plux = glob.glob(path_plux_sync + '*')
file_list_plux.sort()

low_cutoff = 20  # Hz
high_cutoff = 450  # Hz

initial_time = 205
final_time = 3205

for file in file_list_plux:
    #subject = (file.split('/')[-1]).split('_')[0]
    plux_data = np.loadtxt(file)
    # filter the biosignalsPlux's eletromiography signals
    # as use_filtfilt = True, the signal will be filtered once forward and then backwards.
    # The result will have zero phase and twice the order chosen.
    # based on article : https://journals.sagepub.com/doi/abs/10.1177/1541931218621222
    plux_data[:, 2:4] = bandpass(plux_data[:, 2:4], low_cutoff, high_cutoff, order=3, fs=1000, use_filtfilt=True)
    print('data filtered1')
    # data low_pass filtered are converted in Root Mean Square using a consecutive 500 ms moving window
    data_converted = RMS(plux_data[:, 2:4], 100)
    # For normalization of RMS data, the maximal RMS value during the MVC was calculated
    # with a 100 ms moving average window
    # the maximal RMS value during the MVC is calculated
    #amplitude_right = smooth(data_converted[:,0],100)
    #amplitude_left = smooth(data_converted[:,1],100)

    # without smoothing
    amplitude_right = data_converted[:,0]
    amplitude_left = data_converted[:,1]


    maximum_right = np.max(data_converted[:180000,0])
    maximum_left = np.max(data_converted[:180000,1])


    # time axis is zero-aligned
    time_axis = plux_data[initial_time*1000:final_time*1000,0] - plux_data[initial_time*1000,0]

    # normalization of amplitude
    envelope_normalized_right = (amplitude_right[initial_time*1000:final_time*1000]/maximum_right)*100
    #envelope_normalized_right = (amplitude_right/ maximum_right) * 100
    envelope_normalized_left = (amplitude_left[initial_time*1000:final_time*1000] /maximum_left)*100



    # ASSESSMENT PARAMETERS


    print('average right')
    average_right = np.mean(envelope_normalized_right)
    print(average_right)
    print('average left')
    average_left = np.mean(envelope_normalized_left)
    print(average_left)



    print('frequency right per minute')
    frequency_right = np.sum(envelope_normalized_right <= 0.5)/50
    print(frequency_right)
    print('frequency left per minute')
    frequency_left = np.sum(envelope_normalized_left <= 0.5)/50
    print(frequency_left)

    print('Muscle rest right')
    print((np.sum(envelope_normalized_right <= 0.5)/len(envelope_normalized_right))*100)
    print('Muscle rest left')
    print((np.sum(envelope_normalized_left <= 0.5)/len(envelope_normalized_left))*100)


    # use a masked array to suppress the values that are too low
    #a_masked_r = np.ma.masked_less_equal(envelope_normalized_right, 0.5)
    #a_masked_l = np.ma.masked_less_equal(envelope_normalized_left, 0.5)

    #window_size = int(len(envelope_normalized_right)/5)

    #br = chunk_data(envelope_normalized_right, window_size, 0)
    #bl =  chunk_data(envelope_normalized_left, window_size, 0)

    '''
    fig, ((ax1,ax2)) = plt.subplots(2,1)
    # plot the full line
    #ax1.plot(time_axis, envelope_normalized_right)
    #ax2.plot(time_axis,envelope_normalized_left)
    # make a color map of fixed colors
    #cmap = matplotlib.colors.ListedColormap(['white', 'red'])
    #bounds = [0, 0.5, 100]
    #norm = matplotlib.BoundaryNorm(bounds, cmap.N)
    im1 = ax1.imshow([envelope_normalized_right], cmap='YlGnBu',vmin = 0, vmax = 100, aspect='auto')
    ax1.xaxis.set_ticks([0,500000, 1000000, 1500000, 2000000, 2500000, 3000000])
    ax1.set_xticklabels([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax1.axes.yaxis.set_visible(False)
    ax1.set_xlabel('Time(s)')
    plt.colorbar(im1, ax = ax1, orientation = 'horizontal')

    im2 = ax2.imshow([envelope_normalized_left], cmap='YlGnBu',vmin = 0, vmax = 100, aspect = 'auto')
    ax2.xaxis.set_ticks([0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000])
    ax2.set_xticklabels([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax2.axes.yaxis.set_visible(False)
    ax2.set_xlabel('Time(s)')
    plt.colorbar(im2, ax = ax2, orientation = 'horizontal')


    plt.show()
    
    ax1.plot(plux_data[:100000,0],plux_data[:100000,2])
    ax1.set_ylabel('EMG(mV)')
    ax1.set_xlabel('Time(s)')
    ax1.set_facecolor('snow')
    ax2.plot(plux_data[:100000,0],envelope_normalized_right[:100000])
    ax2.set_ylabel('EMG(%MVC)')
    ax2.set_xlabel('Time(s)')
    ax2.set_facecolor('snow')
    #ax1.plot(envelope_normalized_right, 'r')
    #ax1.plot(a_masked_r, 'b')
    # plot only the large values
    #ax2.plot(envelope_normalized_left, 'r')
    #ax2.plot(a_masked_l, 'b')
    ax1.grid()
    ax2.grid()
    plt.show()
    '''

    


    
    #apply histogram in total signal
    count_right, bins_count_right, pdf_right, cdf_right = apdf(envelope_normalized_right, 100)
    count_left, bins_count_left, pdf_left, cdf_left = apdf(envelope_normalized_left, 100)


    print('static right')
    index1 = (np.abs(cdf_right - 0.1)).argmin()
    static_right = bins_count_right[1:][index1]
    print(static_right)

    print('static left')
    index2 = (np.abs(cdf_left - 0.1)).argmin()
    static_left = bins_count_left[1:][index2]
    print(static_left)

    print('median right')
    index3 = (np.abs(cdf_right - 0.5)).argmin()
    median_right = bins_count_right[1:][index3]
    print(median_right)

    print('median left')
    index4 = (np.abs(cdf_left - 0.5)).argmin()
    median_left = bins_count_left[1:][index4]
    print(median_left)

    print('peak right')
    index5 = (np.abs(cdf_right - 0.9)).argmin()
    peak_right = bins_count_right[1:][index5]
    print(peak_right)

    print('peak left')
    index6 = (np.abs(cdf_left - 0.9)).argmin()
    peak_left = bins_count_left[1:][index6]
    print(peak_left)

    '''
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col')
    # matplotlib histogram
    ax1.hist(bins_count_right[:-1], bins_count_right, weights = count_right)
    ax1.set_ylabel('Frequency')
    ax2.hist(bins_count_left[:-1],bins_count_left, weights = count_left)
    ax2.set_ylabel('Frequency')
    ax3.plot(bins_count_right[1:],pdf_right)
    ax3.set_ylabel('Probability')
    ax4.plot(bins_count_left[1:],pdf_left)
    ax4.set_ylabel('Probability')
    ax5.plot(bins_count_right[1:],cdf_right)
    ax5.set_xlabel('% MVC')
    ax5.set_ylabel('Probability')
    ax5.set_facecolor('snow')
    ax6.plot(bins_count_left[1:],cdf_left)
    ax6.set_xlabel('% MVC')
    ax6.set_ylabel('Probability')

    ax5.title.set_text('APDF')
    ax5.grid()
    plt.show()

    '''






