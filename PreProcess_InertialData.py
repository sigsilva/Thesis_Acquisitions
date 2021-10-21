import matplotlib.pyplot as plt
import numpy as np
import glob
import fathon
from fathon import fathonUtils as fu
from aux_functions_2 import inertial_data_filter, bandpass, smooth, plot_data, quaternation_to_euler_angles, chunk_data, angles_histogram, getRotationMatrixFromVector
from aux_functions_2 import calcultate_variation_coeff, DFA
from bokeh.plotting import figure, show, output_file, gridplot
from scipy.spatial.transform import Rotation as R
from ahrs.filters import Madgwick

path_android_wear_sync = '/Users/sarasilva/Desktop/Thesis_Acquisitions/Synchronised_DATA/android_wear_biosignalsPLUX_sync/'
path_plux_sync = '/Users/sarasilva/Desktop/Thesis_Acquisitions/Synchronised_DATA/PLUX_converted/'

file_list_android_wear = glob.glob(path_android_wear_sync + '*')
file_list_android_wear.sort()

config_time = {'JR': 193, 'MIH' : 139, 'FG' : 185, 'SM' : 146 ,
               'CC' : 185 , 'MAH' : 166, 'IM' : 146, 'PP': 145,
               'MD': 201, 'BB' : 130, 'NM' : 166, 'CQ': 146 }

initial_time = 205
final_time = 3205

for file in file_list_android_wear:
    #subject = (file.split('/')[-1]).split('_')[0]
    data = np.loadtxt(file)
    # read the dictionary and look up the range of values for synchronization
    #cut_time = config_time[subject]
    # stores the data of that sensor in a variable assigned to each sensor.
    acc_phone = data[:, 1:4]
    gyr_phone = data[:, 4:7]
    # pass the magnetometer values to milliTesla as required by the filter
    mag_phone = data[:, 7:10]*(1e-3)
    rotation_vector_data = data[:, 11:15]
    print('1')

    # application of filters for signal pre-processing

    acc_phone = inertial_data_filter(acc_phone, 100)
    print(len(acc_phone))
    gyr_phone = inertial_data_filter(gyr_phone, 100)
    gyr_phone = smooth(gyr_phone, 100)
    mag_phone = inertial_data_filter(mag_phone, 100)
    print('2')





    # application of filters for sensor fusion and obtaining the quaternions that represent the orientations in space
    madgwick = Madgwick(gyr=gyr_phone, acc=acc_phone, mag=mag_phone, q0 = [1,0,0,0])
    orientation_from_mf = quaternation_to_euler_angles(madgwick.Q, Mode=True)
    #orientation_from_rv = getRotationMatrixFromVector(rotation_vector_data)
    #r3 = R.from_quat(madgwick.Q)
    #orientation_from_mf = r3.as_euler('xyz', degrees=True)
    #r4 = R.from_quat(rotation_vector_data)
    #orientation_from_rv = r4.as_euler('xyz', degrees=True)
    #orientation_from_rv = quaternation_to_euler_angles(rotation_vector_data)
    '''
    print('3')

    print('mean x')
    print(np.mean(orientation_from_mf[initial_time * 100:final_time * 100, 0]))
    print(np.std(orientation_from_mf[initial_time * 100:final_time * 100, 0]))
    print('mean y')
    print(np.mean(orientation_from_mf[initial_time * 100:final_time * 100, 1]))
    print(np.std(orientation_from_mf[initial_time * 100:final_time * 100, 1]))
    print('mean z')
    print(np.mean(orientation_from_mf[initial_time * 100:final_time * 100, 2]))
    print(np.std(orientation_from_mf[initial_time * 100:final_time * 100, 2]))

    array = [orientation_from_mf[initial_time * 100:final_time * 100, 0],
             orientation_from_mf[initial_time * 100:final_time * 100, 1],
             orientation_from_mf[initial_time * 100:final_time * 100, 2]]
    print(array)

    #max = str(int(np.max(array)))
    #min= str(int(np.min(array)))
    
    fig, ax = plt.subplots()
    im1 = plt.imshow(array, cmap='YlGnBu', vmin = np.min(array), vmax = np.max(array),  aspect='auto')
    ax.xaxis.set_ticks([0, 50000, 100000, 150000, 200000, 250000, 300000])
    ax.set_xticklabels([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.axes.yaxis.set_visible(False)
    ax.set_xlabel('Time(s)')


    plt.colorbar(im1, ax=ax, orientation='horizontal')
    #bar.set_ticks([int(np.min(array)),  int(np.max(array))])
    #bar.set_ticklabels([min, max])
    

    plt.show()

    '''
    time_axis = data[initial_time * 100:final_time * 100, 0] - data[initial_time * 100, 0]
    fig, ax = plt.subplots()
    #ax1.plot(time_axis, orientation_from_rv[initial_time * 100:final_time * 100,0], label='Pitch', color='darkseagreen')
    #ax1.plot(time_axis, orientation_from_rv[initial_time * 100:final_time * 100,1], label='Yaw', color='darkorange')
    #ax1.plot(time_axis, orientation_from_rv[initial_time * 100:final_time * 100,2], label='Roll', color='dodgerblue')
    #ax1.spines["top"].set_visible(False)
    #ax1.spines["right"].set_visible(False)
    #ax1.spines["left"].set_visible(False)
    #ax1.spines["bottom"].set_visible(False)
    #ax1.set_facecolor('snow')
    #ax1.set_xlabel('Time(s)')
    #ax1.set_ylabel('Angle(°)')
    #ax1.legend()
    ax.plot(time_axis, orientation_from_mf[initial_time * 100:final_time * 100,0], label='Pitch', color='darkseagreen')
    ax.plot(time_axis, orientation_from_mf[initial_time * 100:final_time * 100,1], label='Roll', color='darkorange')
    ax.plot(time_axis, orientation_from_mf[initial_time * 100:final_time * 100,2], label='Yaw', color='dodgerblue')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_facecolor('snow')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Angle(°)')
    ax.legend()

    ax.grid(color='lightgrey')
    #ax1.title.set_text('APDF')
    #ax2.grid(color='lightgrey')
    #ax2.title.set_text('APDF')

    #fig.suptitle("Torso' angular position", fontsize=14)
    plt.show()

    '''
    # 1 minute window
    window_size = len(orientation_from_mf)/50

    bx = chunk_data(orientation_from_mf[:,0], int(window_size), int(window_size)-1)
    by = chunk_data(orientation_from_mf[:,1], int(window_size), int(window_size)-1)
    bz = chunk_data(orientation_from_mf[:,2], int(window_size), int(window_size)-1)
    print('5')

    cx = np.sum(np.abs(np.diff(bx, axis=1)), axis=1)
    print('6')
    cy = np.sum(np.abs(np.diff(by, axis=1)), axis=1)
    cz = np.sum(np.abs(np.diff(bz, axis=1)), axis=1)


    
    fig, ((ax1), (ax2), (ax3)) = plt.subplots(3,1)
    # calculate dfa
    dfa_x = DFA(bx)
    ax1.plot(np.arange(1,51,1), dfa_x[1:])
    dfa_y = DFA(by)
    ax2.plot(np.arange(1,51,1), dfa_y[1:])
    dfa_z = DFA(bz)
    ax3.plot(np.arange(1,51,1), dfa_z[1:])
    plt.show()
    


    
    # Histogram with angular intervals and its frequency
    #counts, bins = angles_histogram(orientation_from_mf[initial_time*100:final_time*100])


    #round_list1 = [round(num) for num in bins[0,:]]
    #round_list2 = [round(num) for num in bins[1,:]]
    #round_list3 = [round(num) for num in bins[2,:]]

    #labels1 = str(round_list1[0:2]), str(round_list1[1:3]), str(round_list1[2:4]), str(round_list1[3:5]), str(round_list1[4:6]), str(round_list1[5:7]), \
    #          str(round_list1[6:8]), str(round_list1[7:9]), str(round_list1[8:10]), str(round_list1[9:])
    #labels2 = str(round_list2[0:2]), str(round_list2[1:3]), str(round_list2[2:4]), str(round_list2[3:5]), str(
    #    round_list2[4:6]), str(round_list2[5:7]), str(round_list2[6:8]), str(round_list2[7:9]), str(
    #    round_list2[8:10]), str(round_list2[9:])
    #labels3 = str(round_list3[0:2]), str(round_list3[1:3]), str(round_list3[2:4]), str(round_list3[3:5]), str(
    #    round_list3[4:6]), str(round_list3[5:7]), str(round_list3[6:8]), str(round_list3[7:9]), str(
    #    round_list3[8:10]), str(round_list3[9:])


    labels2 = '[-45,-10]', ' [-10,10]', '[10,45]'
    labels1 = '[50,95]', '[95,110]', '[110,150]'



    count_twist, nbins_twist = np.histogram(orientation_from_mf[initial_time*100:final_time*100,1], bins = [-45,-10,10,45])
    count_bending, nbins_bending = np.histogram(orientation_from_mf[initial_time * 100:final_time * 100,0],bins=[50, 95, 110, 150])
    #print(count_twist)
    #print(nbins_twist)




    time_x = (count_bending/sum(count_bending))*100
    time_y = (count_twist/sum(count_twist))*100

    labels2 = ['%s, %.2f %%' % (l, s) for l, s in zip(labels2, time_y)]
    colors = ['darkred','forestgreen','red']
    #time_x = (counts[0,:]/100)*100
    #time_y = (counts[1,:]/100)*100
    #time_z = (counts[2,:]/100)*100

    
    fig, ((ax1), (ax2), (ax3)) = plt.subplots(3,1, sharex='col')
    ax1.bar(bins[0,:-1], counts[0,:], width=np.diff(bins[0,:]), edgecolor="black", align="edge")
    ax1.set_ylabel('Frequency')
    ax2.bar(bins[1, :-1], counts[1, :], width=np.diff(bins[1,:]), edgecolor="black", align="edge")
    ax2.set_ylabel('Frequency')
    ax3.bar(bins[2, :-1], counts[2, :], width=np.diff(bins[2, :]), edgecolor="black", align="edge")
    ax3.set_xlabel('Angles (degrees)')
    ax3.set_ylabel('Frequency')

    ax1.title.set_text('Pitch')
    ax2.title.set_text('Yaw')
    ax3.title.set_text('Roll')
    
    #plt.bar(bins[1, :-1], counts[1, :], width=np.diff(bins[1, :]), edgecolor="black", align="edge")
    #plt.ylabel('Frequency')
    #plt.title('Yaw')
    #plt.xlabel('Angles (degrees)')
    #plt.show()
    
    fig1, ((ax1,ax2)) = plt.subplots(1,2)
    pie1 = ax1.pie(time_x, shadow=True, colors = colors, startangle=90)
    ax1.legend(pie1[0], labels1, bbox_to_anchor=(0, 0.85), loc=2, fontsize=10,
               bbox_transform=plt.gcf().transFigure, title = 'Angles Intervals (degrees)')
    ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Pitch")
    pie2 = ax2.pie(time_y, shadow=True, colors = colors, startangle=90)
    ax2.legend(pie2[0], labels2, bbox_to_anchor=(1, 0.85), loc=1, fontsize=10,
               bbox_transform=plt.gcf().transFigure, title = 'Angles Intervals (degrees)')
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title("Yaw")
    #pie3 = ax3.pie(time_z, shadow=True, startangle=90)
    #ax3.legend(pie3[0], labels3, bbox_to_anchor=(1, 0.5), loc="center right", fontsize=10,
    #           bbox_transform=plt.gcf().transFigure, title = 'Angles Intervals (degrees)')
    #ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    #ax3.set_title("Roll")


    plt.show()
    '''







