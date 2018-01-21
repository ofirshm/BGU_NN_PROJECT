import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from scipy import stats
import math
from Util import baseline_als


class Cycle:

    def __init__(self,volt,current):
        self.volt = volt
        self.current = current
        self.middle = math.floor(len(volt)/2)
        self.ox_volt = volt[0:self.middle]
        self.ox_current = current[0:self.middle]
        self.red_volt = volt[self.middle:-1]
        self.red_current = current[self.middle:-1]

    def get_volt(self):
        return self.volt

    def get_current(self):
        return self.current

    def get_oxidation_peak(self,electrode_number):
        if electrode_number==1:
            first_index= 1000          #First Index for the electrode
            last_index=   1450              #Last Index for the electrode
            corrected_current= self.ox_current[first_index:last_index]
        if electrode_number==2:
            first_index=    600       #First Index for the electrode
            last_index=    1000             #Last Index for the electrode
            corrected_current= self.ox_current[first_index:last_index]

            # solution = plt.plot(self.ox_current, label="solution")
        # correction = plt.plot(self.ox_current - baseline_als(self.ox_current, 10e6, 0.06), label = "correction")
        # plt.legend()
        # plt.show()
        peak_index = np.argmax(corrected_current)
        inline_factor=20#Arbitrary factor for Inline calculation
        peak_index+=first_index # The original location of the peak has to move
        try:
            current_inline1 = (self.ox_current[peak_index] - self.ox_current[peak_index - inline_factor]) / (self.ox_volt[peak_index] - self.ox_volt[peak_index - inline_factor])
        except(IndexError):
            current_inline1 = 1
        try:
            current_inline2=(self.ox_current[peak_index]-self.ox_current[peak_index+inline_factor])/(self.ox_volt[peak_index]-self.ox_volt[peak_index+inline_factor])
        except(IndexError):
            current_inline2 = 1
        return [self.ox_current[peak_index],self.ox_volt[peak_index],current_inline1,current_inline2]

    def get_reduction_peak(self):
        lambd = 10e6 # 10e2 - 10e9 smoothness
        p = 0.06 # e-3 - e-1 asymmetry
        corrected_current= self.red_current-baseline_als(self.red_current,lambd,p)
        # plt.plot(self.red_current - baseline_als(self.red_current, 10e9, 0.001))
        # plt.show()
        peak_index = np.argmin(corrected_current)
        inline_factor=20
        try:
            current_inline1=(self.red_current[peak_index]-self.red_current[peak_index-inline_factor])/(self.red_volt[peak_index]-self.red_volt[peak_index-inline_factor])
        except(IndexError):
            current_inline1 = 1
        try:
            current_inline2 = (self.red_current[peak_index] - self.red_current[peak_index + inline_factor]) / (self.red_volt[peak_index] - self.red_volt[peak_index + inline_factor])
        except(IndexError):
            current_inline2 = 1
        return [self.red_current[peak_index],self.red_volt[peak_index],current_inline1,current_inline2]

    def get_n_avg_sampling_features(self,n):
        output_feature=[]
        cycle_length=len(self.current)
        if n<6 or n%2==1:
            raise Exception('feature number must be even and higher than 6')
        interval_number= int(n/2)
        interval_length= math.floor(cycle_length /interval_number)
        for interval in range(interval_number-1):
            working_interval_current=self.current[interval_length*interval:interval_length*(interval+1)]
            working_interval_volt= self.volt[interval_length*interval:interval_length*(interval+1)]
            avg_working_interval=np.average(working_interval_current)
            slope_working_interval= stats.linregress(working_interval_volt, working_interval_current)
            output_feature.append(avg_working_interval)
            output_feature.append(slope_working_interval.slope)
        return output_feature

    def get_features(self,features_type,electrode_number):
        if features_type==1:# Oxidation peak features
            return self.get_oxidation_peak(electrode_number)
        else:
            #return self.get_n_avg_sampling_features(16)#ORiginal
            return np.concatenate((self.get_oxidation_peak(electrode_number), self.get_n_avg_sampling_features(16)))
        #FOR plot



        #reduction_peak = self.get_reduction_peak()
        #sampling_size = features_size - (len(oxidation_peak) + len(reduction_peak))

       # if sampling_size > 6:
       #     return np.concatenate((oxidation_peak, reduction_peak, self.get_n_avg_sampling_features(sampling_size)))
       # return np.concatenate((oxidation_peak,reduction_peak))


    def get__first_features_set(self,features_size):

        oxidation_peak = self.get_oxidation_peak()
        sampling_size = features_size - (len(oxidation_peak) + len(reduction_peak))
        if sampling_size > 6:
            return np.concatenate((oxidation_peak, reduction_peak, self.get_n_avg_sampling_features(sampling_size)))
        return np.concatenate((oxidation_peak,reduction_peak))

    def plot_ox_current(self):
        plt.plot(self.ox_current)
        plt.show()

    def plot_red_current(self):
        plt.plot(self.red_current)
        plt.show()

    def plot_cycle(self):
        plt.plot(self.volt,self.current)
        plt.show()


    def plot_current(self):
        plt.plot(self.current)
        plt.show()



    def plot_compared_current(self,lambd,p):
        plt.rcParams.update({'font.size': 18})
        plt.figure(1)
        corrected_current= self.ox_current-baseline_als(self.ox_current,lambd,p)
        plt.plot(corrected_current,label='Corrected')
        plt.plot(self.ox_current,label='Current')
        plt.xlabel('V vs. Ag/AgCl')
        plt.ylabel('I/mA')
        plt.legend()
        plt.show()
