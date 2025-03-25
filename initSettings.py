# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sqlalchemy import true

class initSettings(object):
    """"
    Class initializes and saves settings. 
    All settings are described inside class code.
    settings = initSettings()
        Inputs:  none
        Outputs:
                 settings     - Receiver settings (a structure).
    """
    def __init__(self):
        #%%  Processing settings ===============================================
        # Number of milliseconds to be processed used 36000 + any transients (see
        # below - in Nav parameters) to ensure nav subframes are provided
        self.msToProcess         = 50000          #[ms]
        # Number of channels to be used for signal processing
        self.numberOfChannels    = 15
        # Move the starting point of processing. Can be used to start the 
        # signal processing at any point in the data record (e.g. for long 
        # records). fseek function is used to move the file read point, 
        # therefore advance is samplesbased only. 
        self.skipSamples     = int(0.)
        #%% Raw signal file name and other parameter ==========================
        # This is a "default" name of the data file (signal record) to be used in
        #the post-processing mode
        self.fileName           = './GPS_L1_CA_IF_signal.bin'
        # Data type used to store one sample
        self.dataType           = np.int8
        # File Types
        # 1 - 8 bit real samples S0,S1,S2,...
        # 2 - 8 bit I/Q samples I0,Q0,I1,Q1,I2,Q2,...                      
        self.fileType           = 1
        # Intermediate, sampling and code frequencies
        self.IF                 = 14.58e6       # [Hz]
        self.samplingFreq       = 53e6
        self.codeFreqBasis      = 1.023e6       # [Hz]
        # Define number of chips in in a code period.
        self.codeLength         = 1023          # [Chips]

        #%% Acquisition settings ==============================================
        # Enable use of GPU acceleration for acquisition
        self.gpuACQflag         = True
        # Skips acquisition  in the script postProcessing.m if set to True False
        self.skipAcquisition    = True
        # Use pickle data to skip acquisition
        self.acqPklPath         =  "acqResults.pkl"
        # List of satellites to look for. Some satellites can be excluded to
        self.acqSatelliteList   = np.array(np.arange(1,33) )  # [PRN numbers]
        # One-sided band around IF to search for satellite signal. 
        self.acqSearchBand      = 5000.              # [Hz]
        # Non-coherent integration times after 1ms coherent integration
        self.acqNonCohTime      = 10                # [ms]
        # Threshold for the signal presence decision rule
        self.acqThreshold       = 2.5
        # Frequency search step for coarse acquisition
        self.acqSearchStep      = 500                # [Hz]
         
        #%% Tracking loops settings =============================================
        # Skips tracking
        self.SIMDTrackflag          = False
        self.skipTracking           = False
        # Use pickle data to skip tracking
        self.trkPklPath             = "trackResults.pkl"
        # Enable/dissable use of SIMD/GPU MEX to accelerate tracking
        self.trkMode                = 0;     # 0-Python; 1-SIMD; 2-GPU
        # Code tracking loop parameters
        self.dllDampingRatio        = 0.7
        self.dllNoiseBandwidth      = 1.0         # [Hz]
        self.dllCorrelatorSpacing   = 0.5        # [chips]
        # Carrier tracking loop parameters
        self.pllDampingRatio        = 0.7 
        self.pllNoiseBandwidth      = 20.        # [Hz]
        # Integration time for DLL and PLL
        self.intTime                = 0.001       # [s]
        
        #%% Navigation solution settings ======================================
        # Skips navigation
        self.skipNavigation     = False
        # Use pickle data to skip tracking
        self.navPklPath         = "navSolutions_eph.pkl"
        # Period for calculating pseudoranges and position
        self.navSolPeriod       = 500.           # [ms]
        # Elevation mask to exclude signals from satellites at low elevation
        self.elevationMask      = 5.             # [degrees 0 - 90]
        # Enable/dissable use of tropospheric correction
        self.useTropCorr        = 1              # 0 - Off; 1 - On
        # True position of the antenna in UTM system (if known). Otherwise 
        # enter all NaN's and mean position will be used as a reference.
        self.truePositionE      = float('nan')
        self.truePositionN      = float('nan')
        self.truePositionU      = float('nan')
        
        #%% Plot settings and constants =======================================
        # Enable/disable plotting of the tracking results for each channel
        self.plotAcquision      = 1              # 0 - Off; 1 - On
        # Enable/disable plotting of the tracking results for each channel
        self.plotTracking       = 1              # 0 - Off; 1 - On
        # Enable/disable plotting of the navigation solution
        self.plotNavigation     = 1              # 0 - Off; 1 - On
        
        #%% Constants =========================================================
        self.c                  = 299792458.   # The speed of light 
        self.startOffset        = 68.802          # [ms] Initial sign. travel time
        
        #%% CNo Settings ======================================================
        # Number of correlation values used to compute each C/No point
        self.CNoVSMinterval = 50                # [ms]
