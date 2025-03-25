# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal.windows import hann
import sys

def probeData(settings):
    try:
        print('Probing data ', settings.fileName,'...')
        probeDataPlot(settings)
    except Exception as e:
        # There was an error, print it and exit
        print(repr(e))
        print('  (settings in "initSettings.py" to reconfigure)')
        sys.exit()
    
    print('  Raw IF data plotted ')

def probeDataPlot(settings):
    """ 
    Function plots raw data information: time domain plot, a frequency domain
    plot and a histogram.
    Inputs:
           settings - receiver settings. Type of data file, sampling frequency 
           and the default filename are specified here.
    """
    
    #%% Read samples from file ------------------------------------------------
    fid = open(settings.fileName,"rb")
    
    # Initialize the multiplier to adjust for the data type
    if settings.fileType == 1:
        dataAdaptCoeff = 1
    else:
        dataAdaptCoeff = 1
        
    # Move the starting point of processing. Can be used to start the
    # signal processing at any point in the data record (e.g. good for long
    # records or for signal processing in blocks).
    if settings.dataType == np.int16:
        fid.seek(dataAdaptCoeff * settings.skipSamples * 2)
    else:
        fid.seek(dataAdaptCoeff * settings.skipSamples)
    
    # Find number of samples per spreading code
    samplesPerCode = round(settings.samplingFreq / 
                           (settings.codeFreqBasis / settings.codeLength))
    # Read 10ms of signal
    data = np.fromfile(fid, settings.dataType, dataAdaptCoeff*10*samplesPerCode)
    fid.close()
    #%% Generate plot of raw data ---------------------------------------------

    # Frequency domain plot ---------------------------------------------------
    win = hann(32768, True)
    freq, Pxxf = welch(data, fs=settings.samplingFreq/1e6, window = win,
                       noverlap=2048, nfft=32768, detrend=False)
    # plot
    ax1 = plt.subplot(211)
    ax1.semilogy(freq, Pxxf)
    ax1.grid()
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('PSD magnitude')
    ax1.set_title("Frequency domain plot")
    ax1.set_xlim(0,max(freq))
    
    # Time domain plot --------------------------------------------------------
    timeScale = np.arange(0,25e-3,1/settings.samplingFreq)*1000
    ax2 = plt.subplot(223)  
    ax2.plot(timeScale[1:1000], data[1:1000])
    ax2.grid()
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title("Time domain plot")
    ax2.set_xticks(np.linspace(min(timeScale[1:1000]),max(timeScale[1:1000]),4))

    # Histogram ---------------------------------------------------------------
    ax3 = plt.subplot(224)
    ax3.hist(data,density=True)
    ax3.grid()
    ax3.set_xlabel('Bin')
    ax3.set_ylabel('Number in bin')
    ax3.set_title("Histogram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%% For test purpose    
if __name__=='__main__':
    from initSettings import initSettings
    settings = initSettings()
    probeData(settings)
