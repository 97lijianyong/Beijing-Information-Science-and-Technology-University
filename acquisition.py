# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import caCode

#%% 
class Acquisition():
    def __init__(self, settings): 
        # satellite number to be acquired
        svNum = max(max(settings.acqSatelliteList),settings.numberOfChannels)
        # carrier frequencies of detected signals
        self.carrFreq = np.full(svNum, np.inf)
        # C/A code phases of detected signals 
        self.codePhase = np.full(svNum, np.inf)
        # Correlation peak ratios of the detected signals
        self.peakMetric = np.zeros(svNum)
        # PRN
        self.PRN = np.zeros(svNum).astype(int)
        # code frequency basis of NCO
        self.codeFreq = np.zeros(svNum)
        # Acquisition flag
        self.acqFlag = np.full(svNum, False)
        # Initialize the multiplier to adjust for the data type
        if settings.fileType == 1:
            self.dataAdaptCoeff = 1
        else:
            self.dataAdaptCoeff = 2
    
    #%% Acquisition engine       
    def acqEngine(self,settings):
        '''
        Function performs cold start acquisition on the collected "data". It
        searches for GPS signals of all satellites, which are listed in field
        "acqSatelliteList" in the settings structure. Function saves code phase
        and frequency of the detected signals in the "acqResults" structure.
        
        Inputs:
            settings - receiver settings. Type of data file, sampling frequency
            and the default filename are specified here.
        
        Ouputs:
            acqResults - Function saves code phases and frequencies of the 
            detected signals in the "acqResults" structure. The field "carrFreq"
            is set to 0 if the signal is not detected for the given PRN number.
        '''

        # = == == == == == == == == == == == == == == == == == == == == =
        # Enable use of GPU acceleration for acquisition
        if settings.gpuACQflag == True:
            import cupyx.scipy.fft  
            import cupy as cop
            cofft = cupyx.scipy.fft
        elif settings.gpuACQflag == False:
            cofft = np.fft
            cop = np
        
        # Read data for acquisition ===========================================
        # Find number of samples per spreading code
        samplesPerCode = round(settings.samplingFreq / 
                (settings.codeFreqBasis / settings.codeLength))
        #At least 202ms of signal are needed for fine frequency estimation
        codeLen = max(202,settings.acqNonCohTime+2);
        
        with open(settings.fileName,"rb") as fid:
            # Move the starting point of processing. Can be used to start the 
            # signal processing at any point in the data record (e.g. good for 
            # long records or for signal processing in blocks).
            if settings.dataType == np.int16:
                fid.seek(self.dataAdaptCoeff * settings.skipSamples * 2)
            else:
                fid.seek(self.dataAdaptCoeff * settings.skipSamples)
            # Read data for acquisition.
            longSignal = np.fromfile(fid, settings.dataType, 
                                     self.dataAdaptCoeff*codeLen*samplesPerCode)
        # Covert complex data into normal representation.
        if self.dataAdaptCoeff == 2:
            longSignal = longSignal[::2] + longSignal[1::2] * 1j
        
        # Transfer the IF data into GPU memory
        if settings.gpuACQflag == True:
            longSignal = cop.asarray(longSignal)   
        
        # Initialization ======================================================
        # --- Parameters for acquisition----------------------------------------
        # Sampling period
        ts = 1 / settings.samplingFreq
        # Find phase points of 200ms local carrier wave
        phasePoints = cop.pi *2* ts * cop.arange(samplesPerCode*200).astype(cop.float32) 
    
        # Input signal power for GLRT statistic calculation -------------------
        if settings.gpuACQflag == True:
            sigPower = cop.std(longSignal[:samplesPerCode]).get() * np.sqrt(samplesPerCode)
        elif settings.gpuACQflag == False:
            sigPower = np.std(longSignal[:samplesPerCode])* np.sqrt(samplesPerCode)
        
        # Perform search for all listed PRN numbers ...
        print('   (',end="")

        # Coarse acquisition ==================================================
        for PRN in settings.acqSatelliteList:
            # Generate C/A codes and sample them according to the sampling freq.
            caCodeSample = caCode.codeSampling(settings,PRN,samplesPerCode)
            #  Add zero-padding samples
            caCodes2ms = cop.hstack((caCodeSample, np.zeros(samplesPerCode)))
            # Perform DFT of B1C data code
            caCodeFreqDom = cop.conjugate(cofft.fft(caCodes2ms))

            # --- Initialize acqResults-----------------------------------------------
            codePhaseM, freqM, peakMagM = 0, 0, 0
            # Make the correlation for all frequency bins ---------------------
            for freqBin in range(int(-settings.acqSearchBand),
                                 int(settings.acqSearchBand + settings.acqSearchStep),
                                 int(settings.acqSearchStep)):
                # Generate carrier wave frequency grid
                coarseFreqBin = freqBin + settings.IF
                # Generate local sine and cosine
                sigCarr = cop.exp(-1j*coarseFreqBin*phasePoints[:samplesPerCode*2])
    
                # Search results of one frequency bin and all code shifts
                results = cop.zeros(samplesPerCode*2)
                
                #--- Do non-coherent integration ------------------------------
                for intIndex in range(settings.acqNonCohTime):
                    # Take 2ms vectors of input data to do correlation
                    signal = longSignal[intIndex * samplesPerCode:
                                        (intIndex + 2) * samplesPerCode]
                    # "Remove carrier" from the signal and convert the baseband
                    # signal to frequency domain
                    IQfreqDom = cofft.fft(sigCarr * signal)
                    # Multiplication in the frequency domain
                    convCodeIQ = IQfreqDom * caCodeFreqDom
                    # Perform inverse DFT and non-coherent integration
                    results = results + cop.abs(cofft.ifft(convCodeIQ))

                #  Find the fine carrier freq  and code phase
                if settings.gpuACQflag == True:
                    maxIndex = cop.argmax(results).get()
                    maxPeak = results[maxIndex].get()
                elif settings.gpuACQflag == False:
                    maxIndex = np.argmax(results)
                    maxPeak = results[maxIndex]
                    
                if  maxPeak > peakMagM:
                    peakMagM = maxPeak
                    codePhaseM = maxIndex
                    freqM = coarseFreqBin
            
            # Save code phase acquisition result
            self.codePhase[PRN-1] = codePhaseM
            # Just copy nominal freq for low rate C/A code 
            self.codeFreq[PRN-1] = settings.codeFreqBasis
            # Store GLRT statistic
            self.peakMetric[PRN-1] = peakMagM/sigPower/settings.acqNonCohTime
            # Store PRN
            self.PRN[PRN-1] = PRN
            
            # Fine resolution frequency search ================================
            if self.peakMetric[PRN-1] > settings.acqThreshold :
                self.acqFlag[PRN-1] = True
                # Indicate PRN number of the detected signal
                print(PRN,' ',end="")
                #--- Prepare 200ms code, carrier and input signals ------------
                # C/A code samples
                caCode200ms = caCode.codeSampling(settings,PRN,samplesPerCode*200)
                # Transfer the code samples into GPU memory
                caCode200ms = cop.asarray(caCode200ms)
                # Local carrier signal
                localCarr200cm = cop.exp(1j * freqM * phasePoints)
                # Take 200cm incoming signal for fine acquisition
                sig200cm = longSignal[codePhaseM:codePhaseM + 200*samplesPerCode]
                # --- Integration for each of the 200 codes -------------------
                # Wipe off code and carrier from incoming signals
                basebandSig = sig200cm * caCode200ms * localCarr200cm
                # Integration for each code
                sumPerCode = basebandSig.reshape((-1,200),order = 'F').sum(axis=0)
                # --- Find the fine carrier freq. -----------------------------
                # Index of the max power
                if settings.gpuACQflag == True:
                    maxPowerIndex = cop.abs(cofft.fft(sumPerCode**2)).argmax().get()
                elif settings.gpuACQflag == False:
                    maxPowerIndex = np.abs(cofft.fft(sumPerCode**2)).argmax()
                # FFT shift angle
                shiftAngle = np.angle(np.exp( -2*np.pi*1j* maxPowerIndex/200 ))/2
                self.carrFreq[PRN-1] = freqM + shiftAngle/0.001/2/np.pi
            else:
                #--- No signal with this PRN ----------------------------------
                print('. ',end="")
        
        print(')')
    
    #%% 
    def chAlloc(self,settings):
        # Channel count
        chCnt = settings.numberOfChannels
        PRNindexes = self.peakMetric.argsort()[::-1][:chCnt]
        self.peakMetric = self.peakMetric.take(PRNindexes)
        self.carrFreq = self.carrFreq.take(PRNindexes)
        self.codePhase = self.codePhase.take(PRNindexes)
        self.PRN = self.PRN.take(PRNindexes)
        self.acqFlag =  self.acqFlag.take(PRNindexes)
        self.codeFreq =  self.codeFreq.take(PRNindexes)
    
    #%%     
    def printAcqResults(self,settings):
        pass
    
    #%%     
    def run(self,settings):
        self.acqEngine(settings) 
        # Plot acquisition results
        self.plotAcq(settings)
        # Allocate channels for acquired satellites
        self.chAlloc(settings)
        # Show the acquisition results
        self.printAcqResults(settings)

    def plotAcq(self, settings):
        acqPrn = np.array(np.where(self.carrFreq != np.inf)) + 1
        notAcqPrn = np.array(np.where(self.carrFreq == np.inf)) + 1
        acqPeak = self.peakMetric[acqPrn - 1]
        notacqPeak = self.peakMetric[notAcqPrn - 1]
        fig, ax = plt.subplots()
        ax.bar(acqPrn[0], acqPeak[0], color='b', width=0.85)
        ax.bar(notAcqPrn[0], notacqPeak[0], color='r', width=0.85)
        ax.legend(labels=['Acquired signals', 'Not acquired signals'])
        ax.set_xlabel('PRN number (no bar - SV is not in the acquisition list)')
        ax.set_ylabel('Acquisition Metric')
        ax.set_title('Acquisition results')
        plt.grid(True)
        maxPrn = max(acqPrn.max(), notAcqPrn.max())
        plt.xticks(np.arange(1, maxPrn + 1, 2))
        plt.tight_layout()
        plt.show()
        return

    #%%         


