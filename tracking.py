# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

import caCode
from correlator import cpuCorrelatorReal

#%% 
class Tracking():
    def __init__(self,settings,chNum, acqResults):
        # Iteration count of tracking loop
        loopCnt = settings.msToProcess
        # PRN 
        self.PRN = acqResults.PRN[chNum]
        # Channel status
        self.lockFlag = acqResults.acqFlag[chNum]
        # The absolute sample in the record of the C/A code start
        self.absoluteSample = np.zeros(loopCnt)
        # Freq of the PRN code
        self.codeFreq = np.zeros(loopCnt)
        # Frequency of the tracked carrier wave
        self.carrFreq = np.zeros(loopCnt)
        # Outputs from the correlators (In-phase)
        self.I_P = np.zeros(loopCnt)
        self.I_E = np.zeros(loopCnt)
        self.I_L = np.zeros(loopCnt)
        # Outputs from the correlators (Quadrature-phase)
        self.Q_E = np.zeros(loopCnt)
        self.Q_P = np.zeros(loopCnt)
        self.Q_L = np.zeros(loopCnt)
        # Loop discriminators
        self.dllDiscr       = np.zeros(loopCnt)
        self.dllDiscrFilt   = np.zeros(loopCnt)
        self.pllDiscr       = np.zeros(loopCnt)
        self.pllDiscrFilt   = np.zeros(loopCnt)
        # Remain code and carrier phase
        self.remCodePhase   = np.zeros(loopCnt)
        self.remCarrPhase   = np.zeros(loopCnt)
        # C/No
        self.CNo = np.zeros(loopCnt//settings.CNoVSMinterval+1)
        
#%% 
def calcLoopCoef(settings, loopType, gain = 1.0):
    if loopType == 'pll':
        # Loop noise bandwid
        LBW = settings.pllNoiseBandwidth
        # Damping ratio
        zeta = settings.pllDampingRatio
    elif loopType == 'dll':
        # Loop noise bandwid
        LBW = settings.dllNoiseBandwidth
        # Damping ratio
        zeta = settings.dllDampingRatio    
    # Solve natural frequency
    Wn = LBW * 8 * zeta / (4 * zeta ** 2 + 1)
    # solve for t1 & t2 
    tau1 = gain / (Wn * Wn)
    tau2 = 2.0 * zeta / Wn
    return tau1, tau2

#%% 
def calcCNo(I,Q,T):
    # Calculate Power
    Z = I ** 2 + Q ** 2
    # Calculate the mean and variance of the Power 
    Zm = np.mean(Z)
    Zv = np.var(Z,ddof=1) 
    # Calculate the average carrier power
    Pav = np.sqrt(Zm ** 2 - Zv) 
    # Calculate the variance of the noise 
    Nv = 0.5 * (Zm - Pav)
    # Calculate C/No 
    CNo = 10 * np.log10(np.abs((1/T)*Pav/(2*Nv)))
    return CNo
   
#%%  
def trackingRun(settings, acqResults):
    # --- Initial variables to ba saved for all channels ------------------
    chCnt = min(sum(acqResults.acqFlag), settings.numberOfChannels)
    trkResults = {index:Tracking(settings, index, acqResults) 
                  for index in range(chCnt)}
    
    # --- Initialize common tracking variables ----------------------------
    # Define early-late offset (in chips)
    earlyLateSpc = settings.dllCorrelatorSpacing
    # Summation interval for code and carrier loops
    PDIcode = PDIcarr = settings.intTime
    # Calculate filter coefficient values
    tau1code, tau2code = calcLoopCoef(settings, 'dll', 1.0)
    tau1carr, tau2carr = calcLoopCoef(settings, 'pll',  0.25)
    
    # Iteration count of tracking loop
    loopCnt = settings.msToProcess

    # Open IF file to be processed
    fid = open(settings.fileName,"rb")
    
    for chNum in range(chCnt):
        # Current channel being processed
        chResults = trkResults[chNum]
        if acqResults.acqFlag[chNum] != True: continue
    
        # --- Perform various initializations ---------------------------------
        # Define initial code frequency basis of NCO
        codeFreq = acqResults.codeFreq[chNum]
        codeFreqBasis = acqResults.codeFreq[chNum]
        # Define residual code phase (in chips)
        remCodePhase  = 0.0
        # Define carrier frequency
        carrFreq = acqResults.carrFreq[chNum]
        carrFreqBasis = acqResults.carrFreq[chNum]
        # Define residual carrier phase
        remCarrPhase  = 0.0
        # code tracking loop parameters
        oldCodeNco   = 0.0        
        oldCodeError = 0.0
        # carrier/Costas loop parameters
        oldCarrNco   = 0.0
        oldCarrError = 0.0
        
        # C/No computation 
        vsmCnt  = 0; CNoValue = 0.
        
        # Move the starting point of processing. Can be used to start
        # the signal processing at any point in the data record (e.g. 
        # for long records). In addition skip through that data file 
        # to start at the appropriate sample (corresponding to code 
        # phase). Assumes sample type is schar (or 1 byte per sample)
        if settings.dataType == np.int16:
            fid.seek(acqResults.dataAdaptCoeff * 2 *
                  (settings.skipSamples + int(acqResults.codePhase[chNum])))
        else:
            fid.seek(acqResults.dataAdaptCoeff * 
                  (settings.skipSamples + int(acqResults.codePhase[chNum])))
        
        # Get a vector with the C/A code sampled 1x/chip
        codeChips = caCode.generateCAcode(settings,chResults.PRN)
        # Then make it possible to do early and late versions
        codeChips = np.concatenate([[codeChips[-1]], codeChips, [codeChips[0]]])

        
        # Find number of samples per spreading code
        # samplesPerCode = int(round(settings.samplingFreq * settings.intTime))
        # # Time index for each sampling point; 
        # sampleInd = np.arange(samplesPerCode + 10)

        print('   Tracking: Ch %i' % (chNum + 1), 'of %i' % chCnt, ', PRN: %i' % chResults.PRN)
        pbar = tqdm(total=loopCnt)
        updateCnt = 500
        for loopInd in range(loopCnt):
            if loopInd % updateCnt == 0:
                pbar.set_description('   C/No: %i (dB-Hz)' % CNoValue)
                pbar.update(updateCnt)

            # --- Read next block of data -----------------------------
            # Record sample number (based on samples)
            if settings.dataType == np.int16:
                chResults.absoluteSample[loopInd] = fid.tell()/acqResults.dataAdaptCoeff/2
            else:
                chResults.absoluteSample[loopInd] = fid.tell()/acqResults.dataAdaptCoeff

            # Update the phasestep based on code freq (variable) and
            # sampling frequency (fixed)
            codePhaseStep = codeFreq / settings.samplingFreq 
            #  Find the size of a "block" or code period in whole samples
            blksize = np.ceil((settings.codeLength-remCodePhase) / codePhaseStep).astype(int)
            # Read in the appropriate number of samples to process this
            # interation
            rawSignal = np.fromfile(fid, settings.dataType,
                                    acqResults.dataAdaptCoeff*blksize)
            
            # For complex data 
            if acqResults.dataAdaptCoeff == 2:
                rawSignal = rawSignal[::2] + rawSignal[1::2]
                
            # If did not read in enough samples, then could be out of
            # data - better exit
            if rawSignal.size != blksize:
                print('''Not able to read the specified number of samples
                      for tracking, exiting!''')
                fid.close()
                return
            
            # --- Correlator implementation -----------------------------------
            carrPhaseStep = carrFreq * 2 * np.pi /settings.samplingFreq
            
            I_E, Q_E, I_P, Q_P, I_L, Q_L = cpuCorrelatorReal(rawSignal, codeChips, 
                                  remCodePhase,codePhaseStep, earlyLateSpc, 
                                  remCarrPhase, carrPhaseStep)
            
            # --- Save and update variables for current correlation -----------
            # Save remCodePhase for current correlation
            chResults.remCodePhase[loopInd] = remCodePhase
            # Save remCarrPhase for current correlation
            chResults.remCarrPhase[loopInd] = remCarrPhase
            # Remaining code phase for next tracking update 
            remCodePhase = blksize * codePhaseStep + remCodePhase - settings.codeLength
            # Remaining carrier phase for next tracking update
            remCarrPhase = (carrPhaseStep * blksize + remCarrPhase)%(2 * np.pi)
            
            # --- Find PLL error and update carrier NCO -----------------------
            # Implement carrier loop discriminator (phase detector)
            carrError = np.arctan(Q_P / I_P) / (2.0 * np.pi)
            # Implement carrier loop filter and generate NCO command
            carrNco = oldCarrNco + (tau2carr/tau1carr) * \
                (carrError - oldCarrError) + carrError * (PDIcarr/tau1carr)
            oldCarrNco   = carrNco
            oldCarrError = carrError
            
            # Save carrier frequency for current correlation
            chResults.carrFreq[loopInd] = carrFreq
            # Modify carrier freq based on NCO command
            carrFreq = carrFreqBasis + carrNco
            
            # --- Find DLL error and update code NCO ------------------
            E = np.sqrt(I_E ** 2 + Q_E ** 2) 
            L = np.sqrt(I_L ** 2 + Q_L ** 2)
            codeError = (E-L) / (E+L) 
            # Implement code loop filter and generate NCO command 
            codeNco = oldCodeNco + (tau2code/tau1code) *   \
                (codeError - oldCodeError) + codeError * (PDIcode/tau1code)
            oldCodeNco   = codeNco
            oldCodeError = codeError
            
            # Save code frequency for current correlation 
            chResults.codeFreq[loopInd] = codeFreq
            # Modify code freq based on NCO command
            codeFreq = codeFreqBasis - codeNco
            
            # --- Record various measures to show ---------------------
            chResults.dllDiscr[loopInd] = codeError
            chResults.dllDiscrFilt[loopInd] = codeNco
            chResults.pllDiscr[loopInd] = carrError
            chResults.pllDiscrFilt[loopInd] = carrNco
            
            chResults.I_E[loopInd] = I_E
            chResults.I_P[loopInd] = I_P
            chResults.I_L[loopInd] = I_L
            chResults.Q_E[loopInd] = Q_E
            chResults.Q_P[loopInd] = Q_P
            chResults.Q_L[loopInd] = Q_L
            
            # --- CNo calculation -------------------------------------
            if (loopInd+1)% settings.CNoVSMinterval == 0:
                tempI = chResults.I_P[loopInd-settings.CNoVSMinterval+1:loopInd+1]
                tempQ = chResults.Q_P[loopInd-settings.CNoVSMinterval+1:loopInd+1]
                CNoValue = calcCNo(tempI,tempQ,settings.intTime)
                # Save the calculated CNo values
                chResults.CNo[vsmCnt] = CNoValue
                vsmCnt +=1
            
        # If we got so far, this means that the tracking was successful
        # Now we only copy status, but it can be update by a lock detector
        # if implemented
        chResults.lockFlag = True
        
        pbar.close()

    # Tracking results for all channles
    return trkResults

#%% 
if __name__ == "__main__":
    import time
    
    for PRN in range(5):
        print(' PRN')
        pbar = tqdm(total=1000)
        for ind in range(1000):
            time.sleep(0.01)
            if ind%2 == 0:
                pbar.set_description('CNo is %.2f' %ind)
                pbar.update(2)
        pbar.close()


                
  