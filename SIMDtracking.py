
import numpy as np
from tqdm import tqdm
import ctypes
import caCode

# %%
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
        # Outputs from the correlator (In-phase)
        self.I_P = np.zeros(loopCnt)
        self.I_E = np.zeros(loopCnt)
        self.I_L = np.zeros(loopCnt)
        # Outputs from the correlator (Quadrature-phase)
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
def SIMDtrackingRun(settings, acqResults):

    # --- Initial variables to ba saved for all channels ------------------
    chCnt = min(sum(acqResults.acqFlag), settings.numberOfChannels)
    trkResults = {index:Tracking(settings, index, acqResults)
                  for index in range(chCnt)}
    # --- Initialize common tracking variables ----------------------------
    # Summation interval for code and carrier loops
    PDIcode = PDIcarr = settings.intTime
    # Calculate filter coefficient values
    tau1code, tau2code = calcLoopCoef(settings, 'dll', 1.0)
    tau1carr, tau2carr = calcLoopCoef(settings, 'pll',  0.25)
    # Iteration count of tracking loop
    loopCnt = settings.msToProcess

    # Initialize the parameters to be passed to the C dynamic link library.
    # Calculate block blksize: Number of samples per code chip
    blksize = round(settings.samplingFreq /
                     (settings.codeFreqBasis / settings.codeLength))
    # Convert blksize to ctypes.c_int for compatibility with the C function in the dynamic link library
    blksize_ctypes = ctypes.c_int(blksize)
    # Initialize the number of code chips processed per millisecond.: Add 2 to handle boundary conditions
    rawsignal = np.zeros(blksize+2 )
    # Convert the raw signal data to int16 format to match the expected input type of the C function
    rawSignal_pt = rawsignal.astype(np.int16)
    # Convert the NumPy array to a C-compatible pointer (ctypes.POINTER(ctypes.c_short))
    # This allows the data to be passed directly to the C dynamic link library function
    rawSignal_ptr = rawSignal_pt.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
    #  codelength (codelen)
    codelen = settings.codeLength
    # Convert codelen to ctypes.c_int for compatibility with the C function in the dynamic link library
    codelen_ctypes = ctypes.c_int(codelen)

    # Load the dynamic link library (DLL) from the specified path
    dll = ctypes.CDLL("E:\program data\Dll11 final.dll")
    # Call the allocMemory function in the DLL to allocate memory based on the block size (blksize_ctypes)
    dll.allocMemory(blksize_ctypes)
    # Set the return type of the cFunc_cpuCorrelatorReal function in the DLL
    # The function returns a pointer to an array of 6 double values
    dll.cFunc_cpuCorrelatorReal.restype = ctypes.POINTER(ctypes.c_double * 6)

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

        # Open IF file to be processed
        fid = open(settings.fileName, "rb")
        # Initialize the multiplier to adjust for the data type
        if settings.dataType == np.int16:
            fid.seek(acqResults.dataAdaptCoeff * 2 *
                  (settings.skipSamples + int(acqResults.codePhase[chNum])))
        else:
            fid.seek(acqResults.dataAdaptCoeff *
                  (settings.skipSamples + int(acqResults.codePhase[chNum])))
        # Get a vector with the C/A code sampled 1x/chip
        codeChips = caCode.generateCAcode(settings,chResults.PRN)
        # Then make it possible to do early and late version
        codeChips = np.concatenate([[codeChips[-1]], codeChips, [codeChips[0]]])
        # Convert the codeChips NumPy array to a C-compatible pointer (ctypes.POINTER(ctypes.c_short))
        codeChips_ptr = codeChips.ctypes.data_as(ctypes.POINTER(ctypes.c_short))

        print('   Tracking: Ch %i' % (chNum + 1), 'of %i' % chCnt, ', PRN: %i' % chResults.PRN)
        pbar = tqdm(total=loopCnt)
        updateCnt = 200  # updateCnt是一个计数器，用于指定每隔多少次迭代更新一次进度条。
        # 示例循环


        # 关闭进度条
        # pbar.close()
        # print('   Tracking: Ch %i'%(chNum+1),'of %i'%chCnt,', PRN: %i'%chResults.PRN)
        # # Initialize a progress bar with the total number of iterations (loopCnt)
        # pbar = tqdm(total=loopCnt)
        # # Set the update frequency of the progress bar (update every 200 iterations)
        # updateCnt = 200

        # --- Process the number of specified code periods --------------------
        for loopInd in range(loopCnt):
            # for i in range(1, num_iterations + 1):
            if loopInd%updateCnt == 0:
                pbar.set_description('   C/No: %i (dB-Hz)' %CNoValue)
                pbar.update(updateCnt)

            # --- Read next block of data -----------------------------
            # Record sample number (based on samples)
            if settings.dataType == np.int16:
                chResults.absoluteSample[loopInd] = fid.tell()/acqResults.dataAdaptCoeff/2
            else:
                chResults.absoluteSample[loopInd] = fid.tell()/acqResults.dataAdaptCoeff
            # Calculate the code phase step (codePhaseStep): the change in code phase per sample
            codePhaseStep = codeFreq / settings.samplingFreq
            #  Find the size of a "block" or code period in whole samples
            blksize  = np.ceil((settings.codeLength-remCodePhase) / codePhaseStep).astype(int)
            # Read in the appropriate number of samples to process this
            # interation
            rawSignal = np.fromfile(fid, settings.dataType,
                                    acqResults.dataAdaptCoeff * blksize)

            rawSignal_pt[:rawSignal.size] = rawSignal  # 将 rawSignal 的值赋值给 rawSignal_pt 的部分元素

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
            # Create a NumPy array of size 5 to store  parameters for passing to the C function
            arr = np.zeros(5)
            # Convert the NumPy array to a C-compatible pointer (ctypes.POINTER(ctypes.c_double))
            # This allows the array to be passed directly to the C dynamic link library function
            array_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            # Pack the parameters into the array:
            arr[0] = remCodePhase
            arr[1] = codePhaseStep
            arr[2] = settings.dllCorrelatorSpacing
            arr[3] = remCarrPhase
            arr[4] = carrPhaseStep
            # Call the C function from the dynamic link library, passing the array and other parameters
            # The function returns a pointer to the correlator output array
            correValues_ptr = dll.cFunc_cpuCorrelatorReal(rawSignal_ptr, codeChips_ptr, array_ptr, blksize_ctypes, codelen_ctypes)

            # Extract the correlator output values from the returned pointer
            I_E, Q_E, I_P, Q_P, I_L, Q_L = correValues_ptr.contents

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
    dll.cleanup()
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

