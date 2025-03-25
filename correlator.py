# -*- coding: utf-8 -*-

import numpy as np
import numba

@numba.jit(nopython=False)
def cpuCorrelatorReal(rawSignal, codeChips, remCodePhase,codePhaseStep,
                      earlyLateSpc, remCarrPhase, carrPhaseStep):
    blksize = rawSignal.size
    # --- Generate local code replic -------------------------- 
    # Time index for each sampling point; 
    sampleInd = np.arange(blksize)
    
    codeInd = sampleInd * codePhaseStep
    # Define index into early code vector
    tcode1 = np.ceil((remCodePhase - earlyLateSpc) + codeInd).astype(np.int32)
    earlyCode = codeChips.take(tcode1)
    
    # Define index into late code vector
    tcode2 = np.ceil((remCodePhase + earlyLateSpc) + codeInd).astype(np.int32)
    lateCode = codeChips.take(tcode2)
    
    # Define index into prompt code vector
    tcode = remCodePhase + codeInd
    promptCode = codeChips.take(np.ceil(tcode).astype(np.int32))
    
    # --- Generate local carrier ------------------------------
    # Get the argument to sin/cos functions
    trigarg = sampleInd * carrPhaseStep + remCarrPhase
    # Compute the signal to mix the collected data to bandband
    carrsigI = np.cos(trigarg)
    carrsigQ = np.sin(trigarg)
    
    # --- Do correlation --------------------------------------
    # First mix to baseband
    # basebandSignal = carrsig * rawSignal
    qBasebandSignal = carrsigI * rawSignal
    iBasebandSignal = carrsigQ * rawSignal
    # Now get early, late, and prompt values for each
    I_E = iBasebandSignal.dot(earlyCode)        
    Q_E = qBasebandSignal.dot(earlyCode)
    I_P = iBasebandSignal.dot(promptCode)
    Q_P = qBasebandSignal.dot(promptCode)
    I_L = iBasebandSignal.dot(lateCode)
    Q_L = qBasebandSignal.dot(lateCode)
    
    return I_E, Q_E, I_P, Q_P, I_L, Q_L
    
#%% 
if __name__ == "__main__":
    pass