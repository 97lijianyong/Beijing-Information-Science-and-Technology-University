# -*- coding: utf-8 -*-
import time
import os, pickle
from initSettings import initSettings
from probeData import probeData
from acquisition import Acquisition
from SIMDtracking import SIMDtrackingRun
from tracking import trackingRun
from postNavigation import postNavigation
from plotTracking import plotTracking
from plotNavigation import plotNavigation
import warnings

warnings.filterwarnings("ignore")
os.system('')
print("\x1B[2J")

if __name__ == "__main__":
    # --- Initialize constants, settings --------------------------------------
    settings = initSettings()
    # %% This is the postprocessing ============================================
    # Generate plot of raw data and ask if ready to start processing ----------
    probeData(settings)
    # Flag for start processing
    gnssStart = eval(input('Enter "1" to initiate GNSS processing or "0" to exit: '))

    # --- --------------------------------------------------------------------
    # %% Starting processing ==================================================
    if gnssStart == 1:
        # ----- Do the acquisition ---------------------------------------------------
        if settings.skipAcquisition==False:

            print('\nAcquiring satellites............')
            # ====== Acq processing ======
            acqResults = Acquisition(settings)
            acqResults.run(settings)

            with open("acqResults.pkl", "wb") as fp:
                pickle.dump(acqResults, fp)
        else:
            print("Skip acquisition use pickle data")
            with open(settings.acqPklPath, "rb") as fp:
                acqResults = pickle.load(fp)

        # --- --------------------------------------------------------------------
        # --- Track the signal ---------------------------------------------------
        if settings.skipTracking==False:

            startTime = time.time()
            startTimeStr = time.strftime("%m-%d-%Y, %H:%M:%S", time.localtime())
            print('Tracking started at ', startTimeStr)
            if settings.SIMDTrackflag   == True:
              trackResults = SIMDtrackingRun(settings, acqResults)
            else:
              trackResults = trackingRun(settings, acqResults)
            with open("trackResults.pkl", "wb") as fp:
                pickle.dump(trackResults, fp)
            # Elapsed time show
            elapsedT = time.time() - startTime
            elapsedStr = time.strftime("%H:%M:%S", time.gmtime(elapsedT))
            print("   Elapsed time for tracking is ", elapsedStr)
        else:
            print("Skip tracking use pickle data")
            with open(settings.trkPklPath, "rb") as fp:
                trackResults = pickle.load(fp)

        # --- Plot tracking ----------------------------------------------------
        if settings.plotTracking:
            plotTracking(trackResults, settings)

        # --- --------------------------------------------------------------------
        # ------------Calculate navigation solutions -----------------------------
        if settings.skipNavigation == False:

            navSolutions, eph = postNavigation(trackResults, settings)
            with open("navSolutions_eph.pkl", "wb") as fp:
                pickle.dump((navSolutions, eph), fp)
        else:
            with open("navSolutions_eph.pkl", "rb") as fp:
                (navSolutions, eph) = pickle.load(fp)

        # --- Plot Navigation ----------------------------------------------------
        if settings.plotNavigation:
            plotNavigation(navSolutions, settings)


