
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot,grid
import initSettings

settings = initSettings.initSettings()
def plotTracking(trackResults,settings): 
    
    # Protection - if the list contains incorrect channel numbers
    #channelList = intersect(channelList,np.arange(1,settings.numberOfChannels+1))
    #=== For all listed channels ==============================================
    for channelNr in trackResults:
        if True: 
            ## Select (or create) and clear the figure ========================
            # The number 200 is added just for more convenient handling of the open
            # figure windows, when many figures are closed and reopened.
            # Figures drawn or opened by the user, will not be "overwritten" by
            # this function.
            plt.figure(channelNr + 200)
            # plt.clf(channelNr + 200)
            #set(channelNr + 200,'Name',np.array(['Channel ',num2str(channelNr),' (PRN ',num2str(trackResults[channelNr].PRN),') results']))
            ## Draw axes ======================================================
            """
            # Row 1    
            handles11 = subplot(3,3,1)
            handles12 = subplot(3,3,(2,3))
            # Row 2
            handles21 = subplot(3,3,4)
            handles22 = subplot(3,3,(5,6))
            # Row 3
            handles31 = subplot(3,3,7)
            handles32 = subplot(3,3,8)
            handles33 = subplot(3,3,9)
            """
            ## Plot all figures ===============================================
            timeAxisInSeconds = (np.arange(1,settings.msToProcess+1)) / 1000
            #----- Discrete-Time Scatter Plot ---------------------------------
            subplot(3,3,1)
            plt.plot(trackResults[channelNr].I_P,trackResults[channelNr].Q_P,'.')
            grid()
            plt.axis('equal')
            plt.title('Discrete-Time Scatter Plot')
            plt.xlabel('I prompt')
            plt.ylabel('Q prompt')
            #----- Nav bits ---------------------------------------------------
            subplot(3,3,(2,3))
            plt.plot(timeAxisInSeconds,trackResults[channelNr].I_P)
            grid()
            plt.title('Bits of the navigation message')
            plt.xlabel('Time (s)')
            plt.axis('tight')
            #----- PLL discriminator unfiltered--------------------------------
            subplot(3,3,4)
            plt.plot(timeAxisInSeconds,trackResults[channelNr].pllDiscr,'r')
            #grid()
            plt.axis('tight')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Raw PLL discriminator')
            #----- Correlation ------------------------------------------------
            subplot(3,3,(5,6))
            plt.plot(timeAxisInSeconds,np.array([np.transpose(np.sqrt(trackResults[channelNr].I_E ** 2 + trackResults[channelNr].Q_E ** 2)),np.transpose(np.sqrt(trackResults[channelNr].I_P ** 2 + trackResults[channelNr].Q_P ** 2)),np.transpose(np.sqrt(trackResults[channelNr].I_L ** 2 + trackResults[channelNr].Q_L ** 2))]).T,'-*')
            #grid()
            plt.title('Correlation results')
            plt.xlabel('Time (s)')
            plt.axis('tight')
            # hLegend = plt.legend('$\sqrt{I_{E}^2 + Q_{E}^2}$','$\sqrt{I_{P}^2 + Q_{P}^2}$','$\sqrt{I_{L}^2 + Q_{L}^2}$')
            #set interpreter from tex to latex. This will draw \sqrt correctly
            # set(hLegend,'Interpreter','Latex')
            #----- PLL discriminator filtered----------------------------------
            subplot(3,3,7)
            plt.plot(timeAxisInSeconds,trackResults[channelNr].pllDiscrFilt,'b')
            #grid()
            plt.axis('tight')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Filtered PLL discriminator')
            #----- DLL discriminator unfiltered--------------------------------
            subplot(3,3,8)
            plt.plot(timeAxisInSeconds,trackResults[channelNr].dllDiscr,'r')
            #grid()
            plt.axis('tight')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Raw DLL discriminator')
            #----- DLL discriminator filtered----------------------------------
            subplot(3,3,9)
            plt.plot(timeAxisInSeconds,trackResults[channelNr].dllDiscrFilt,'b')
            #grid()
            plt.axis('tight')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Filtered DLL discriminator')
            #----- Plot CNo----------------------------------
            plt.figure(channelNr + 300)
            # plt.clf(channelNr + 300)
            #set(channelNr + 300,'Name',np.array(['Channel ',num2str(channelNr),' (PRN ',num2str(trackResults[channelNr].PRN),') CNo']))
            plt.plot(trackResults[channelNr].CNo[:-1])
            # hold('on')
            plt.scatter(np.arange(len(trackResults[channelNr].CNo)-1),trackResults[channelNr].CNo[:-1],alpha=0.3,edgecolors="m")
            # hold('off')
            plt.title('CNo Estimation (computed only every 400msec (or as specified in initSettings.m)')
            plt.ylabel('dB-Hz')
            plt.xlabel('400msec (or as set in initSettings.m) epoch computation')
    plt.show()
if __name__=="__main__":
    import pickle
    with open("./trackResults.pickle","rb") as f:
        trackResults = pickle.load(f)
    
    plotTracking(trackResults,settings)
    
