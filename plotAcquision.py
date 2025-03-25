

import numpy as np
import matplotlib.pyplot as plt
import initSettings

settings = initSettings.initSettings()
def plotAcquision(self, settings):
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
