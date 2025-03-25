
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot, grid
from mpl_toolkits.mplot3d import Axes3D
from postNavigation import *
import initSettings

settings = initSettings.initSettings() 

def deg2dms(deg):  # 度数转度分秒，仅用于输出表示	
    a = deg - int(deg)  # 获取度数的小数部分	
    b = a * 3600  # 小数部分全部转为角秒表示	
    m = int(b / 60)  # 整数部分为分	
    s = b % 60  # 余数部分为秒	
    return int(deg), m, round(s, 2)

def plotNavigation(navSolutions,settings): 
    refCoord = type("refCoord",(),{})
    ## Plot results in the necessary data exists ==============================
    if (navSolutions ):
        navSolutions.el = navSolutions.el[~np.isnan(navSolutions.el)] 
        navSolutions.az = navSolutions.az[~np.isnan(navSolutions.az)] 
        
        ## If reference position is not provided, then set reference position
        ## to the average postion
        if np.isnan(settings.truePositionE) or np.isnan(settings.truePositionN) or np.isnan(settings.truePositionU):
            #=== Compute mean values ==========================================
            # Remove NaN-s or the output of the function MEAN will be NaN.
            refCoord.E = np.mean(navSolutions.E[~np.isnan(navSolutions.E) ])
            refCoord.N = np.mean(navSolutions.N[~np.isnan(navSolutions.N) ])
            refCoord.U = np.mean(navSolutions.U[~np.isnan(navSolutions.U) ])
            #Also convert geodetic coordinates to deg:min:sec vector format
            meanLongitude = deg2dms(np.mean(navSolutions.longitude[~np.isnan(navSolutions.longitude) ]))
            meanLatitude = deg2dms(np.mean(navSolutions.latitude[~np.isnan(navSolutions.latitude) ]))

            refPointLgText = 'Mean Position\newline  Lat: ' + \
                             str(meanLatitude[0]) + '°' + \
                             str(meanLatitude[1]) + '\'' + \
                             str(meanLatitude[2]) + '\'\'' + \
                             '\nLng: ' + \
                             str(meanLongitude[0]) + '°' + \
                             str(meanLongitude[1]) + '\'' + \
                             str(meanLongitude[2]) + '\'\'' + \
                             '\nHgt: ' + \
                             '{:+6.1f}'.format(np.mean(navSolutions.height[~np.isnan(navSolutions.height)]))

           # refPointLgText = 'Mean Position\ newline  Lat: ' ,num2str (meanLatitude(1)),'{\circ}',num2str(meanLatitude(2)),'{\prime}',num2str(meanLatitude(3)),'{\prime}{\prime}','\newline Lng: ',num2str(meanLongitude(1)),'{\circ}',num2str(meanLongitude(2)),'{\prime}',num2str(meanLongitude(3)),'{\prime}{\prime}','\newline Hgt: ',num2str(mean(navSolutions.height(not np.isnan(navSolutions.height) )),'%+6.1f')])
        #refPointLgText = 'Mean Position\ newline  Lat: ', num2str (meanLatitude(1))

        else:
            refPointLgText = 'Reference Position'
            refCoord.E = settings.truePositionE
            refCoord.N = settings.truePositionN
            refCoord.U = settings.truePositionU
        figureNumber = 300
        # The 300 is chosen for more convenient handling of the open
        # figure windows, when many figures are closed and reopened. Figures
        # drawn or opened by the user, will not be "overwritten" by this
        # function if the auto numbering is not used.
        #=== Select (or create) and clear the figure ==========================
        fig = plt.figure(figureNumber)
        plt.clf()
        plt.title('Navigation solutions')
       
        #%% Plot all figures =======================================================
        #--- Coordinate differences in UTM system -----------------------------
        subplot(4,2,(1,4))
        E, = plt.plot(navSolutions.E - refCoord.E)
        N, = plt.plot(navSolutions.N - refCoord.N)
        U, = plt.plot(navSolutions.U - refCoord.U)
        plt.title('Coordinates variations in UTM system')
        plt.legend([E,N,U],['E','N','U'],loc = "upper right")
        plt.xlabel('Measurement period: '+str(settings.navSolPeriod)+'ms')
        plt.ylabel('Variations (m)')
        grid()
        plt.axis('tight')
        #--- Position plot in UTM system --------------------------------------
        subplot(4,2,(5,7))
        c, = plt.plot(navSolutions.E - refCoord.E,\
                       navSolutions.N - refCoord.N,'+')
        #Plot the reference point
        s, = plt.plot(0,0,'r+',linewidth=1.5,ms=10)
        grid('minor')
        plt.legend([c,s],['Measurements',refPointLgText])
        plt.title('Positions in UTM system (3D plot)')
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        #ax.zlabel('Upping (m)')
        #--- Satellite sky plot -----------------------------------------------
        polarAxis = subplot(4,2,(6,8),projection="polar")
        polarAxis.set_theta_zero_location('N')
        polarAxis.set_theta_direction(-1)
        plt.yticks(90*np.cos((np.pi/180)*np.arange(0,91,15)),np.arange(0,91,15))
        syita = navSolutions.az*np.pi/180
        rou = 90*np.cos(navSolutions.el* np.pi/180)
        
        for prn,i,j in zip(navSolutions.PRN[0],syita,rou):
            plt.annotate(int(prn), xy=(i,j))
        plt.scatter(syita,rou)
        plt.title('Sky plot (mean PDOP: %0.2f)'%(np.mean(navSolutions.DOP[:,1])))
        plt.show()
    else:
        print('plotNavigation: No navigation data to plot.')
    

if __name__=="__main__":
    import pickle
    with open("./navSolutions.pickle","rb") as f:
        navSolutions,eph = pickle.load(f)
    
    plotNavigation(navSolutions,settings)
    