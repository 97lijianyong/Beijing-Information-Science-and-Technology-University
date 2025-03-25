import numpy as np
from numpy.linalg import norm,matrix_rank,inv
from scipy.signal import correlate
import pickle
import initSettings
from math import atan2,atan,atanh

settings = initSettings.initSettings()
#%% helper class defination
class EPH:
    def __init__(self):
        self.idValid = np.zeros(5)
        self.PRN = []

        #subframe 1
        self.weekNumber = []
        self.accuracy = []
        self.health = []
        self.T_GD = []
        self.IODC = []
        self.t_oc = []
        self.a_f2 = []
        self.a_f1 = []
        self.a_f0 = []

        #subframe 2
        self.IODE_sf2 = []
        self.C_rs = []
        self.deltan = []
        self.M_0 = []
        self.C_uc = []
        self.e = []
        self.C_us = []
        self.sqrtA = []
        self.t_oe = []

        #subframe 3
        self.C_ic = []
        self.omega_0 = []
        self.C_is = []
        self.i_0 = []
        self.C_rc = []
        self.omega = []
        self.omegaDot = []
        self.IODE_sf3 = []
        self.iDot = []

        self.TOW = []

class navSolution:
    def __init__(self,activeChnList,currMeasNr):
        self.PRN = np.empty((currMeasNr,activeChnList))
        
        self.el = np.full((currMeasNr, settings.numberOfChannels),np.nan)
        self.az = np.full((currMeasNr, settings.numberOfChannels),np.nan)
        
        self.transmitTime = np.empty((currMeasNr, settings.numberOfChannels))
        self.satClkCorr = np.empty((currMeasNr, settings.numberOfChannels))
        
        self.rawP = np.empty((currMeasNr,settings.numberOfChannels))
        self.DOP = np.zeros((currMeasNr,5))
        
        self.X = np.full(currMeasNr,np.nan)
        self.Z = np.full(currMeasNr,np.nan)
        self.Y = np.full(currMeasNr,np.nan)
        self.dt = np.full(currMeasNr,np.nan)
        
        self.currMeasSample = np.empty((currMeasNr,activeChnList))
        self.localTime = np.empty(currMeasNr)
        
        self.correctedP = np.empty((currMeasNr, activeChnList))
        
        self.height = np.full(currMeasNr,np.nan)
        self.longitude = np.full(currMeasNr,np.nan)
        self.latitude = np.full(currMeasNr,np.nan)
        
        self.utmZone = None
        
        self.E = np.full(currMeasNr,np.nan)
        self.N = np.full(currMeasNr,np.nan)
        self.U = np.full(currMeasNr,np.nan)
        


#%% libs
def bin2dec(n):
    return int("".join(n),2)
def twosComp2dec(n):
    return bin2dec(n)-(2**len(n) if n[0]=='1' else 0)


def navPartyChk(ndat):
    if ndat[1] != 1:
        ndat[2:26]= -ndat[2:26]
    parity=np.empty((6,))
    parity[0] = np.prod(ndat[[ 0,  2,  3,  4,  6,  7, 11, 12, 13, 14, 15, 18, 19, 21, 24]])
    parity[1] = np.prod(ndat[[ 1,  3,  4,  5,  7,  8, 12, 13, 14, 15, 16, 19, 20, 22, 25]])
    parity[2] = np.prod(ndat[[ 0,  2,  4,  5,  6,  8,  9, 13, 14, 15, 16, 17, 20, 21, 23]])
    parity[3] = np.prod(ndat[[ 1,  3,  5,  6,  7,  9, 10, 14, 15, 16, 17, 18, 21, 22, 24]])
    parity[4] = np.prod(ndat[[ 1,  2,  4,  6,  7,  8, 10, 11, 15, 16, 17, 18, 19, 22, 23, 25]])
    parity[5] = np.prod(ndat[[ 0,  4,  6,  7,  9, 10, 11, 12, 14, 16, 20, 23, 24, 25]])

    
    return (-ndat[1] if ((np.sum(parity == ndat[26:32])) == 6) else 0)

def ephemeris(bits, D30Star,eph):
    
    def checkPhase(word):
        if D30Star == "1":
            wordp = list(bin((2**24-1)^int("".join(word[:24]),2))[2:])
            while len(wordp)<24:
                wordp = ["0"] + wordp
            word[:24] = wordp
        return word
    
    #% Check if there is enough data ==========================================
    if len(bits) < 1500:
        raise Exception('The parameter BITS must contain 1500 bits!');
    
    gpsPi = 3.1415926535898
    
    for i in range(5):
        subframe = bits[300*i:300*(i+1)]
        for j in range(10):
            subframe[j*30:(j+1)*30]=checkPhase(subframe[j*30:(j+1)*30])
            D30Star = subframe[30*(j+1)-1]
        
        subframeID = bin2dec(subframe[49:52])
        if subframeID == 1:
            eph.weekNumber  = bin2dec(subframe[60:70]) + 1024
            eph.accuracy    = bin2dec(subframe[72:76]);
            eph.health      = bin2dec(subframe[76:82]);
            eph.T_GD        = twosComp2dec(subframe[196:204]) * 2**(-31);
            eph.IODC        = bin2dec(subframe[82:84]+subframe[196:204]);
            eph.t_oc        = bin2dec(subframe[218:234]) * 2**4;
            eph.a_f2        = twosComp2dec(subframe[240:248]) * 2**(-55);
            eph.a_f1        = twosComp2dec(subframe[248:264]) * 2**(-43);
            eph.a_f0        = twosComp2dec(subframe[270:292]) * 2**(-31);      
            eph.idValid[0]  = 1;
        
        elif subframeID == 2:
            eph.IODE_sf2    = bin2dec(subframe[60:68]);
            eph.C_rs        = twosComp2dec(subframe[68: 84]) * 2**(-5);
            eph.deltan      = twosComp2dec(subframe[90:106]) * 2**(-43) * gpsPi;
            eph.M_0         = twosComp2dec(subframe[106:114]+subframe[120:144]) \
                * 2**(-31) * gpsPi;
            eph.C_uc        = twosComp2dec(subframe[150:166]) * 2**(-29);
            eph.e           = bin2dec(subframe[166:174]+subframe[180:204]) * 2**(-33);
            eph.C_us        = twosComp2dec(subframe[210:226]) * 2**(-29);
            eph.sqrtA       = bin2dec(subframe[226:234]+subframe[240:264]) * 2**(-19);
            eph.t_oe        = bin2dec(subframe[270:286]) * 2**4;
            eph.idValid[0]  = 2;
        elif subframeID == 3:
            eph.C_ic        = twosComp2dec(subframe[60:76]) * 2**(-29);
            eph.omega_0     = twosComp2dec(subframe[76:84]+subframe[90:114]) \
                * 2**(-31) * gpsPi;
            eph.C_is        = twosComp2dec(subframe[120:136]) * 2**(-29);
            eph.i_0         = twosComp2dec(subframe[136:144]+subframe[150:174]) \
                * 2**(-31) * gpsPi;
            eph.C_rc        = twosComp2dec(subframe[180:196]) * 2**(-5);
            eph.omega       = twosComp2dec(subframe[196:204]+subframe[210:234]) \
                * 2**(-31) * gpsPi;
            eph.omegaDot    = twosComp2dec(subframe[240:264]) * 2**(-43) * gpsPi;
            eph.IODE_sf3    = bin2dec(subframe[270:278]);
            eph.iDot        = twosComp2dec(subframe[278:292]) * 2**(-43) * gpsPi;
            eph.idValid[2]  = 3;
        elif subframeID == 4:
            # % Almanac, ionospheric model, UTC parameters.
            # % SV health (PRN: 25-32).
            # % Not decoded at the moment.
            pass
        elif subframeID == 5:
            # % SV almanac and health (PRN: 1-24).
            # % Almanac reference week number and time.
            # % Not decoded at the moment.
            pass
        
    TOW = bin2dec(subframe[30:47]) * 6 - 30
    eph.TOW = TOW;
    return eph,TOW

def NAVdecoding(I_P_Input):
    #%% --- Initialize ephemeris structure
    eph = EPH()
    
    subFrameStart = 0
    TOW = 0

    #%% Bit and frame synchronization
    searchStartOffset = 0
    preamble_bits = np.array([1, -1, -1, -1, 1, -1, 1, 1])
    preamble_ms = np.kron(preamble_bits, np.ones(20))
    bits = I_P_Input[searchStartOffset:].copy()
    bits[bits > 0] = 1
    bits[bits <= 0] = -1

    tlmXcorrResult = correlate(bits, np.append(preamble_ms,[0]*49840))
    xcorrLength = int((len(tlmXcorrResult) + 1) /2)-1

    t = tlmXcorrResult[xcorrLength:]
    index = np.where(np.abs(t) > 153)[0] +searchStartOffset
    
    subFrameStart = None
    for i in range(len(index)):
        index2 = index-index[i]
        
        if(len(np.where(index2==6000)[0])!=0 and index[i]>40):
            bits = I_P_Input[index[i]-40 : index[i] + 20 * 60]
            
            bits = bits.reshape((int(len(bits)/20),20)).T
            bits = np.sum(bits,axis=0)
    
            bits[bits > 0] = 1
            bits[bits <= 0] = -1
            if (navPartyChk(bits[:32]) != 0) and (navPartyChk(bits[30:62]) != 0):
                subFrameStart = index[i]
                break
    
    if subFrameStart ==None:
        print('Could not find valid preambles in channel! ');
        return eph,subFrameStart,TOW
    navBitsSamples = I_P_Input[subFrameStart - 20 : subFrameStart + (1500 * 20)]
    navBitsSamples = navBitsSamples.reshape((int(len(navBitsSamples) / 20),20)).T
    
    navBits=np.sum(navBitsSamples,axis=0)
    navBits = navBits>0
    
    navBitsBin = list(map(lambda x:str(int(x)),navBits))
    
    eph, TOW = ephemeris(navBitsBin[1:1501], navBitsBin[0],eph);
    return eph,int(subFrameStart),TOW

searchIndex = []
def calculatePseudoranges(trackResults,subFrameStart,TOW,currMeasSample, localTime,\
                          channelList, settings):
    
    # %% Transmitting Time of all channels at current measurement sample location
    transmitTime = np.full(settings.numberOfChannels,np.inf);
    
    # % This is used to accelerate the search process
    global searchIndex;
    if not( len(searchIndex) and localTime):
        searchIndex = np.ones(settings.numberOfChannels);
    
    
    # %--- For all channels in the list \
    for channelNr in channelList:
        
        # % Find index of I_P stream whose integration contains current
        # % measurment point location
        for index in range(int(searchIndex[channelNr]), len(trackResults[channelNr].absoluteSample)+1):
            if trackResults[channelNr].absoluteSample[index] > currMeasSample:
                break
        
        searchIndex[channelNr] = index;
        index = index - 1;
        
        # % Update the phasestep based on code freq and sampling frequency
        codePhaseStep = trackResults[channelNr].codeFreq[index] / settings.samplingFreq;
        
        # % Code phase from start of a PRN code to current measement sample location
        codePhase     = trackResults[channelNr].remCodePhase[index] +  \
            codePhaseStep * (currMeasSample - \
            trackResults[channelNr].absoluteSample[index] );
        
        # % Transmitting Time (in unite of s)at current measurement sample location
        # % codePhase/settings.codeLength: fraction part of a PRN code
        # % index - subFrameStart(channelNr): integer number of PRN code
        transmitTime[channelNr] =  (codePhase/settings.codeLength + index - \
            subFrameStart[channelNr]) * settings.codeLength/ \
            settings.codeFreqBasis + TOW[channelNr];
    
    
    # % At first time of fix, local time is initialized by transmitTime and
    # % settings.startOffset
    if not localTime:
        maxTime   = max(transmitTime[channelList]);
        localTime = maxTime + settings.startOffset/1000;
    
    
    # %--- Convert travel time to a distance ------------------------------------
    # % The speed of light must be converted from meters per second to meters
    # % per millisecond.
    pseudoranges    = (localTime - transmitTime) * settings.c;
    
    return pseudoranges,transmitTime,localTime

def satpos(transmitTime, eph):
    def check_t(time):
        half_week = 302400;     #% seconds
        corrTime = time;
        if time > half_week:
            corrTime = time - 2*half_week;
        elif time < -half_week:
            corrTime = time + 2*half_week;
        return corrTime
    rem = (lambda x,y : (x - int(x/y)*y) if y else 0)
    
    numOfSatellites = len(eph)
    
    gpsPi          = 3.1415926535898;  #% Pi used in the GPS coordinate 
                                        # % system
    
    # %--- Constants for satellite position calculation -------------------------
    Omegae_dot     = 7.2921151467e-5;  #% Earth rotation rate, [rad/s]
    GM             = 3.986005e14;      #% Earth's universal gravitational constant,
                                       #% [m^3/s^2]
    F              = -4.442807633e-10; #% Constant, [sec/(meter)^(1/2)]
    
    # %% Initialize results =====================================================
    satClkCorr   = np.zeros(numOfSatellites);
    satPositions = np.zeros((numOfSatellites,3));
    
    # %% Process each satellite =================================================
    
    for satNr,prn in enumerate(eph):
        
    # %% Find initial satellite clock correction --------------------------------
    
        # %--- Find time difference ---------------------------------------------
        dt = check_t(transmitTime[satNr] - eph[prn].t_oc);
    
        # %--- Calculate clock correction ---------------------------------------
        satClkCorr[satNr] = (eph[prn].a_f2 * dt + eph[prn].a_f1) * dt + \
                             eph[prn].a_f0 - \
                             eph[prn].T_GD;
    
        time = transmitTime[satNr] - satClkCorr[satNr];
    
    # %% Find satellite's position ----------------------------------------------
    
        # % Restore semi-major axis
        a   = eph[prn].sqrtA * eph[prn].sqrtA;
    
        # % Time correction
        tk  = check_t(time - eph[prn].t_oe);
    
        # % Initial mean motion
        n0  = np.sqrt(GM / a**3);
        # % Mean motion
        n   = n0 + eph[prn].deltan;
    
        # % Mean anomaly
        M   = eph[prn].M_0 + n * tk;
        # % Reduce mean anomaly to between 0 and 360 deg
        M   = rem(M + 2*gpsPi, 2*gpsPi);
    
        # %Initial guess of eccentric anomaly
        E   = M;
        
        # %--- Iteratively compute eccentric anomaly ----------------------------
        for ii in range(10):
            E_old   = E;
            E       = M + eph[prn].e * np.sin(E);
            dE      = rem(E - E_old, 2*gpsPi);
    
            if abs(dE) < 1.e-12:
                # % Necessary precision is reached, exit from the loop
                break;
        
        # % Reduce eccentric anomaly to between 0 and 360 deg
        E   = rem(E + 2*gpsPi, 2*gpsPi);
    
        # % Relativistic correction
        dtr = F * eph[prn].e * eph[prn].sqrtA * np.sin(E);
    
        # %Calculate the true anomaly
        nu   = atan2(np.sqrt(1 - eph[prn].e**2) * np.sin(E), np.cos(E)-eph[prn].e);
    
        # %Compute angle phi
        phi = nu + eph[prn].omega;
        # %Reduce phi to between 0 and 360 deg
        phi = rem(phi, 2*gpsPi);
    
        # %Correct argument of latitude
        u = phi + \
            eph[prn].C_uc * np.cos(2*phi) + \
            eph[prn].C_us * np.sin(2*phi);		
        # % Correct radius
        r = a * (1 - eph[prn].e*np.cos(E)) + \
            eph[prn].C_rc * np.cos(2*phi) + \
            eph[prn].C_rs * np.sin(2*phi);
        # % Correct inclination
        i = eph[prn].i_0 + eph[prn].iDot * tk + \
            eph[prn].C_ic * np.cos(2*phi) + \
            eph[prn].C_is * np.sin(2*phi);
        
        # % 2.9 SV position in orbital plane
        xk1 = np.cos(u)*r;
        yk1 = np.sin(u)*r;
    
        # %Compute the angle between the ascending node and the Greenwich meridian
        Omega = eph[prn].omega_0 + (eph[prn].omegaDot - Omegae_dot)*tk - \
                Omegae_dot * eph[prn].t_oe;
        # %Reduce to between 0 and 360 deg
        Omega = rem(Omega + 2*gpsPi, 2*gpsPi);
    
        # %--- Compute satellite coordinates ------------------------------------
        xk = xk1 * np.cos(Omega) - yk1 * np.cos(i)*np.sin(Omega);
        yk = xk1 * np.sin(Omega) + yk1 * np.cos(i)*np.cos(Omega);
        zk = yk1 * np.sin(i);
        satPositions[satNr] = [xk,yk,zk]
    
    # %% Include relativistic correction in clock correction --------------------
        satClkCorr[satNr] = (eph[prn].a_f2 * dt + eph[prn].a_f1) * dt + \
                             eph[prn].a_f0 - \
                             eph[prn].T_GD + dtr;      
    return satPositions,satClkCorr

def cart2utm(X, Y, Z, zone):
    def clsin(ar,degree,argument): 
        cos_arg = 2 * np.cos(argument)
        hr1 = 0
        hr = 0
        for t in np.arange(degree-1,-1,- 1):
            hr2 = hr1
            hr1 = hr
            hr = ar[t] + cos_arg * hr1 - hr2
        
        result = hr * np.sin(argument)
        return result.getA1()[0]
    
    def clksin(ar = None,degree = None,arg_real = None,arg_imag = None): 
        sin_arg_r = np.sin(arg_real)
        cos_arg_r = np.cos(arg_real)
        sinh_arg_i = np.sinh(arg_imag)
        cosh_arg_i = np.cosh(arg_imag)
        r = 2 * cos_arg_r * cosh_arg_i
        i = - 2 * sin_arg_r * sinh_arg_i
        hr1 = 0
        hr = 0
        hi1 = 0
        hi = 0
        for t in np.arange(degree-1,-1,- 1):
            hr2 = hr1
            hr1 = hr
            hi2 = hi1
            hi1 = hi
            z = ar[t] + r * hr1 - i * hi - hr2
            hi = i * hr1 + r * hi1 - hi2
            hr = z
        
        r = sin_arg_r * cosh_arg_i
        i = cos_arg_r * sinh_arg_i
        re = r * hr - i * hi
        im = r * hi + i * hr
        return re.getA1()[0],im.getA1()[0]
    
    a     = 6378388;
    f     = 1/297;
    ex2   = (2-f)*f / ((1-f)**2);
    c     = a * np.sqrt(1+ex2);
    vec   = np.array([[X, Y, Z-4.5]]).T;
    alpha = .756e-6;
    R     = np.mat(f"[1 {-alpha} 0;{alpha} 1 0;0 0 1]");
    trans = np.mat("[89.5; 93.8; 127.6]");
    scale = 0.9999988;
    v     = scale*R*vec + trans;	  #% coordinate vector in ED50
    L     = atan2(v[1], v[0]);
    N1    = 6395000;		          #% preliminary value
    B     = atan2(v[2]/((1-f)**2*N1), norm(v[:2])/N1); #% preliminary value
    U     = 0.1;  oldU = 0;
    
    while abs(U-oldU) > 1.e-4:
        oldU = U;
        N1   = c/np.sqrt(1+ex2*(np.cos(B))**2);
        B    = atan2(v[2]/((1-f)**2*N1+U), norm(v[:2])/(N1+U) );
        U    = norm(v[:2])/np.cos(B)-N1;

    
    # %Normalized meridian quadrant, KW p. 50 (96), p. 19 (38b), p. 5 (21)
    m0  = 0.0004;
    n   = f / (2-f);
    m   = n**2 * (1/4 + n*n/64);
    w   = (a*(-n-m0+m*(1-m0))) / (1+n);
    Q_n = a + w;
    
    # %Easting and longitude of central meridian
    E0      = 500000;
    L0      = (zone-30)*6 - 3;
    
    # %Check tolerance for reverse transformation
    tolutm  = np.pi/2 * 1.2e-10 * Q_n;
    tolgeo  = 0.000040;
    bg = np.mat("""[-3.37077907e-3;
       4.73444769e-6;
      -8.29914570e-9;
       1.58785330e-11]""")

    gb = np.mat("""[ 3.37077588e-3;
           6.62769080e-6;
           1.78718601e-8;
           5.49266312e-11]""")
    
    gtu = np.mat("""[ 8.41275991e-4;
            7.67306686e-7;
            1.21291230e-9;
            2.48508228e-12]""")
    
    utg = np.mat("""[-8.41276339e-4;
           -5.95619298e-8;
           -1.69485209e-10;
           -2.20473896e-13]""")
    
    # %Ellipsoidal latitude, longitude to spherical latitude, longitude
    neg_geo = 'FALSE';
    
    if B < 0:
        neg_geo = 'TRUE ';
    
    Bg_r    = abs(B);
    res_clensin = clsin(bg, 4, 2*Bg_r);
    Bg_r    = Bg_r + res_clensin;
    L0      = L0*np.pi / 180;
    Lg_r    = L - L0;
    
    # %Spherical latitude, longitude to complementary spherical latitude
    # %  i.e. spherical N, E
    cos_BN  = np.cos(Bg_r);
    Np      = atan2(np.sin(Bg_r), np.cos(Lg_r)*cos_BN);
    Ep      = atanh(np.sin(Lg_r) * cos_BN);
    
    # %Spherical normalized N, E to ellipsoidal N, E
    Np      = 2 * Np;
    Ep      = 2 * Ep;
    dN, dE = clksin(gtu, 4, Np, Ep);
    Np      = Np/2;
    Ep      = Ep/2;
    Np      = Np + dN;
    Ep      = Ep + dE;
    N       = Q_n * Np;
    E       = Q_n*Ep + E0;
    
    if neg_geo == 'TRUE ':
        N = -N + 20000000;
    return E, N, U

def leastSquarePos(satpos,obs):
    """
    Inputs:
    %       satpos      - Satellites positions (in ECEF system: [X; Y; Z;] -
    %                   one column per satellite)
    %       obs         - Observations - the pseudorange measurements to each
    %                   satellite corrected by SV clock error
    %                   (e.g. [20000000 21000000 .... .... .... .... ....]) 
    %       settings    - receiver settings
    %
    %   Outputs:
    %       pos         - receiver position and receiver clock error 
    %                   (in ECEF system: [X, Y, Z, dt]) 
    %       el          - Satellites elevation angles (degrees)
    %       az          - Satellites azimuth angles (degrees)
    %       dop         - Dilutions Of Precision ([GDOP PDOP HDOP VDOP TDOP])
    """
    def e_r_corr(traveltime, X_sat):
        Omegae_dot = 7.292115147e-5;          # %  rad/sec

        # %--- Find rotation angle --------------------------------------------------
        omegatau   = Omegae_dot * traveltime;
        
        # %--- Make a rotation matrix -----------------------------------------------
        R3 = np.mat([[np.cos(omegatau),np.sin(omegatau),0],
              [-np.sin(omegatau),np.cos(omegatau),0],
               [0,0,1]]);
        # %--- Do the rotation ------------------------------------------------------
        X_sat_rot = R3 * np.array([X_sat]).T;
        return X_sat_rot
    def topocent(X, dx):
        def togeod(a, finv, X, Y, Z):
            h = 0
            tolsq = 1e-10
            maxit = 10
            # compute radians-to-degree factor
            rtd = 180 / np.pi
            # compute square of eccentricity
            if finv < 1e-20:
                esq = 0
            else:
                esq = (2 - 1 / finv) / finv
            
            oneesq = 1 - esq
            # first guess
        # P is distance from spin axis
            P = np.sqrt(X ** 2 + Y ** 2)
            # direct calculation of longitude
            
            if P > 1e-20:
                dlambda = atan2(Y,X) * rtd
            else:
                dlambda = 0
            
            if (dlambda < 0):
                dlambda = dlambda + 360
            
            # r is distance from origin (0,0,0)
            r = np.sqrt(P ** 2 + Z ** 2)
            if r > 1e-20:
                sinphi = Z / r
            else:
                sinphi = 0
            
            dphi = np.arcsin(sinphi)
            # initial value of height  =  distance from origin minus
            # approximate distance from origin to surface of ellipsoid
            if r < 1e-20:
                h = 0
                return dphi,dlambda,h
            
            h = r - a * (1 - sinphi * sinphi / finv)
            # iterate
            for i in np.arange(1,maxit+1).reshape(-1):
                sinphi = np.sin(dphi)
                cosphi = np.cos(dphi)
                # compute radius of curvature in prime vertical direction
                N_phi = a / np.sqrt(1 - esq * sinphi * sinphi)
                # compute residuals in P and Z
                dP = P - (N_phi + h) * cosphi
                dZ = Z - (N_phi * oneesq + h) * sinphi
                # update height and latitude
                h = h + (sinphi * dZ + cosphi * dP)
                dphi = dphi + (cosphi * dZ - sinphi * dP) / (N_phi + h)
                # test for convergence
                if (dP * dP + dZ * dZ < tolsq):
                    break
                # Not Converged--Warn user
                if i == maxit:
                    print(" Problem in TOGEOD, did not converge in %2.0f \
                          iteration"%(i))
            
            dphi = dphi * rtd
            ######## end togeod.m  ######################
            return dphi,dlambda,h
        
        
        dtr = np.pi / 180
        phi,lambda_,h = togeod(6378137,298.257223563,*X)
        cl = np.cos(lambda_ * dtr)
        sl = np.sin(lambda_ * dtr)
        cb = np.cos(phi * dtr)
        sb = np.sin(phi * dtr)
        F = np.array([[- sl,- sb * cl,cb * cl],[cl,- sb * sl,cb * sl],[0,cb,sb]])
        local_vector = F.T * dx
        E = local_vector[0].getA()
        N = local_vector[1].getA()
        U = local_vector[2].getA()
        hor_dis = np.sqrt(E ** 2 + N ** 2)
        if hor_dis < 1e-20:
            Az = 0
            El = 90
        else:
            Az = atan2(E,N) / dtr
            El = atan2(U,hor_dis) / dtr
        
        if Az < 0:
            Az = Az + 360
        D = np.sqrt(dx[0] ** 2 + dx[1] ** 2 + dx[2] ** 2)
        return Az,El,D
    def tropo(sinel, hsta, p, tkel, hum, hp, htkel, hhum):
        a_e = 6378.137
        
        b0 = 7.839257e-05
        tlapse = - 6.5
        tkhum = tkel + tlapse * (hhum - htkel)
        atkel = 7.5 * (tkhum - 273.15) / (237.3 + tkhum - 273.15)
        e0 = 0.0611 * hum * 10 ** atkel
        tksea = tkel - tlapse * htkel
        em = - 978.77 / (2870400.0 * tlapse * 1e-05)
        tkelh = tksea + tlapse * hhum
        e0sea = e0 * (tksea / tkelh) ** (4 * em)
        tkelp = tksea + tlapse * hp
        psea = p * (tksea / tkelp) ** em
        if sinel < 0:
            sinel = 0
        
        tropo = 0
        done = 'FALSE'
        refsea = 7.7624e-05 / tksea
        htop = 1.1385e-05 / refsea
        refsea = refsea * psea
        ref = refsea * ((htop - hsta) / htop) ** 4
        while 1:

            rtop = (a_e + htop) ** 2 - (a_e + hsta) ** 2 * (1 - sinel ** 2)
            # check to see if geometry is crazy
            if rtop < 0:
                rtop = 0
            rtop = np.sqrt(rtop) - (a_e + hsta) * sinel
            a = - sinel / (htop - hsta)
            b = - b0 * (1 - sinel ** 2) / (htop - hsta)
            rn = np.zeros((8,1))
            for i in range(8):
                rn[i] = rtop ** (i + 2)
            alpha = np.array([2 * a,2 * a ** 2 + 4 * b / 3,a * (a ** 2 + 3 * b),a ** 4 / 5 + 2.4 * a ** 2 * b + 1.2 * b ** 2,2 * a * b * (a ** 2 + 3 * b) / 3,b ** 2 * (6 * a ** 2 + 4 * b) * 0.1428571,0,0])
            if b ** 2 > 1e-35:
                alpha[6] = a * b ** 3 / 2
                alpha[7] = b ** 4 / 9
            dr = rtop.copy()
            dr = dr + np.dot(alpha ,rn)[0]
            tropo = tropo + dr * ref * 1000
            if done == 'TRUE ':
                ddr = tropo.copy()
                break
            done = 'TRUE '
            refsea = (0.3719 / tksea - 1.292e-05) / tksea
            htop = 1.1385e-05 * (1255 / tksea + 0.05) / refsea
            ref = refsea * e0sea * ((htop - hsta) / htop) ** 4
        return ddr

    nmbOfIterations = 10
    
    dtr = np.pi / 180
    pos = np.zeros(4)
    X = satpos.copy()
    nmbOfSatellites = len(satpos)
    
    A = np.zeros((nmbOfSatellites,4),dtype=np.float64)
    omc = np.zeros(nmbOfSatellites)
    az = np.zeros(nmbOfSatellites)
    el = az.copy()
    #=== Iteratively find receiver position ===================================
    for _iter in range(nmbOfIterations):
        for i in range(nmbOfSatellites):
            if _iter == 0:
                #--- Initialize variables at the first iteration --------------
                Rot_X = X[i].copy()
                trop = 2
            else:
                #--- Update equations -----------------------------------------
                rho2 = (X[i,0] - pos[0]) ** 2 + (X[i,1] - pos[1]) ** 2 + \
                       (X[i,2] - pos[2]) ** 2
                traveltime = np.sqrt(rho2) / settings.c
                #--- Correct satellite position (do to earth rotation) --------
                # Convert SV position at signal transmitting time to position
                # at signal receiving time. ECEF always changes with time as
                # earth rotates.
                Rot_X = e_r_corr(traveltime,X.T[:,i].copy())
                #--- Find the elevation angel of the satellite ----------------
                az[i],el[i],__ = topocent(pos[np.arange(3)],\
                                              (Rot_X - pos[np.arange(3)]).diagonal().T)
                if (settings.useTropCorr == 1):
                    #--- Calculate tropospheric correction --------------------
                    trop = tropo(np.sin(el[i] * dtr),\
                                 0.0,1013.0,293.0,50.0,0.0,0.0,0.0)
                else:
                    # Do not calculate or apply the tropospheric corrections
                    trop = 0
            #--- Apply the corrections ----------------------------------------
            
            __rotx_pos = (Rot_X - pos[np.arange(3)])
            if __rotx_pos.ndim!=1:
                __rotx_pos = __rotx_pos.diagonal()
            
            omc[i] = (obs[i] -\
                          norm(__rotx_pos) - pos[3] - trop)
            
            #--- Construct the A matrix ---------------------------------------
            
            if type(Rot_X) == np.matrix:
                Rot_X = Rot_X.getA1()
            A[i] = np.array([\
                    (- (Rot_X[0] - pos[0])) / norm(__rotx_pos),\
                    (- (Rot_X[1] - pos[1])) / norm(__rotx_pos),\
                    (- (Rot_X[2] - pos[2])) / norm(__rotx_pos),\
                     1])
        # These lines allow the code to exit gracefully in case of any errors
        if matrix_rank(A) != 4:
            pos = np.zeros(4)
            dop = np.fill(5,np.inf)
            print('Cannot get a converged solotion!')
            return pos,el,az,dop
        #--- Find position update (in the least squares sense)-----------------
        x = np.linalg.lstsq(A,omc,rcond=None)[0]
        #--- Apply position update --------------------------------------------
        pos = pos + x
    
    #--- Fixing resulut -------------------------------------------------------
    pos = np.transpose(pos)
    #=== Calculate Dilution Of Precision ======================================
    if True:#nargout == 4:
        #--- Initialize output ------------------------------------------------
        dop = np.zeros(5)
        #--- Calculate DOP ----------------------------------------------------
        Q = inv(np.dot(np.transpose(A) , A))
        dop[0] = np.sqrt(np.trace(Q))
        dop[1] = np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])
        dop[2] = np.sqrt(Q[0,0] + Q[1,1])
        dop[3] = np.sqrt(Q[2,2])
        dop[4] = np.sqrt(Q[3,3])
    
    return pos,el,az,dop

def cart2geo(X, Y, Z, i):
    a = [6378388,6378160,6378135,6378137,6378137];
    f = [1/297,1/298.247, 1/298.26,1/298.257222101,1/298.257223563];
    
    lambda_ = atan2(Y,X);
    ex2 = (2-f[i])*f[i]/((1-f[i])**2);
    c = a[i]*np.sqrt(1+ex2);
    phi = atan(Z/((np.sqrt(X**2+Y**2)*(1-(2-f[i]))*f[i])));
    
    h = 0.1; oldh = 0;
    iterations = 0;
    while abs(h-oldh) > 1.e-12:
       oldh = h;
       N = c/np.sqrt(1+ex2*np.cos(phi)**2);
       phi = atan(Z/((np.sqrt(X**2+Y**2)*(1-(2-f[i])*f[i]*N/(N+h)))));
       h = np.sqrt(X**2+Y**2)/np.cos(phi)-N;
    
       iterations = iterations + 1;
       if iterations > 100:
           print('Failed to approximate h with desired precision. h-oldh: %e.\n'%(h-oldh));
           break
    phi =  phi*180/np.pi
    lambda_ = lambda_*180/np.pi;
    return phi, lambda_, h

def findUtmZone(latitude, longitude):
    if ((longitude > 180) or (longitude < - 180)):
        raise Exception('Longitude value exceeds limits (-180:180).')
    if ((latitude > 84) or (latitude < - 80)):
        raise Exception('Latitude value exceeds limits (-80:84).')
    ## Find zone ==============================================================
    
    # Start at 180 deg west = -180 deg
    
    utmZone = np.rint((180 + longitude) / 6) + 1
    ## Correct zone numbers for particular areas ==============================
    
    if (latitude > 72):
        # Corrections for zones 31 33 35 37
        if ((longitude >= 0) and (longitude < 9)):
            utmZone = 31
        elif ((longitude >= 9) and (longitude < 21)):
            utmZone = 33
        elif ((longitude >= 21) and (longitude < 33)):
            utmZone = 35
        elif ((longitude >= 33) and (longitude < 42)):
            utmZone = 37
    elif ((latitude >= 56) and (latitude < 64)):
        # Correction for zone 32
        if ((longitude >= 3) and (longitude < 12)):
            utmZone = 32
    return utmZone

def postNavigation(trkResult,settings):

    #%% Check is there enough data to obtain any navigation solution ===========
    if (settings.msToProcess < 36000):
        print('Record is to short. Exiting!')
        navSolutions = []
        eph = []
        return navSolutions,eph

    #%% Pre-allocate space ========
    subFrameStart = np.zeros(settings.numberOfChannels,np.int16)
    TOW = np.zeros(settings.numberOfChannels)
    eph = dict()
    
    activeChnList = list(trkResult)
    trkResult = list(trkResult.values())
    
    #%% Decode ephemerides ============================
    for channelNr in activeChnList:
        PRN = trkResult[channelNr].PRN
        print("Decoding NAV for PRN %02d -------------------- "%(PRN))
        eph[PRN],subFrameStart[channelNr],TOW[channelNr] = \
            NAVdecoding(trkResult[channelNr].I_P)
        
        if not (bool(eph[PRN].IODC) and bool(eph[PRN].IODE_sf2) and \
            bool(eph[PRN].IODE_sf3)):
            #%--- Exclude channel from the list (from further processing) ------
            activeChnList = list(set(activeChnList)-set([channelNr]))
            print('    Ephemeris decoding fails for PRN %02d !'%(PRN));
        else:
            print('    Three requisite messages for PRN %02d all decoded!'%(PRN));
    
    #%% Check if the number of satellites is still above 3 =====================
    if (len(activeChnList) < 4):
        #% Show error message and exit
        print('Too few satellites with ephemeris data for postion calculations. Exiting!');
        navSolutions = []
        eph          = []
        return navSolutions,eph

    #%% Set measurement-time point and step  =====================================
    # % Find start and end of measurement point locations in IF signal stream with available
    # % measurements
    sampleStart = np.zeros(settings.numberOfChannels);
    sampleEnd = np.full(settings.numberOfChannels,np.inf);
    
    for channelNr in activeChnList:
        __o = np.array(trkResult[channelNr].absoluteSample)
        
        sampleStart[channelNr] = \
              __o[subFrameStart[channelNr]];
        
        sampleEnd[channelNr] = trkResult[channelNr].absoluteSample[-1]
    
    
    # % Second term is to make space to aviod index exceeds matrix dimensions, 
    # % thus a margin of 1 is added.
    sampleStart = np.max(sampleStart) + 1;  
    sampleEnd = np.min(sampleEnd) - 1;
    
    # %--- Measurement step in unit of IF samples -------------------------------
    measSampleStep = int(settings.samplingFreq * settings.navSolPeriod/1000);
    
    # %---  Number of measurment point from measurment start to end ------------- 
    measNrSum = int((sampleEnd-sampleStart)/measSampleStep);
    
    
    #%% Initialization =========================================================
    # % Set the satellite elevations array to INF to include all satellites for
    # % the first calculation of receiver position. There is no reference point
    # % to find the elevation angle as there is no receiver position estimate at
    # % this point.
    satElev  = np.full(settings.numberOfChannels,np.inf);
    
    # % Save the active channel list. The list contains satellites that are
    # % tracked and have the required ephemeris data. In the next step the list
    # % will depend on each satellite's elevation angle, which will change over
    # % time.  
    readyChnList = activeChnList;
    
    # % Set local time to inf for first calculation of receiver position. After
    # % first fix, localTime will be updated by measurement sample step.
    localTime = None;
    
    # %##########################################################################
    # %#   Do the satellite and receiver position calculations                  #
    # %##########################################################################
    
    navSolutions = navSolution(len(readyChnList),measNrSum)
    
    print('Positions are being computed. Please wait\ ');
    for currMeasNr in range(measNrSum):
        # %% Initialization of current measurement ==============================          
        # % Exclude satellites, that are belove elevation mask 
        activeChnList = [i for i,v in enumerate(satElev) if v >= settings.elevationMask \
                         and i in readyChnList]
        
        # % Save list of satellites used for position calculation
        navSolutions.PRN[currMeasNr] = [trkResult[i].PRN for i in activeChnList]
        
        # % These two lines help the skyPlot function. The satellites excluded
        # % do to elevation mask will not "jump" to possition (0,0) in the sky
        # % plot.
        navSolutions.el[currMeasNr] = np.full(settings.numberOfChannels,np.nan);
        navSolutions.az[currMeasNr] = np.full(settings.numberOfChannels,np.nan);
        
        # % Signal transmitting time of each channel at measurement sample location
        navSolutions.transmitTime[currMeasNr] = \
                                             np.full(settings.numberOfChannels, np.NaN);
        navSolutions.satClkCorr[currMeasNr] = \
                                             np.full(settings.numberOfChannels, np.NaN);                                                                  
           
        # % Position index of current measurement time in IF signal stream
        # % (in unit IF signal sample point)
        currMeasSample = sampleStart + measSampleStep * currMeasNr;
        
        # %% Find pseudoranges ======================================================
        # % Raw pseudorange = (localTime - transmitTime) * light speed (in m)
        # % All output are 1 by settings.numberOfChannels columme vecters.
        navSolutions.rawP[currMeasNr],transmitTime,localTime=  \
                         calculatePseudoranges(trkResult,subFrameStart,TOW, \
                         currMeasSample,localTime,activeChnList, settings);
        
        # % Save transmitTime
        navSolutions.transmitTime[currMeasNr,:len(activeChnList)] = transmitTime[:len(activeChnList)]
        
        # %% Find satellites positions and clocks corrections =======================
        # % Outputs are all colume vectors corresponding to activeChnList
        satPositions, satClkCorr = satpos(transmitTime, eph); 
                                        
        # % Save satClkCorr
        navSolutions.satClkCorr[currMeasNr,:len(satClkCorr)] = satClkCorr
        
        # %% Find receiver position =================================================
        # % 3D receiver position can be found only if signals from more than 3
        # % satellites are available  
        if len(activeChnList) > 3:
    
            # %=== Calculate receiver position ==================================
            # % Correct pseudorange for SV clock error
            clkCorrRawP = navSolutions.rawP[currMeasNr,activeChnList] + \
                                                       satClkCorr * settings.c;
    
            # % Calculate receiver position
            xyzdt,navSolutions.el[currMeasNr,activeChnList], \
                   navSolutions.az[currMeasNr,activeChnList], \
                   navSolutions.DOP[currMeasNr] =\
                           leastSquarePos(satPositions, clkCorrRawP);
    
            # %=== Save results ===========================================================
            # % Receiver position in ECEF
            navSolutions.X[currMeasNr]  = xyzdt[0];
            navSolutions.Y[currMeasNr]  = xyzdt[1];
            navSolutions.Z[currMeasNr]  = xyzdt[2];       
            
    		# % For first calculation of solution, clock error will be set 
            # % to be zero
            if (currMeasNr == 0):
                navSolutions.dt[currMeasNr] = 0; # % in unit of (m)
            else:
                navSolutions.dt[currMeasNr] = xyzdt[3];  
            
    
    		# %=== Correct local time by clock error estimation =================
            localTime = localTime - xyzdt[3]/settings.c;       
            navSolutions.localTime[currMeasNr] = localTime;
            
            # % Save current measurement sample location 
            navSolutions.currMeasSample[currMeasNr] = currMeasSample;
    
            # % Update the satellites elevations vector
            satElev = navSolutions.el[currMeasNr]
    
            # %=== Correct pseudorange measurements for clocks errors ===========
            navSolutions.correctedP[currMeasNr] = \
                    navSolutions.rawP[currMeasNr,activeChnList] + \
                    satClkCorr.T * settings.c - xyzdt[3];
            # %% Coordinate conversion ==================================================
            
            # %=== Convert to geodetic coordinates ==============================
            navSolutions.latitude[currMeasNr], \
            navSolutions.longitude[currMeasNr], \
            navSolutions.height[currMeasNr] = cart2geo(\
                                                navSolutions.X[currMeasNr], \
                                                navSolutions.Y[currMeasNr], \
                                                navSolutions.Z[currMeasNr], \
                                                4);
        
            # %=== Convert to UTM coordinate system =============================
            navSolutions.utmZone = findUtmZone(navSolutions.latitude[currMeasNr], \
                                           navSolutions.longitude[currMeasNr]);
            
        
            # % Position in ENU
            navSolutions.E[currMeasNr], \
            navSolutions.N[currMeasNr], \
            navSolutions.U[currMeasNr] = cart2utm(xyzdt[0], xyzdt[1], \
                                                    xyzdt[2], \
                                                    navSolutions.utmZone);
        else:
            # %--- There are not enough satellites to find 3D position ----------
            print('   Measurement No. ', currMeasNr, \
                           ': Not enough information for position solution.');
    
            # %--- Set the missing solutions to NaN. These results will be
            # %excluded automatically in all plots. For DOP it is easier to use
            # %zeros. NaN values might need to be excluded from results in some
            # %of further processing to obtain correct results.
            navSolutions.el[currMeasNr] = np.full(settings.numberOfChannels,np.nan);
            navSolutions.az[currMeasNr] = np.full(settings.numberOfChannels,np.nan);   
            # % TODO: Know issue. Satellite positions are not updated if the
            # % satellites are excluded do to elevation mask. Therefore rasing
            # % satellites will be not included even if they will be above
            # % elevation mask at some point. This would be a good place to
            # % update positions of the excluded satellites.
        # %=== Update local time by measurement  step  ====================================
        localTime = localTime + measSampleStep/settings.samplingFreq ;
    
        
    return navSolutions, eph
        

if __name__=="__main__":
    import pickle
    
    with open("trackResults.pickle","rb") as f:
        trackResult = pickle.load(f)
   
    nav,eph = postNavigation(trackResult, settings)
    with open("navSolutions.pickle","wb") as f:
        pickle.dump((nav,eph),f)