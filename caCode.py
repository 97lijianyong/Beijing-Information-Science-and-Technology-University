# -*- coding: utf-8 -*-

#%% Import modules 
import numpy as np

#%% For GPS L1 C/A codes ------------------------------------------------------ 
def generateCAcode(settings,PRN):
    # --- Make the code shift array. The shift depends on the PRN number ------
    # The g2s vector holds the appropriate shift of the g2 code to generate
    # the C/A code (ex. for SV#19 - use a G2 shift of g2s(19) = 471)
    g2s = {
        1:    5,    2:    6,    3:    7,    4:    8,
        5:   17,    6:   18,    7:  139,    8:  140,
        9:  141,   10:  251,   11:  252,   12:  254,
        13:  255,   14:  256,   15:  257,   16:  258,
        17:  469,   18:  470,   19:  471,   20:  472,
        21:  473,   22:  474,   23:  509,   24:  512,
        25:  513,   26:  514,   27:  515,   28:  516,
        29:  859,   30:  860,   31:  861,   32:  862,
        33:  863,   34:  950,   35:  947,   36:  948,
        37:  950,
        38:   67,   39:  103,   40:   91,   41:   19,
        42:  679,   43:  225,   44:  625,   45:  946,
        46:  638,   47:  161,   48: 1001,   49:  554,
        50:  280,   51:  710,   52:  709,   53:  775,
        54:  864,   55:  558,   56:  220,   57:  397,
        58:   55,   59:  898,   60:  759,   61:  367,
        62:  299,   63: 1018,
        64:  729,   65:  695,   66:  780,   67:  801,
        68:  788,   69:  732,   70:   34,   71:  320,
        72:  327,   73:  389,   74:  407,   75:  525,
        76:  405,   77:  221,   78:  761,   79:  260,
        80:  326,   81:  955,   82:  653,   83:  699,
        84:  422,   85:  188,   86:  438,   87:  959,
        88:  539,   89:  879,   90:  677,   91:  586,
        92:  153,   93:  792,   94:  814,   95:  446,
        96:  264,   97: 1015,   98:  278,   99:  536,
        100:  819,  101:  156,  102:  957,  103:  159,
        104:  712,  105:  885,  106:  461,  107:  248,
        108:  713,  109:  126,  110:  807,  111:  279,
        112:  122,  113:  197,  114:  693,  115:  632,
        116:  771,  117:  467,  118:  647,  119:  203,
        120:  145,  121:  175,  122:   52,  123:   21,
        124:  237,  125:  235,  126:  886,  127:  657,
        128:  634,  129:  762,  130:  355,  131: 1012,
        132:  176,  133:  603,  134:  130,  135:  359,
        136:  595,  137:   68,  138:  386,  139:  797,
        140:  456,  141:  499,  142:  883,  143:  307,
        144:  127,  145:  211,  146:  121,  147:  118,
        148:  163,  149:  628,  150:  853,  151:  484,
        152:  289,  153:  811,  154:  202,  155: 1021,
        156:  463,  157:  568,  158:  904,  159:  670,
        160:  230,  161:  911,  162:  684,  163:  309,
        164:  644,  165:  932,  166:   12,  167:  314,
        168:  891,  169:  212,  170:  185,  171:  675,
        172:  503,  173:  150,  174:  395,  175:  345,
        176:  846,  177:  798,  178:  992,  179:  357,
        180:  995,  181:  877,  182:  112,  183:  144,
        184:  476,  185:  193,  186:  109,  187:  445,
        188:  291,  189:   87,  190:  399,  191:  292,
        192:  901,  193:  339,  194:  208,  195:  711,
        196:  189,  197:  263,  198:  537,  199:  663,
        200:  942,  201:  173,  202:  900,  203:   30,
        204:  500,  205:  935,  206:  556,  207:  373,
        208:   85,  209:  652,  210:  310,
        }
    
    assert 1 <= PRN <= 210, 'Invalid PRN code:'+str(PRN)
    
    # --- Pick right shift for the given PRN number ---------------------------
    g2shift = g2s[PRN]
    
    # --- Generate G1 code ----------------------------------------------------
    # Initialize g1 output to speed up the function
    g1 = np.zeros(settings.codeLength)
    # Load shift register
    reg = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    
    # Generate all G1 signal chips based on the G1 feedback polynomial
    for ind in range(settings.codeLength):
        g1[ind] = reg[9]
        reg = [reg[9] * reg[2]] + reg[0:9]
    
    # --- Generate G2 code ----------------------------------------------------
    # Initialize g2 output to speed up the function
    g2 = np.zeros(settings.codeLength)
    # Load shift register
    reg = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    # Generate all G2 signal chips based on the G2 feedback polynomial
    g2 = np.zeros(settings.codeLength)
    for ind in range(settings.codeLength):
      g2[ind] = reg[9]
      reg = [reg[9] * reg[8]*reg[7]*reg[5]*reg[2]*reg[1]] + reg[0:9]

    # --- Shift G2 code -------------------------------------------------------
    # The idea: g2 = concatenate[ g2_right_part, g2_left_part ]
    g2 = np.concatenate([g2[settings.codeLength-g2shift:],
                         g2[:settings.codeLength-g2shift]])
    
    # --- Form single sample C/A code by multiplying G1 and G2 ----------------
    return -(g1 * g2).astype(np.int16)
    #

    # \
    #     .astype(np.int16)       # convert to int16


#%%
def codeSampling(settings,PRN,sampleLen):
    # Generate CA code for given PRN ------------------------------------------
    caCode= generateCAcode(settings,PRN)
       
    # Find time constants -----------------------------------------------------
    ts = 1/settings.samplingFreq           # Sampling period in sec
    tc = 1/settings.codeFreqBasis          # code chip period in sec

    # Digitizing --------------------------------------------------------------
    # Make indices to read C/A code values. The length of the index array 
    # depends on the sampling frequency
    codeValueIndex = np.mod(np.floor(ts/tc * np.arange(sampleLen)),
                            settings.codeLength).astype(np.int32)
    
    # Make the digitized version of the C/A code ------------------------------
    # The "upsampled" code is made by selecting values from the C/A code chip 
    # array for the time instances of each sample.
    return caCode[codeValueIndex]
    

#%% For test purpose    
if __name__=='__main__':
    from initSettings import initSettings
    settings = initSettings()
    PRN = 1
    codeSample = generateCAcode(settings,PRN)
    