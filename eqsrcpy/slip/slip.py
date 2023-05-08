from scipy.interpolate import RBFInterpolator 
import matplotlib.pyplot as plt 
from matplotlib import cm, colors, colorbar
import numpy as np

def profiles(S, normalized=True):
    Nz,Nx = S.shape
    if normalized:
        mean_S = np.mean(S)
    else:
        mean_S = 1
   
    Sxi = [np.mean(S[:,i])/mean_S for i in range(Nx)]
    Szi = [np.mean(S[i,:])/mean_S for i in range(Nz)]
    return Sxi, Szi

def plotslipSPL(ax, S, dimWL, cmap = 'viridis',  cbpad=0, \
		iscolorbar=True, title='', aspect = None, cbshrink = 0.4, \
		ylabel='down-dip (km)', xlabel ='along-strike (km)'):
    extent = [0, dimWL[1], dimWL[0], 0]
    if aspect is None:
        aspect = dimWL[0]/dimWL[1]
    im = ax.imshow(S, extent = extent, aspect=aspect, \
                   cmap= cmap, interpolation='None')
                   
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if iscolorbar:
        cmap = cm.get_cmap(cmap)
        Smax = max(map(max, S))
        normcolor = colors.Normalize(vmin=0, vmax=Smax)        
        cbar = plt.colorbar(cm.ScalarMappable(norm=normcolor, cmap=cmap),\
                               ax=ax, shrink= cbshrink,\
                               orientation='vertical', label='slip (m)', pad=cbpad)

def resample(S,dimWL,newNzNx):
    [W,L] = dimWL
    #S_km = resampling(S,[W,L],[int(W/2),int(L/2)])
    new_S = resampling(S, [W,L], newNzNx)
    return new_S


def resampling(S,dimWL,newNzNx):
    #
    nrows, ncols = len(S), len(S[0])
    #print(nrows, ncols)
    dz, dx = dimWL[0]/nrows,  dimWL[1]/ncols
    #
    z = [i for i in np.arange(dz/2, dimWL[0], dz)]
    x = [i for i in np.arange(dx/2, dimWL[1], dx)]
    #
    [X, Z] = np.meshgrid(x, z)
    grid  = np.transpose([X.ravel(), Z.ravel()])
    fintp = RBFInterpolator(grid, np.array(S).ravel(), \
                            smoothing=0.5, kernel='linear') 
    # new grid
    new_dz, new_dx = dimWL[0]/newNzNx[0], dimWL[1]/newNzNx[1] 
    new_z = [i for i in np.arange(new_dz/2, dimWL[0], new_dz)]
    new_x = [i for i in np.arange(new_dx/2, dimWL[1], new_dx)]
    [new_X, new_Z] = np.meshgrid(new_x, new_z);
    new_grid = np.transpose([new_X.ravel(), new_Z.ravel()])
    mean_S = np.mean([np.mean(x) for x in S])
    new_S = fintp(new_grid).reshape(newNzNx[0], newNzNx[1])
    mean_new_S = np.mean(new_S)
    new_S = (new_S/mean_S)*mean_new_S
    return new_S


# lets workout effective extents
# Not just a python version but improved version of 
# MATLAB: http://equake-rc.info/cers-software/effsrcdim.  
# Note that: curently only for slipSPL

def normautocorr(x):
    # normalized autocorrelation 
    cor = np.correlate(x, x, mode='full')
    return [r/max(cor) for r in cor]


def get_Ntrim(Sx, dx):
    # Sx is one-sided right tapering profile of cummultive slip.
    # dx is spatial sampling
    rSx = np.array(Sx+[0.0]*5)
    Cx = normautocorr(rSx) 
    Wacf = np.trapz(Cx,x= None, dx = dx)/max(Cx)
    N = np.ceil(Wacf/dx)
    return int(len(rSx)-N-5) # Ntrim


def trim_profile(Sx, dx):
    # trim N for both ends of Sx  
    Idx = Sx.index(max(Sx))
      
    lSx = Sx[0:Idx+1]
    if len(lSx)>0:
       klSx = list(reversed(lSx))
       L = get_Ntrim([klSx[0]]*10+klSx, dx)
    else:
       L=0
 
    rSx = Sx[-(len(Sx)-Idx):]
    if len(rSx)>0:
       R = get_Ntrim([rSx[0]]*10+rSx, dx)
    else:
       R=0
    return (int(L), int(R))


def autocorrwidth(S, samp):
    # S is 2-D slip distrbution
    # samp is spatial smapling or grid spacing
    S = np.array(S)
    # calculate cummulative slips
    Sx = sum(S).tolist()       # columns - length
    Sz = [sum(x) for x in S]   # rows - width
    [L, R] = trim_profile(Sx, samp[1])
    [T, B] = trim_profile(Sz, samp[0])
   
    trim = [L,R, T, B] # - Ntrims
    Nx = len(Sx)-L-R
    Lacf = samp[1]*Nx
    Nz = len(Sz)-T-B
    Wacf = samp[0]*Nz

    return([Wacf,Lacf],trim)
    
    
def effextents(S, samp, doplot=False, tlabel='', \
            surfacerupture=False):
    # note - trim can be made false here        
    # get the autocorrwidth -----------------
    [WLacf, trim] = autocorrwidth(S, samp)
    [L, R, T, B] = trim
    S = np.array(S)
    origNz, origNx = S.shape
    large_slip = S.max()/3.0
    
    def adjust_trimN(N, smax, large_asperity_slip):
        if N ==0:
           adjusted_N = N
        else:            
           n=0
           is_large_asperity_slip = False
           for s in smaxs:
               if s>=large_asperity_slip:
                  is_large_asperity_slip = True
               if s>=large_asperity_slip/2:
                  n+=1  
                  if (N-n)==0:
                      break   
               
               else:
                   is_large_asperity_slip = False
                   break
                
           if is_large_asperity_slip:
               if (N-n)>0:
                   n+=1
        adjusted_N = N-n    
        doloop = True
        while doloop: 
           if smaxs[len(smax)-1-adjusted_N]==0:
              adjusted_N=adjusted_N+1
           else:
              doloop=False	
                
        return adjusted_N
    
    if surfacerupture:
        T = 0
    
    # adjust the trimming ---------------------------    
    # LEFT trim
    if L>0:
        leftS = S[::,:L+1]
        smaxs = leftS.max(axis=0)
        smaxs = smaxs[::-1]
        L = adjust_trimN(L, smaxs, large_slip)
    
    # Right trim
    if R>0:
        rightS = S[::,(origNx-(R+1)):]
        smaxs = rightS.max(axis=0)
        R = adjust_trimN(R, smaxs, large_slip)
    
    # Top trim
    if T>0:
        if not surfacerupture:
            topS = S[:T+1,::]
            smaxs = topS.max(axis=1)
            smaxs = smaxs[::-1]
            T = adjust_trimN(T, smaxs, large_slip)
    
    # Bottom trim
    if B>0:
        botS = S[(origNz-(B+1)):,::]
        smaxs = botS.max(axis=1)
        B = adjust_trimN(B, smaxs, large_slip)
    
    efftrim = [L, R, T, B]

    effLacf = (origNx-(L+R))*samp[1]
    effWacf = (origNz-(T+B))*samp[0]
    
    effmod = {'WLeff': [effWacf, effLacf],
              'WLacf': WLacf,
              'trimeff': efftrim,
              'trimacf': trim, }
    
    return effmod


def is_surfacerupture(mod, slipspl=True):
    # identify surface rupture events based on 2 conditions: 
    # (1) the top of the rupture model is within 1.5 km of 
    #  the surface (accounting for depth uncertainty), and 
    # (2) a very-large slip asperity is located within a 
    #  distance equal to 1/6 of rupture width from the surface. 
    #  A very-large slip asperity is defined as having slip ≥2/3 umax,
    #  where umaxis the overall maximum slip (see Mai et al., 2005).
    
    if slipspl:
        # currently we deal only with slipSPL
        if 'slipSPL' in mod.keys():
            S = np.array(mod['slipSPL'])/100
            S = S.tolist()
        else:
            print(modtag + ' no SPL found and is exlcuded')
            return None
    # get in meters
    if 'geoZ' in mod.keys():
        geoZ = mod['geoZ']
        ztop = np.min(geoZ) #min([min(z) for z in geoZ])
    else:
        nSEGM = mod['invSEGM']
        ztop = 9999
        for k in range(nSEGM):
            zkey = 'seg'+ str(k+1) + 'geoZ'
            geoZ = mod[zkey]
            ztop = np.min(geoZ)
        
    if ztop <=1.5:
        return True
    else:
    	return False



# we need to make a mod 
def get_effmod(mod, slipspl=True, doplot=False, plot_title=''):
    # warpper to effdims
    if slipspl:
        S = np.array(mod['slipSPL'])/100
        if S.size<50:
           print(mod['evTAG'], ': grids < 50, not processed')
           return None 
        S = S.tolist()
        
    else:
        print('Multi-fault is not yet implemented')
        
    
    dzdx = mod['invDzDx'].tolist()
    issurface = is_surfacerupture(mod)
    
    if issurface:
        effmod = effextents(S, dzdx, surfacerupture=True)
    else:
        effmod = effextents(S, dzdx)
        
    # trim 
    L,R,T,B = effmod['trimeff']
    effS = []
    Bk = len(S)-B
    meanS = np.mean(np.array(S))
    
    for k,s in enumerate(S):
        if (k<=T) | (k>=Bk):
            continue;
        del s[:L]
        if R>0:
           del s[-R:]
        effS.append(s)
    
    # scale effS
    meank = np.mean(np.array(effS))
    effS = np.array(effS)*(meanS/meank)
    effS = effS.tolist()
        
    # somehow S got f*, so gettting it again
    if slipspl:
        S = np.array(mod['slipSPL'])/100
        S = S.tolist()
    
    effmod.update({ 'evtag': mod['evTAG'], 'slip': S, 
    		    'slipeff': effS,
                   'dzdx': dzdx, 
                   'issurface': issurface, })
    
    if doplot:
        fig, ax = plt.subplots(figsize=(13,7))
      
        plotslipSPL(ax, S, mod['srcDimWL'], cmap = 'inferno_r', cbpad = 0.02, \
                   title=plot_title, cbshrink= 0.3)
        
        def get_bounds(effmod, isacf = False):
            dzdx = effmod['dzdx']
            if isacf:
                trimfunc, WL = 'trimacf', effmod['WLacf']
            else:
                trimfunc, WL = 'trimeff', effmod['WLeff']
                
            L,R,T,B = effmod[trimfunc]
            x = [L*dzdx[1], L*dzdx[1]+WL[1], 
                 L*dzdx[1]+ WL[1],L*dzdx[1],
                 L*dzdx[1],]
            y = [T*dzdx[0], T*dzdx[0],
                 T*dzdx[0]+ WL[0], T*dzdx[0]+ WL[0],
                T*dzdx[0]]
            return x,y
        
        x,y = get_bounds(effmod, isacf = False)
        ax.plot(x,y, 'r-', linewidth=4)
        x,y = get_bounds(effmod, isacf = True)
        ax.plot(x,y, 'k--', linewidth=2)
    
    return effmod


def get_profiles(effmod, NzNx= [40, 40], normalized = True):
    S = effmod['slipeff']
    dz,dx = effmod['dzdx']
    W,L = effmod['WLeff']
    new_S = resample(S,[W,L],NzNx)
    Sxi, Szi = profiles(new_S, normalized= normalized)
    ddz, ddx = W/NzNx[0], L/NzNx[1]
    zi = np.arange(ddz/2, W, ddz).tolist()
    xi = np.arange(ddx/2, L, ddx).tolist()
    #nxi = [round(x,3) for x in np.arange(1/(NzNx[1]+1), 1, 1/(NzNx[1]+1)).tolist()]
    #nzi = [round(z,3) for z in np.arange(1/(NzNx[0]+1), 1, 1/(NzNx[0]+1)).tolist()]
    nxi = [round(x,3) for x in np.arange(1/(NzNx[1]*2), 1, 1/(NzNx[1])).tolist()]
    nzi = [round(z,3) for z in np.arange(1/(NzNx[0]*2), 1, 1/(NzNx[0])).tolist()]
    
    	
    effmod.update({'profile': {
                        'strike':[Sxi, xi, nxi],
                        'dip':[Szi, zi, nzi],
                              }
                  })
    return effmod


def get_topprofile(effmod, NzNx= [40, 40], normalized = True):
    S = effmod['slipeff']
    dz,dx = effmod['dzdx']
    W,L = effmod['WLeff']
    new_S = resample(S,[W,L],NzNx)
    if normalized:
        ss = new_S[0].tolist()
        ss = [s/np.mean(ss) for s in ss]
    else:
        ss = new_S[0].tolist()
    ddx =  L/NzNx[1]
    xi = np.arange(ddx/2, L, ddx).tolist()
    nxi = [round(x,3) for x in np.arange(1/(2*NzNx[1]), 1, 1/(NzNx[1])).tolist()]
    
    effmod.update({'surface': [ss, xi, nxi],
                  })
    return effmod
     


