"""
To analyze and plot heavy-ion collision data produced by the Parton-Hadron-String Dynamics (PHSD) model.
"""

__version__ = '1.0.0'

from matplotlib.pyplot import rc
import matplotlib.pyplot as pl
import numpy as np

########################################################################
def str2bool(val):
  """
  convert string to bool (used to pass arguments with argparse)
  """
  dict = {'True': True, 'False': False, 'true': True, 'false': False, '1': True, '0': False}
  return dict[val]

########################################################################
# list of possible nuclei
nuclei = {1:'p', 2:'d', 12:'C', 16:'O', 40: 'Ca', 63: 'Cu', 129: 'Xe',197:'Au', 208: 'Pb', 238: 'U'}

DMN = 0.938 # nucleon mass
########################################################################
# list of particles of interest with KF conversion, masses, latex label:
list_part = {}
list_part.update({211:['pi+',0.138,r'$\pi^+$']}) 
list_part.update({-211:['pi-',0.138,r'$\pi^-$']}) # pi-
list_part.update({111:['pi0',0.138,r'$\pi^0$']}) # pi0

list_part.update({321:['K+',0.494,r'$K^+$']}) # K+
list_part.update({-321:['K-',0.494,r'$K^-$']}) # K-

list_part.update({2212:['p',DMN,r'$p$']}) # p
list_part.update({-2212:['pbar',DMN,r'$\bar{p}$']}) # pbar
list_part.update({2112:['n0',DMN,r'$n$']}) # n0
list_part.update({-2112:['nbar0',DMN,r'$\bar{n}$']}) # nbar0

list_part.update({3122:['Lambda0',1.115,r'$\Lambda$']}) # Lambda0
list_part.update({-3122:['Lambdabar0',1.115,r'$\bar{\Lambda}$']}) # Lambdabar0

list_part.update({3212:['Sigma0',1.189,r'$\Sigma^0$']}) # Sigma0
list_part.update({-3212:['Sigmabar0',1.189,r'$\bar{\Sigma}^0$']}) # Sigmabar0
list_part.update({3112:['Sigma-',1.189,r'$\Sigma^-$']}) # Sigma-
list_part.update({-3112:['Sigmabar+',1.189,r'$\bar{\Sigma}^+$']}) # Sigmabar+
list_part.update({3222:['Sigma+',1.189,r'$\Sigma^+$']}) # Sigma+
list_part.update({-3222:['Sigmabar-',1.189,r'$\bar{\Sigma}^-$']}) # Sigmabar-

list_part.update({3312:['Xi-',1.315,r'$\Xi^-$']}) # Xi-
list_part.update({-3312:['Xibar+',1.315,r'$\bar{\Xi}^+$']}) # Xibar+
list_part.update({3322:['Xi0',1.315,r'$\Xi^0$']}) # Xi0
list_part.update({-3322:['Xibar0',1.315,r'$\bar{\Xi}^0$']}) # Xibar0

list_part.update({3334:['Omega-',1.672,r'$\Omega^-$']}) # Omega-
list_part.update({-3334:['Omegabar+',1.672,r'$\bar{\Omega}^+$']}) # Omegabar+

########################################################################
# settings for plots
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30
rc('axes', linewidth=3) # width of axes
font = {'family' : 'Arial',
        'size' : MEDIUM_SIZE,
        'weight' : 'bold'}
rc('font', **font)  # controls default text sizes
rc('axes', titlesize=MEDIUM_SIZE, titleweight='bold')     # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE, labelweight='bold')    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE, direction='in', top='True')    # fontsize of the tick labels
rc('xtick.major', size=7, width=3, pad=10)
rc('ytick', labelsize=SMALL_SIZE, direction='in', right='True')    # fontsize of the tick labels
rc('ytick.major', size=7, width=3, pad=10)
rc('legend', fontsize=SMALL_SIZE, title_fontsize=SMALL_SIZE, handletextpad=0.25)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE, titleweight='bold')  # fontsize of the figure title
rc('savefig', dpi=300, bbox='tight')

########################################################################
def from_part(namepart):
    # find label in dict list_part
    for xpart,mass,label in list_part.values():
      if(xpart==namepart):
        break
    return xpart,mass,label

########################################################################
def plot_quant(data_x,data_y,xlabel,ylabel,title,outname,partplot=None,log=False):
    """
    Plot quantities and export
    """
    # inititialize plot
    f,ax = pl.subplots(figsize=(10,7))

    # check if data_x contain values + errors
    # only keep values
    try:
      data_x.shape[1]
      data_x = data_x[:,0]
    except:
      pass

    if(partplot==None):
      nonzero = data_y[:,0] != 0.
      data_x = data_x[nonzero]
      data_y = data_y[nonzero]
      ax.plot(data_x, data_y[:,0], color='black', linewidth='2.5')
      ax.fill_between(data_x, data_y[:,0]-data_y[:,1], data_y[:,0]+data_y[:,1], alpha=0.5, color='black')

      if(log):
        ymin = np.amin(data_y[:,0][data_y[:,0] != 0])
        ax.set_ylim(ymin)
        ax.set_yscale("log")

    else:
      for ip,part in enumerate(partplot):
        nonzero = data_y[:,ip,0] != 0.
        data_x = data_x[nonzero]
        data_y = data_y[nonzero]

        # find label in dict list_part
        _,_,label = from_part(part)
        
        if(part=='Lambda0'):
          label += '+'+list_part[3212][2]
        if(part=='Lambdabar0'):
          label += '+'+list_part[-3212][2]
        line = ax.plot(data_x, data_y[:,ip,0], linewidth='2.5', label=label)
        ax.fill_between(data_x, data_y[:,ip,0]-data_y[:,ip,1], data_y[:,ip,0]+data_y[:,ip,1], alpha=0.5, color=line[0].get_color())
        ax.legend(title_fontsize=SMALL_SIZE, loc='best', borderaxespad=0., frameon=False)

        if(log):
          ymin = np.amin(data_y[:,:,0][data_y[:,:,0] != 0])
          ax.set_ylim(ymin)
          ax.set_yscale("log")

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    f.savefig(f"{outname}.png")
