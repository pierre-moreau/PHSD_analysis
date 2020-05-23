"""
To analyze and plot heavy-ion collision data produced by the Parton-Hadron-String Dynamics (PHSD) model.
"""

__version__ = '1.0.1'

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
        return xpart,mass,label
    raise Exception(f'particle {namepart} not found in list_part')

########################################################################
def plot_quant(df,xlabel,ylabel,title,outname,log=False):
  """
  Plot quantities and export
  """
  # inititialize plot
  f,ax = pl.subplots(figsize=(10,7))

  # label of all columns in the dataframe
  columns = [col for col in df]

  # check if there is a x_err column
  if(columns[1]==(columns[0]+'_err')):
    xerr = True
  else:
    xerr = False

  # columns in y-axis
  if(xerr):
    columns_y = columns[2:]
  else:
    columns_y = columns[1:]
    
  # initialize limit on y-axis for plot
  ymin = 1000.
  ymax = -1000.

  # iterate over columns_y
  for col in columns_y[::2]:
    # select data where it's nonzero
    nonzero = df[col] != 0.
    data_x = df[columns[0]][nonzero]
    data_y = df[col][nonzero]
    data_y_err = df[col+'_err'][nonzero]
    ymin_col = np.amin(data_y)
    if(ymin_col<ymin):
      ymin = ymin_col
    ymax_col = np.amax(data_y+data_y_err)
    if(ymax_col>ymax):
      ymax = ymax_col

    # try to find label in particles
    try:
      _,_,label = from_part(col)
    # if not found, use column name
    except:
      label = r'$'+col+'$'

    if(col=='Lambda0'):
      label += '+'+list_part[3212][2]
    if(col=='Lambdabar0'):
      label += '+'+list_part[-3212][2]

    # check number of columns
    # if just one column (y & y_err) to plot, no label
    if(len(columns_y)==2):
      label = None

    line = ax.plot(data_x, data_y, linewidth='2.5', label=label)
    ax.fill_between(data_x, data_y-data_y_err, data_y+data_y_err, alpha=0.5, color=line[0].get_color())
    if(label!=None):
      ax.legend(title_fontsize=SMALL_SIZE, loc='best', borderaxespad=0., frameon=False)

    if(log):
      ax.set_yscale("log")
    else:
      ymin = 0.

  ax.set(ylim=[ymin,ymax])

  ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
  f.savefig(f"{outname}.png")
