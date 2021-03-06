"""
To analyze and plot heavy-ion collision data produced by the Parton-Hadron-String Dynamics (PHSD) model.
"""

__version__ = '2.0.0'

from matplotlib.pyplot import rc
import matplotlib.pyplot as pl
import numpy as np
from particle import Particle

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
particles_of_interest = ['pi+','pi-','pi0','K+','K-','K0','K~0','eta','p','p~','n','n~','Lambda','Lambda~',\
  'Sigma0','Sigma~0','Sigma-','Sigma~+','Sigma+','Sigma~-','Xi-','Xi~+','Xi0','Xi~0','Omega-','Omega~+',\
    'D-','D+','D0','D~0','D(s)+','D(s)-','B-','B+','B0','B~0','B(s)0','B(s)~0']

list_quarks = ['u','d','s','c','b','U','D','S','C','B']
latex_quarks = dict(zip(list_quarks,['u','d','s','c','b','\\bar{u}','\\bar{d}','\\bar{s}','\\bar{c}','\\bar{b}']))

########################################################################
def N_quarks(particle_name):
    """
    Return number of quarks contained in a given particle
    """
    # convert to particle object
    try:
      part_obj = Particle.find(lambda p: p.name==particle_name)
    except:
      part_obj = Particle.findall(lambda p: p.name==particle_name)[0]
    # string containing the quark content
    string_quarks = part_obj.quarks
    # initialize array
    count = np.zeros(len(list_quarks))
    # special cases
    if(particle_name=='pi0'):
      count[0] += 1./2.
      count[1] += 1./2. 
      count[5] += 1./2. 
      count[6] += 1./2. 
      return count
    elif(particle_name=='eta'):
      count[0] += 1./6.
      count[1] += 1./6. 
      count[2] += 2./3. 
      count[5] += 1./6. 
      count[6] += 1./6. 
      count[7] += 2./3. 
      return count
      
    # scan string to find occurence of quarks
    for iq,quark in enumerate(list_quarks):
      startIndex = 0
      for _ in range(len(string_quarks)):
        # find 1st occurence of quark in string, from startIndex
        k = string_quarks.find(quark, startIndex)
        if(k != -1):
          # if found, count +1 and shift startIndex to look for other occurences
          startIndex = k+1
          count[iq] += 1.
          k = 0

    return count

# create a list containing name of particles of interests for the analysis
particle_analysis = particles_of_interest[:]
particle_analysis.remove('Sigma0')
particle_analysis.remove('Sigma~0')
particle_analysis.append('ch')

particle_info = {}
latex_name = {}
for ipart,name in enumerate(particles_of_interest):
  try:
    part_obj = Particle.find(lambda p: p.name==name)
  except:
    part_obj = Particle.findall(lambda p: p.name==name)[0]
  ID = part_obj.pdgid
  xname=name
  if(name=='Sigma0'):
    xname = 'Lambda'
  elif(name=='Sigma~0'):
    xname = 'Lambda~'
  particle_info.update({int(ID): [xname,part_obj.mass/1000.,np.sign(ID)*ID.is_baryon,ipart]})
  latex_name.update({name: r'$'+part_obj.latex_name+'$'})

particles_of_interest.remove('Sigma0')
particles_of_interest.remove('Sigma~0')

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
      label = latex_name[col]
    # if not found, use column name
    except:
      label = r'$'+col+'$'
      if(col=='\mu_{Q}'):
        data_y = -data_y
        label = r'$-'+col+'$'

    if(col=='Lambda'):
      label += '+'+latex_name['Sigma0']
    if(col=='Lambda~'):
      label += '+'+latex_name['Sigma~0']

    # check number of columns
    # if just one column (y & y_err) to plot, no label
    if(len(columns_y)==2):
      label = None

    line = ax.plot(data_x, data_y, linewidth='2.5', label=label)
    ax.fill_between(data_x, data_y-data_y_err, data_y+data_y_err, alpha=0.5, color=line[0].get_color())
    if(label!=None):
      ax.legend(title_fontsize=SMALL_SIZE, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)
    if(log):
      ax.set_yscale("log")
    else:
      ymin = 0.

  ax.set(ylim=[ymin,ymax])

  ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
  f.savefig(f"{outname}.png")
  f.clf()
  pl.close(f)
