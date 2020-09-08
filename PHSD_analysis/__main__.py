import os # to get the path of the current directory
import re
import pandas as pd
import argparse
from numpy import pi
from . import *
from EoS_HRG.test.plot_HRG import plot_freezeout
import time
import subprocess
import tempfile

###############################################################################
__doc__ = """Analyse the PHSD.dat files, output and plot observables"""
###############################################################################
parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
        '--folder', type=str, default='./',
        help='folder containing the TmunuTAU.dat file or the job_# folders to analyse')
parser.add_argument(
        '--BMIN', type=float, default=None,
        help='Select BMIN')
parser.add_argument(
        '--BMAX', type=float, default=None,
        help='Select BMAX')
parser.add_argument(
        '--midrapy', type=float, default=0.5,
        help='Select mid-rapidity region |y| < midrapy')
parser.add_argument(
        '--midrapeta', type=float, default=0.5,
        help='Select mid-rapidity region |eta| < midrapeta')
parser.add_argument(
        '--ybin', type=float, default=0.25,
        help='bin in y')
parser.add_argument(
        '--etabin', type=float, default=0.25,
        help='bin in eta')
parser.add_argument(
        '--pTbin', type=float, default=0.25,
        help='bin in pT')
parser.add_argument(
        '--mTbin', type=float, default=0.25,
        help='bin in mT')
parser.add_argument(
        '--freezeout', type=str2bool, default=False,
        help='Evaluate chemical freeze out parameters')
args = parser.parse_args()

# info about the analysis
xBMIN = args.BMIN
xBMAX = args.BMAX
midrapy = args.midrapy
midrapeta = args.midrapeta
ybin = args.ybin
etabin = args.etabin
pTbin = args.pTbin
mTbin = args.mTbin
freezeout = args.freezeout
# path where to look for files
path = args.folder

########################################################################
# set the bin size at midrapidity
deta = 2.*midrapeta
dy = 2.*midrapy

########################################################################
dir_path = os.path.dirname(os.path.realpath(__file__)) # path where the script is
cwd = os.getcwd() 
if(path=='./'):
    path = cwd+'/'
else:
    path = cwd+'/'+path+'/' 
print(f'Analysis files are: {dir_path}')
print(f'Current working directory: {cwd}')
print(f'Looking for files in: {path}')

########################################################################
def detect_files():
    """
    Detect if there is a phsd.dat file in the folder
    OR folders denoted as job_#
    OR ask if nothing is found
    """

    # get all files/folders in path
    entries_all = os.listdir(str(path))

    # where to read the files
    path_input = []
    path_folders = []
    path_files = []

    # try to find a match with phsd.dat first in the current folder
    for xfile in entries_all:
        if(xfile == 'phsd.dat'):
            path_files = [path+xfile]

    # if it didn't work
    if(len(path_files) == 0):
        # try to find a match with folders as job_#
        for xfold in entries_all:
            match = re.match('job_([0-9]+)', xfold)
            try:
                match = match.group(1)
                path_folders += [path+xfold]
            except:
                pass

        # now check is there is actually a phsd.dat in the selected folders
        for xfold in path_folders:
            files_all = os.listdir(str(xfold))
            for xfile in files_all:
                if(xfile == 'phsd.dat'):
                    path_files += [xfold+'/'+xfile]

    # just read inputPHSD once
    path_input = path_files[0].replace('phsd.dat','inputPHSD')

    path_files.sort()
    # display list of folders found
    print(f'\nThese files have been found: {path_files}')

    return path_input,path_files

########################################################################
def read_input(path_input):
    """
    Read inputPHSD file and display information
    """
    inputf = {}
    with open (path_input,"r") as inputfile:
        for line in inputfile:
            list_line=line.split()
            if(len(list_line)==0):
                continue
            x = list_line[0].replace(',','')
            inputf.update({list_line[1].replace(':',''): float(x) if '.' in x else int(x)})

    global xBMIN
    global xBMAX
    # check impact param
    if(xBMIN==None):
        xBMIN = inputf['BMIN']
    elif((xBMIN!=None) and (xBMIN < inputf['BMIN'])):
        xBMIN = inputf['BMIN']
        print('Entered BMIN changed to min value in b')
    if(xBMAX==None):
        xBMAX = inputf['BMAX']
    elif((xBMAX!=None) and (xBMAX > inputf['BMAX'])):
        xBMAX = inputf['BMAX']
        print('Entered BMAX changed to max value in b')
    
    inputf.update({'xBMIN':xBMIN,'xBMAX':xBMAX})

    global list_b_all
    list_b_all = np.arange(start=inputf["BMIN"],stop=inputf["BMAX"]+0.0001,step=inputf['DBIMP'])

    # nuclei information
    N1 = nuclei[inputf["MASSTA"]]
    N2 = nuclei[inputf["MASSPR"]]

    # if pp collision
    global pp
    pp = False
    if(N1=='p' and N2=='p'):
        pp = True

    if(inputf['IGLUE'] == 0):
        print('      __ __  ____  ___                        ')
        print('     / // / / __/ / _ \                       ')
        print('    / _  / _\ \  / // /                       ')
        print('   /_//_/ /___/ /____/ ____  __ ____ ____ ___ ')
        print('  / _ |  / |/ // _ |  / /\ \/ // __//  _// __/')
        print(' / __ | /    // __ | / /__\  /_\ \ _/ / _\ \  ')
        print('/_/ |_|/_/|_//_/ |_|/____//_//___//___//___/  ')
        print('                                              ')
    else:
        print('       ___   __ __ ____ ___                   ')
        print('      / _ \ / // // __// _ \                  ')
        print('     / ___// _  /_\ \ / // /                  ')
        print('    /_/   /_//_//__ //____/  __ ____ ____ ____')
        print('  / _ |  / |/ // _ |  / /\ \/ // __//  _// __/')
        print(' / __ | /    // __ | / /__\  /_\ \ _/ / _\ \  ')
        print('/_/ |_|/_/|_//_/ |_|/____//_//___//___//___/  ')
        print('                                              ')

    # Calculation of sqrt(s_NN) and y_beam/y_proj
    TLAB = inputf['ELAB'] # kinetic lab energy
    ELAB = TLAB + DMN # lab energy
    SRT=np.sqrt(2.*DMN*(ELAB+DMN)) # collisional energy in the CM
    gamma=SRT/2./DMN # gamma factor of the nuclei

    Pproj = np.sqrt((SRT/2.)**2-DMN**2) # nucleon momentum in the CM
    yproj=0.5*np.log((SRT/2.+Pproj)/(SRT/2.-Pproj)) # rapidity

    R1 = 1.124*inputf['MASSTA']**(1./3) # radius of target
    R2 = 1.124*inputf['MASSPR']**(1./3) # radius of projectile

    # update dictionnary
    inputf.update({'SRT': SRT, 'y': yproj, 'gamma': gamma, 'R1': R1, 'R2':R2})

    # create string for output files
    inputf.update({'collstring': f'{N1}{N2}{int(SRT)}GeV_'})

    # select particles in abs(y) < ylim or abs(eta) < etalim
    etalim = int(round(inputf['y'])+1)
    ylim = int(round(inputf['y'])+1)
    inputf.update({'etalim': etalim,'ylim': ylim})
    inputf.update({'etabin': etabin,'ybin': ybin})
    inputf.update({'midrapeta': midrapeta,'midrapy': midrapy})

    # list of y and eta for each bin
    global list_eta,list_y,edges_eta,edges_y
    list_eta = np.arange(start=-etalim,stop=etalim+0.0001,step=etabin)
    list_y = np.arange(start=-ylim,stop=ylim+0.0001,step=ybin)  
    edges_eta = np.arange(start=-etalim-etabin/2.,stop=etalim+etabin/2.+0.0001,step=etabin)
    edges_y = np.arange(start=-ylim-ybin/2.,stop=ylim+ybin/2.+0.0001,step=ybin)
    inputf.update({'edges_eta': edges_eta,'edges_y': edges_y})
    inputf.update({'etabin': etabin,'ybin': ybin})

    # max values of pT, mT
    pTmax = 4.
    mTmax = 4.
    inputf.update({'pTmax': pTmax,'mTmax': mTmax})
    inputf.update({'pTbin': pTbin,'mTbin': mTbin})
    # list of mT and pT for each bin
    global list_pT,list_mT,edges_pT,edges_mT
    list_pT = np.arange(start=pTbin/2.,stop=pTmax-pTbin/2.+0.0001,step=pTbin)
    list_mT = np.arange(start=mTbin/2.,stop=mTmax-mTbin/2.+0.0001,step=mTbin)
    edges_pT = np.arange(start=0.,stop=pTmax+0.0001,step=pTbin)
    edges_mT = np.arange(start=0.,stop=mTmax+0.0001,step=mTbin)
    inputf.update({'edges_pT': edges_pT,'edges_mT': edges_mT})

    # display information about the collision
    print(f'{N1}+{N2} at $\sqrt{{s_{{NN}}}}$ = {SRT:4.1f} GeV -- TLAB = {TLAB} GeV -- yproj = {yproj}')
    print(f'BMIN = {xBMIN} fm -- BMAX = {xBMAX} fm -- DELTAB = {inputf["DBIMP"]} fm')
    print(f'ISUBS = {int(inputf["ISUBS"])} -- NUM = {int(inputf["NUM"])} \n')

    return inputf

########################################################################
def read_data_F(path_files,inputf):
    """
    Interface to read_files.F
    All necessary data/information are transferred in temporary folder /analysis_files/
    """

    ########################################################################
    def load_data(string_obs,xb,part=None):
        """
        Read data produced in folder /analysis_files/
        """
        if(part==None):
            return np.loadtxt(f'./{string_obs}_b{int(10.*xb):03d}.dat')
        else:
            return np.loadtxt(f'./{string_obs}_b{int(10.*xb):03d}_{part}.dat')

    ########################################################################
    # create temporary folder
    with tempfile.TemporaryDirectory(prefix='PHSD-', dir=path) as resdir:
        # change working directory
        os.chdir(resdir)

        # output quantities from inputf
        with open('./inputf.dat','w') as input_file:
            for val in ['BMIN','xBMIN','BMAX','xBMAX','DBIMP','ISUBS','NUM','MASSTA','MASSPR','y',\
                'midrapy','midrapeta','ybin','ylim','etabin','etalim','pTbin','pTmax','mTbin','mTmax']:
                input_file.write(f'{inputf[val]},{val}\n')

        # output paths of phsd.dat files
        with open('./path_phsd.dat','w') as path_file:
            path_file.write(f'{len(path_files)}\n')
            for val in path_files:
                path_file.write(f'{val}\n')

        # output particle information
        with open('./particle_info.dat','w') as particle_file:
            particle_file.write(f'{len(particle_info)}\n')
            for val in particle_info:
                particle_file.write(f'{particle_info[val][3]},{val},{particle_info[val][0]},{particle_info[val][1]},{particle_info[val][2]}\n')

        # execute subroutines to read files
        start = time.time()
        # compile fortran program if it doesn't exist yet
        #if(not(os.path.exists(f'{dir_path}/read_files_F'))):
        print('Compilation of Fortran code to read files\n')
        subprocess.run(['gfortran','-O3','-g','-fbacktrace',f'{dir_path}/read_files.F', '-o',f'{dir_path}/read_files_F'])
        # execute fortran program
        subprocess.run([f'{dir_path}/read_files_F'])
        end = time.time()
        print(f'Took in total {end-start}s for the analysis')

        dict_obs = {}
        # create new entry for each b in dict if it doesn't already exist
        for xb in list_b_all:
            dict_obs.update({f'Nevents(b={xb})': load_data('Nevents',xb)})
            dict_obs.update({f'Npart(b={xb})': load_data('Npart',xb)})
            for part in particle_analysis:
                dict_obs.update({f'dNdeta(b={xb};{part})': load_data('dNdeta',xb,part)})
                dict_obs.update({f'dNdy(b={xb};{part})': load_data('dNdy',xb,part)})
                dict_obs.update({f'mean_pT(b={xb};{part})': load_data('mean_pT',xb,part)})
                dict_obs.update({f'N_mean_pT(b={xb};{part})': load_data('N_mean_pT',xb,part)})
                dict_obs.update({f'dNdeta_eta(b={xb};{part})': load_data('dNdeta_eta',xb,part)})
                dict_obs.update({f'dNdy_y(b={xb};{part})': load_data('dNdy_y',xb,part)})
                dict_obs.update({f'dNdpT_pT(b={xb};{part})': load_data('dNdpT_pT',xb,part)})
                dict_obs.update({f'dNdmT_mT(b={xb};{part})': load_data('dNdmT_mT',xb,part)})
            dict_obs.update({f'dNBBBARdy_y(b={xb})': load_data('dNBBBARdy_y',xb)})
    
    return dict_obs

########################################################################
def read_data(path_files,inputf):
    """
    read all the phsd.dat files specified in path_files
    """

    # create dictionnary containing information about all observables of interests
    # dict_obs[<Observable>(b=<impact parameter>;<particle name>)]
    # Ex: dict_obs['dNdpT(b=2;Omega-)']
    dict_obs = {}

    # create new entry for each b in dict if it doesn't already exist
    for xb in list_b_all:
        dict_obs.update({f'Nevents(b={xb})': 0})
        dict_obs.update({f'Npart(b={xb})': np.zeros(2)})
        for part in particle_analysis:
            dict_obs.update({f'dNdeta(b={xb};{part})': np.zeros(2)})
            dict_obs.update({f'dNdy(b={xb};{part})': np.zeros(2)})
            dict_obs.update({f'mean_pT(b={xb};{part})': np.zeros(2)})
            dict_obs.update({f'N_mean_pT(b={xb};{part})': 0})
            dict_obs.update({f'dNdeta_eta(b={xb};{part})': np.zeros((len(list_eta),2))})
            dict_obs.update({f'dNdy_y(b={xb};{part})': np.zeros((len(list_y),2))})
            dict_obs.update({f'dNdpT_pT(b={xb};{part})': np.zeros((len(list_pT),2))})
            dict_obs.update({f'dNdmT_mT(b={xb};{part})': np.zeros((len(list_mT),2))})
            dict_obs.update({f'dNBBBARdy_y(b={xb})': np.zeros((len(list_y),2))})

    # data type to read the phsd.dat files
    # header format (first line)
    # N, ISUB, IRUN, BIMP, IBweight
    header1_dtype = [
                ('N_particles', np.int16),
                ('ISUB', np.int16),
                ('NUM', np.int16),
                ('b', np.float64)
                ]
    # header format (2nd line)
    # Np, phi2, epsi2, phi3, epsi3, phi4, epsi4, phi5, epsi5
    header2_dtype = [
                ('Np', np.int16),
                ('phi2', np.float64),
                ('epsi2', np.float64),
                ('phi3', np.float64),
                ('epsi3', np.float64),
                ('phi4', np.float64),
                ('epsi4', np.float64),
                ('phi5', np.float64),
                ('epsi5', np.float64)
                ]
    # format to read particle information
    parts_dtype = [
                ('ID', np.int16),
                ('IDQ', np.int16),
                ('PX', np.float64),
                ('PY', np.float64),
                ('PZ', np.float64),
                ('P0', np.float64),
                ('iHist', np.int16),
                ('IPI5', np.int16)
                ]

    ########################################################################
    def add_to_bin(string_event,value,edges,bin_size,weight=1.):
        """
        Add value to histogram stored in dict_event[string_event]
        """
        if((value>edges[0]) and (value<edges[-1])):
            # find bin in list
            ibin = int((value-edges[0])/bin_size)
            # add weight to the bin
            dict_event[string_event][ibin] += weight

    ########################################################################
    def add_obs(obs,xb,part,bool_float=True):
        """
        Add observable of each event in dict_event to dict_obs
        """
        if(part):
            string_event = f'{obs}({part})'
            string_obs = f'{obs}(b={xb};{part})'
        else:
            string_event = f'{obs}'
            string_obs = f'{obs}(b={xb})'

        data = dict_event[string_event]
        if(bool_float):
            dict_obs[string_obs] += np.array([data,data**2.])
        else:
            dict_obs[string_obs] += np.column_stack((data,data**2.))

    ########################################################################
    # loop over all phsd.dat files
    for xfile in path_files:
        print(f'Reading {xfile}')
        with open (xfile,"r") as data_file:
            start = time.time()

            # loop inside files over impact parameter
            for xb in list_b_all:
                # skip reading files if B > xBMAX
                if(xb>xBMAX):
                    break
                # loop inside impact parameters over ISUB
                for ISUB in range(inputf["ISUBS"]):

                    if(xb<xBMIN):
                        print(f'   - ISUB = {ISUB+1} ; B = {xb} fm ; (not counted)')
                    else:
                        print(f'   - ISUB = {ISUB+1} ; B = {xb} fm')

                    # loop inside ISUB over NUM
                    for _ in range(inputf["NUM"]):

                        try:
                            # header (1st line)
                            header1 = np.loadtxt(data_file,dtype=header1_dtype,max_rows=1)
                            N_particles=header1['N_particles']
                            current_b=header1['b']
                            # header (2nd line)
                            header2 = np.loadtxt(data_file,dtype=header2_dtype,max_rows=1)
                            # read particles
                            list_particles = np.array(np.loadtxt(data_file,dtype=parts_dtype,max_rows=N_particles).tolist())
                        except:
                            continue

                        # only count inelastic events
                        if(N_particles>(inputf["MASSTA"]+inputf["MASSPR"])):

                            # count number of events per impact parameter
                            dict_obs[f'Nevents(b={current_b})'] += 1

                            # save Npart for each event
                            Np = header2['Np'] # number of participants
                            dict_obs[f'Npart(b={current_b})'] += [Np,Np**2.]

                            #############################################################################
                            # calculate observables for this event and update corresponding dictionnary #
                            #############################################################################
                            
                            # initialize dict containing particle info for this event
                            dict_event = {}
                            for part in particle_analysis:
                                dict_event.update({f'dNdeta({part})': 0.})
                                dict_event.update({f'dNdy({part})': 0.})
                                dict_event.update({f'mean_pT({part})': np.zeros(2)})
                                dict_event.update({f'N_mean_pT({part})': 0})
                                dict_event.update({f'dNdeta_eta({part})': np.zeros((len(list_eta)))})
                                dict_event.update({f'dNdy_y({part})': np.zeros((len(list_y)))})
                                dict_event.update({f'dNdpT_pT({part})': np.zeros((len(list_pT)))})
                                dict_event.update({f'dNdmT_mT({part})': np.zeros((len(list_mT)))})
                            dict_event.update({'dNBBBARdy_y': np.zeros((len(list_y)))})

                            for ID,IDQ,PX,PY,PZ,P0,iHist,_ in list_particles:
                                try:
                                    name,part_mass,baryon_N,_ = particle_info[ID]
                                except:
                                    continue

                                y = 0.5*np.log((P0+PZ)/(P0-PZ))

                                if(name=='p' or name=='n'):
                                # if |y| > yproj-0.5, don't count
                                # 0.5 is arbitrary, but works fine
                                    if((iHist==-1) and (abs(y)>(inputf['y']-0.5))):
                                        continue

                                PP = np.sqrt(PX**2.+PY**2.+PZ**2.)
                                eta = 0.5*np.log((PP+PZ)/(PP-PZ))
                                pT = np.sqrt(PX**2.+PY**2.)
                                mT = np.sqrt(pT**2.+part_mass**2.)
                                #phi = np.arctan2(PY,PX)

                                if(IDQ!=0):
                                    if(abs(eta)<=midrapeta):
                                        # dN/deta
                                        dict_event['dNdeta(ch)'] += 1./deta
                                    if(abs(y)<=midrapy):
                                        # dN/dy
                                        dict_event['dNdy(ch)'] += 1./dy
                                        # mean pT
                                        dict_event['mean_pT(ch)'] += [pT,pT**2.]
                                        dict_event['N_mean_pT(ch)'] += 1.
                                        # dNdpT_pT
                                        add_to_bin('dNdpT_pT(ch)',pT,edges_pT,pTbin,weight=1./(2.*pi*pT*pTbin*dy))
                                        # dNdmT_mT
                                        add_to_bin('dNdmT_mT(ch)',mT-part_mass,edges_mT,mTbin,weight=1./(2.*pi*mT*mTbin*dy))
                                    
                                    # dNdeta_eta
                                    add_to_bin('dNdeta_eta(ch)',eta,edges_eta,etabin,weight=1./etabin)
                                    # dNdy_y
                                    add_to_bin('dNdy_y(ch)',y,edges_y,ybin,weight=1./ybin)
                                
                                if(abs(eta)<=midrapeta):
                                    # dN/deta
                                    dict_event[f'dNdeta({name})'] += 1./deta
                                if(abs(y)<=midrapy):
                                    # dN/dy
                                    dict_event[f'dNdy({name})'] += 1./dy
                                    # mean pT
                                    dict_event[f'mean_pT({name})'] += [pT,pT**2.]
                                    dict_event[f'N_mean_pT({name})'] += 1.
                                    # dNdpT_pT
                                    add_to_bin(f'dNdpT_pT({name})',pT,edges_pT,pTbin,weight=1./(2.*pi*pT*pTbin*dy))
                                    # dNdmT_mT
                                    add_to_bin(f'dNdmT_mT({name})',mT-part_mass,edges_mT,mTbin,weight=1./(2.*pi*mT*mTbin*dy))
                                
                                # dNdeta_eta
                                add_to_bin(f'dNdeta_eta({name})',eta,edges_eta,etabin,weight=1./etabin)
                                # dNdy_y
                                add_to_bin(f'dNdy_y({name})',y,edges_y,ybin,weight=1./ybin)
                                # dNdy_y of baryons minus antibaryons
                                add_to_bin('dNBBBARdy_y',y,edges_y,ybin,weight=baryon_N/ybin)
                            
                            # loop over all particles
                            for part in particle_analysis:
                                # dNdeta
                                add_obs('dNdeta',current_b,part)
                                # dNdy
                                add_obs('dNdy',current_b,part)
                                # mean_pT
                                dict_obs[f'mean_pT(b={current_b};{part})'] += dict_event[f'mean_pT({part})']
                                dict_obs[f'N_mean_pT(b={current_b};{part})'] += dict_event[f'N_mean_pT({part})']
                                # dNdeta_eta
                                add_obs('dNdeta_eta',current_b,part,False)
                                # dNdy_y
                                add_obs('dNdy_y',current_b,part,False)
                                # dNdpT_pT
                                add_obs('dNdpT_pT',current_b,part,False)
                                # dNdmT_mT
                                add_obs('dNdmT_mT',current_b,part,False)
                            # dNBBBARdy_y
                            add_obs('dNBBBARdy_y',current_b,None,False)  

            end = time.time()
            print(f'Took in total {end-start}s for the analysis')

    return dict_obs

########################################################################
def calculate_obs(dict_obs,inputf,particles):

    print('\nCalculating outputs')

    # format for output files
    out_str = inputf['collstring']

    # list of impact parameters
    # which b to select for output?
    list_b = list_b_all[(xBMIN <= list_b_all) & (list_b_all <= xBMAX)]

    # nuclei information
    N1 = nuclei[inputf["MASSTA"]]
    N2 = nuclei[inputf["MASSPR"]]
    plot_title = f"{N1}+{N2} at $\sqrt{{s_{{NN}}}}$ = {inputf['SRT']:4.1f} GeV"

    # label for particles
    label_part = []
    for part in particles:
        label_part += [part,part+'_err']

    ####################################################################
    def return_obs(obs,xb,part=None,weight=1.,squared=False):
        """
        Return mean and standard mean error from dict_obs
        where each list contains [\sum_i x_i, \sum_i x_i**2.] with i indicating the event number
        The confidence interval is given as 3.3 (\approx t-value for 99.9% confidence interval) times the standard mean error
        """
        # total number of events per b
        Nevtot = dict_obs[f'Nevents(b={xb})'] 
        if(obs=='mean_pT'):
            Nevtot = dict_obs[f'N_mean_pT(b={xb};{part})']
        
        if(Nevtot==0):
            return None,None

        if(part==None):
            string_obs = f'{obs}(b={xb})'
        else:
            string_obs = f'{obs}(b={xb};{part})'
        
        # mean value for the observable, just divide the sum by total number of events
        if(isinstance(dict_obs[string_obs][0],float)):
            mean_x = dict_obs[string_obs][0]/Nevtot
            mean_x2 = dict_obs[string_obs][1]/Nevtot
        else:
            mean_x = dict_obs[string_obs][:,0]/Nevtot
            mean_x2 = dict_obs[string_obs][:,1]/Nevtot

        # the standard deviation is calculated as
        # sqrt(E[x**2.]-E[x]**2) where E[x] is the expected value of x: 1/N \sum_i x_i
        std_dev = np.sqrt(mean_x2 - mean_x**2.)

        # standard error of the mean is \sigma/sqrt(N)
        sem = std_dev/np.sqrt(Nevtot)

        # confidence interval
        t_value = 3.3
        error = t_value*sem

        # if mean is a float, just return floats
        if(isinstance(mean_x, float)):
            return mean_x,error
        # if it's an array, return zipped array of mean and sem or sem**2.
        else:
            if(squared):
                return np.array(list(zip(weight*mean_x,weight**2.*error**2.)))
            else:
                return np.array(list(zip(weight*mean_x,abs(weight)*error)))

    ####################################################################
    # quantities as a function of Npart
    def quant_Npart(Nparts):
        """
        Export observables as a function of Npart
        dN_ch/deta & dN_ch/dy for charged particles
        dN/dy & <pT> for each particles
        Chemical freeze-out parameters as a function of Npart
        """
        print("   - observables as a function of Npart")

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_b).T,Nparts),axis=1), columns=['b','Npart','Npart_err'])
        dict_out.to_csv(path+out_str+'Npart_b.csv', index=False, header=True)
        if(len(Nparts)>1):
            plot_quant(dict_out,r'$b [fm]$',r'$N_{part}$',plot_title,path+out_str+'Npart_b')

        # dN_ch/deta & dN_ch/dy + error
        dNchdeta = np.zeros((len(list_b),2))
        dNchdy = np.zeros((len(list_b),2))
        for ib,xb in enumerate(list_b):
            dNchdeta[ib] = return_obs('dNdeta',xb,part='ch')
            dNchdy[ib] = return_obs('dNdy',xb,part='ch')

        dict_out = pd.DataFrame(np.concatenate((Nparts,dNchdeta),axis=1), columns=['Npart','Npart_err','dNchdeta','dNchdeta_err'])
        dict_out.to_csv(path+out_str+'dNchdeta_Npart.csv', index=False, header=True)
        if(len(Nparts)>1):
            plot_quant(dict_out,r'$N_{part}$',f'$dN_{{ch}}/d\eta|_{{|\eta|<{midrapeta}}}$',plot_title,path+out_str+'dNchdeta_Npart')

        dict_out = pd.DataFrame(np.concatenate((Nparts,dNchdy),axis=1), columns=['Npart','Npart_err','dNchdy','dNchdy_err'])
        dict_out.to_csv(path+out_str+'dNchdy_Npart.csv', index=False, header=True)
        if(len(Nparts)>1):
            plot_quant(dict_out,r'$N_{part}$',f'$dN_{{ch}}/dy|_{{|y|<{midrapy}}}$',plot_title,path+out_str+'dNchdy_Npart')

        # dN/dy & <pT>
        dNdy = np.zeros((len(list_b),len(particles),2))
        mean_pT = np.zeros((len(list_b),len(particles),2))
        for ib,xb in enumerate(list_b):
            for ip,part in enumerate(particles):
                dNdy[ib,ip] = return_obs('dNdy',xb,part=part)
                mean_pT[ib,ip] = return_obs('mean_pT',xb,part=part)

        dict_out = pd.DataFrame(np.concatenate((Nparts,dNdy.reshape((len(list_b),len(particles)*2))),axis=1), columns=['Npart','Npart_err']+label_part)
        dict_out.to_csv(path+out_str+'dNdy_Npart.csv', index=False, header=True)
        if(len(Nparts)>1):
            plot_quant(dict_out,r'$N_{part}$',r'$dN/dy$',plot_title,path+out_str+'dNdy_Npart',log=True)

        # Evaluate freeze-out parameters by using yields dN/dy at midrapidity
        if(freezeout):
            n_freeze_yields = 7 # T,muB,muQ,muS,gamma_S,dV/dy,s/n_B
            n_freeze_ratios = 6 # T,muB,muQ,muS,gamma_S,s/n_B
            n_freeze_yields_nS0 = 7 # T,muB,muQ,muS,gamma_S,dV/dy,s/n_B
            n_freeze_ratios_nS0 = 6 # T,muB,muQ,muS,gamma_S,s/n_B
            freeze_out_yields = np.zeros((len(list_b),n_freeze_yields,2))
            freeze_out_ratios = np.zeros((len(list_b),n_freeze_ratios,2))
            freeze_out_yields_nS0 = np.zeros((len(list_b),n_freeze_yields_nS0,2))
            freeze_out_ratios_nS0 = np.zeros((len(list_b),n_freeze_ratios_nS0,2))
            for ib,xb in enumerate(list_b):
                dict_yield = {}
                for ip,part in enumerate(particles):
                    dict_yield.update({part:dict_out[part][ib]})
                    dict_yield.update({part+'_err':dict_out[part+'_err'][ib]})
                # PHSD results are without weak decays of hyperons (=no_feeddown for all particles)
                method = 'all'
                fit_result = plot_freezeout(dict_yield,method=method,chi2_plot=False,offshell=False,freezeout_decay='PHSD',no_feeddown='all',\
                    plot_file_name=path+out_str+f'freezout_b{int(10*xb):03d}') # data with plot
                if(method=='all' or method=='yields'):
                    freeze_out_yields[ib] = np.append(fit_result['fit_yields'],[fit_result['snB_yields']],axis=0)
                    freeze_out_yields_nS0[ib] = np.append(fit_result['fit_yields_nS0'],[fit_result['snB_yields_nS0']],axis=0)
                if(method=='all' or method=='ratios'):
                    freeze_out_ratios[ib] = np.append(fit_result['fit_ratios'],[fit_result['snB_ratios']],axis=0)
                    freeze_out_ratios_nS0[ib] = np.append(fit_result['fit_ratios_nS0'],[fit_result['snB_ratios_nS0']],axis=0)

            if(method=='all' or method=='yields'):
                dict_out = pd.DataFrame(np.concatenate((Nparts,freeze_out_yields.reshape((len(list_b),n_freeze_yields*2))),axis=1), \
                    columns=['Npart','Npart_err','T_{ch}','T_{ch}_err','\mu_{B}','\mu_{B}_err','\mu_{Q}','\mu_{Q}_err','\mu_{S}','\mu_{S}_err',\
                        '\gamma_{S}','\gamma_{S}_err','dV/dy','dV/dy_err','s/n_{B}','s/n_{B}_err'])
                dict_out.to_csv(path+out_str+'freezout_Npart_yields.csv', index=False, header=True)
                if(len(Nparts)>1):
                    plot_quant(dict_out,r'$N_{part}$','freezeout parameters',plot_title,path+out_str+'freezeout_Npart_yields',log=True)

                dict_out = pd.DataFrame(np.concatenate((Nparts,freeze_out_yields_nS0.reshape((len(list_b),n_freeze_yields_nS0*2))),axis=1), \
                    columns=['Npart','Npart_err','T_{ch}','T_{ch}_err','\mu_{B}','\mu_{B}_err','\mu_{Q}','\mu_{Q}_err','\mu_{S}','\mu_{S}_err',\
                        '\gamma_{S}','\gamma_{S}_err','dV/dy','dV/dy_err','s/n_{B}','s/n_{B}_err'])
                dict_out.to_csv(path+out_str+'freezout_Npart_yields_nS0.csv', index=False, header=True)
                if(len(Nparts)>1):
                    plot_quant(dict_out,r'$N_{part}$','freezeout parameters',plot_title,path+out_str+'freezeout_Npart_yields_nS0',log=True)

            if(method=='all' or method=='ratios'):
                dict_out = pd.DataFrame(np.concatenate((Nparts,freeze_out_ratios.reshape((len(list_b),n_freeze_ratios*2))),axis=1), \
                    columns=['Npart','Npart_err','T_{ch}','T_{ch}_err','\mu_{B}','\mu_{B}_err','\mu_{Q}','\mu_{Q}_err','\mu_{S}','\mu_{S}_err',\
                        '\gamma_{S}','\gamma_{S}_err','s/n_{B}','s/n_{B}_err'])
                dict_out.to_csv(path+out_str+'freezout_Npart_ratios.csv', index=False, header=True)
                if(len(Nparts)>1):
                    plot_quant(dict_out,r'$N_{part}$','freezeout parameters',plot_title,path+out_str+'freezeout_Npart_ratios',log=True)

                dict_out = pd.DataFrame(np.concatenate((Nparts,freeze_out_ratios_nS0.reshape((len(list_b),n_freeze_ratios_nS0*2))),axis=1), \
                    columns=['Npart','Npart_err','T_{ch}','T_{ch}_err','\mu_{B}','\mu_{B}_err','\mu_{Q}','\mu_{Q}_err','\mu_{S}','\mu_{S}_err',\
                        '\gamma_{S}','\gamma_{S}_err','s/n_{B}','s/n_{B}_err'])
                dict_out.to_csv(path+out_str+'freezout_Npart_ratios_nS0.csv', index=False, header=True)
                if(len(Nparts)>1):
                    plot_quant(dict_out,r'$N_{part}$','freezeout parameters',plot_title,path+out_str+'freezeout_Npart_ratios_nS0',log=True)
            
            del freeze_out_yields,freeze_out_ratios


        dict_out = pd.DataFrame(np.concatenate((Nparts,mean_pT.reshape((len(list_b),len(particles)*2))),axis=1), columns=['Npart','Npart_err']+label_part)
        dict_out.to_csv(path+out_str+'pT_Npart.csv', index=False, header=True)
        if(len(Nparts)>1):
            plot_quant(dict_out,r'$N_{part}$',r'$\langle p_T \rangle$ [GeV]',plot_title,path+out_str+'pT_Npart')

        del Nparts,dNchdeta,dNchdy,dNdy,mean_pT,dict_out

    ####################################################################
    # quantities as a function of y and eta
    def quant_y():
        """
        Export observables as a function of rapidities y and eta
        dN/dy & dN/deta for charged particles
        dN/dy & dN/deta for each particles
        """
        print("   - observables as a function of y & eta")

        # dN_ch/deta & dN_ch/dy
        dNchdeta = np.zeros((len(list_eta),2))
        dNchdy = np.zeros((len(list_y),2))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for xb in list_b])
        # for each b, add the contribution to spectrum
        for xb in list_b:
            if(pp==True):
                weightb = 1.
            else:
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b

            dNchdeta += return_obs('dNdeta_eta',xb,part='ch',weight=weightb,squared=True)
            dNchdy += return_obs('dNdy_y',xb,part='ch',weight=weightb,squared=True)

        # calculate standard mean error as \sigma = sqrt( \sum_i \sigma(b_i)**2.)
        dNchdeta[:,1] = np.sqrt(dNchdeta[:,1]) 
        dNchdy[:,1] = np.sqrt(dNchdy[:,1]) 

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_eta).T,dNchdeta),axis=1), columns=['eta','dNchdeta','dNchdeta_err'])
        dict_out.to_csv(path+out_str+'dNchdeta_eta.csv', index=False, header=True)
        plot_quant(dict_out,r'$\eta$',f'$dN_{{ch}}/d\eta$',plot_title,path+out_str+'dNchdeta_eta')

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_y).T,dNchdy),axis=1), columns=['y','dNchdy','dNchdy_err'])
        dict_out.to_csv(path+out_str+'dNchdy_y.csv', index=False, header=True)
        plot_quant(dict_out,r'$y$',f'$dN_{{ch}}/dy$',plot_title,path+out_str+'dNchdy_y')

        # dN/dy & dN/deta
        dNdeta = np.zeros((len(list_eta),len(particles),2))
        dNdy = np.zeros((len(list_y),len(particles),2))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for xb in list_b])
        for xb in list_b:
            if(pp==True):
                weightb = 1.
            else:
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b

            for ip,part in enumerate(particles):
                dNdeta[:,ip] += return_obs('dNdeta_eta',xb,part=part,weight=weightb,squared=True)
                dNdy[:,ip] += return_obs('dNdy_y',xb,part=part,weight=weightb,squared=True)

        # calculate standard mean error as \sigma = sqrt( \sum_i \sigma(b_i)**2.)
        dNdeta[:,:,1] = np.sqrt(dNdeta[:,:,1]) 
        dNdy[:,:,1] = np.sqrt(dNdy[:,:,1])

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_eta).T,dNdeta.reshape((len(list_eta),len(particles)*2))),axis=1), columns=['eta']+label_part)
        dict_out.to_csv(path+out_str+'dNdeta_eta.csv', index=False, header=True)
        plot_quant(dict_out,r'$\eta$',f'$dN/d\eta$',plot_title,path+out_str+'dNdeta_eta',log=True)

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_y).T,dNdy.reshape((len(list_y),len(particles)*2))),axis=1), columns=['y']+label_part)
        dict_out.to_csv(path+out_str+'dNdy_y.csv', index=False, header=True)
        plot_quant(dict_out,r'$y$',f'$dN/dy$',plot_title,path+out_str+'dNdy_y',log=True)

        del dNchdeta,dNchdy,dNdeta,dNdy,dict_out

    ####################################################################
    # stopping as a function of y
    def stopping_y(Nparts):
        """
        Export observables as a function of rapidities y 
        dN/dy of net baryons
        stopping power
        """
        print("   - dNdy of net baryons as a function of y")

        # initialize stopping
        delta_y = np.array([inputf['y'],0])
        # dN/dy
        dNdy = np.zeros((len(list_y),2))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for xb in list_b])
        for ib,xb in enumerate(list_b):
            if(pp==True):
                weightb = 1.
            else:
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b

            # calculate net baryon distribution for one b
            dNdyb = return_obs('dNBBBARdy_y',xb,weight=weightb,squared=True)

            # add dNdy for each b
            dNdy += dNdyb

            # stopping (add contribution from this impact parameter)
            delta_y[0] -= sum([dNdyb[iy,0]/Nparts[ib,0]*abs(y)*ybin for iy,y in enumerate(list_y) if ((y<=inputf['y']) and (dNdyb[iy,0]>0))])
            # standard mean error on stopping
            # error in Npart and dNdyb should be taken into account as
            # \sigma**2(dNdyb/Npart) = (dNdyb/Npart)**2*((\sigma(dNdyb)/dNdyb)**2 + (\sigma(Npart)/Npart)**2) 
            dNdyb[:,1] = np.sqrt(dNdyb[:,1]) 
            delta_y[1] = sum([(dNdyb[iy,0]/Nparts[ib,0]*abs(y)*ybin)**2.*((dNdyb[iy,1]/dNdyb[iy,0])**2. + (Nparts[ib,1]/Nparts[ib,0])**2.) \
                for iy,y in enumerate(list_y) if ((y<=inputf['y']) and (dNdyb[iy,0]>0))])

        # calculate standard mean error as \sigma = sqrt( \sum_i \sigma(b_i)**2.)
        dNdy[:,1] = np.sqrt(dNdy[:,1])
        delta_y[1] = np.sqrt(delta_y[1])

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_y).T,dNdy),axis=1), columns=['y','dNdy','dNdy_err'])
        dict_out.to_csv(path+out_str+'dNdyBBAR_y.csv', index=False, header=True)
        plot_quant(dict_out,r'$y$',r'$dN_{B-\bar{B}}/dy$',plot_title,path+out_str+'dNdyBBAR_y')

        dict_out = pd.DataFrame(np.array([[inputf['SRT'],inputf['y'],delta_y[0],delta_y[1],delta_y[0]/inputf['y'],delta_y[1]/inputf['y']]]), \
            columns=['sqrt(s)','y','delta_y','delta_y_err','delta_y/y','delta_y/y_err'])
        dict_out.to_csv(path+out_str+'stopping.csv', index=False, header=True)

        del Nparts,dNdyb,dNdy,delta_y,dict_out

    ####################################################################
    # quantities as a function of pT and mT
    def quant_pT():
        """
        Export observables as a function of pT & mT
        dN/dpT & dN/dmT for charged particles
        dN/dpT & dN/dmT for each particles
        """
        print("   - observables as a function of pT & mT")

        # dN_ch/dpT
        dNchdpT = np.zeros((len(list_pT),2))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for xb in list_b])
        # for each b, add the contribution to spectrum
        for xb in list_b:
            if(pp==True):
                weightb = 1.
            else:
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b

            dNchdpT += return_obs('dNdpT_pT',xb,part='ch',weight=weightb,squared=True)

        # calculate standard mean error as \sigma = sqrt( \sum_i \sigma(b_i)**2.)
        dNchdpT[:,1] = np.sqrt(dNchdpT[:,1])

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_pT).T,dNchdpT),axis=1), columns=['pT','dNchdpT','dNchdpT_err'])
        dict_out.to_csv(path+out_str+'dNchdpT_pT.csv', index=False, header=True)
        plot_quant(dict_out,r'$p_T$ [GeV]',f'$Ed^3N_{{ch}}/d^3p|_{{|y|<{midrapy}}}\ [GeV^{{-2}}]$',plot_title,path+out_str+'dNchdpT_pT',log=True)

        # dN/dpT & dN/dmT
        dNdpT = np.zeros((len(list_pT),len(particles),2))
        dNdmT = np.zeros((len(list_mT),len(particles),2))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for xb in list_b])
        for xb in list_b:
            if(pp==True):
                weightb = 1.
            else:
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b

            for ip,part in enumerate(particles):
                dNdpT[:,ip] += return_obs('dNdpT_pT',xb,part=part,weight=weightb,squared=True)
                dNdmT[:,ip] += return_obs('dNdmT_mT',xb,part=part,weight=weightb,squared=True)

        # calculate standard mean error as \sigma = sqrt( \sum_i \sigma(b_i)**2.)
        dNdpT[:,:,1] = np.sqrt(dNdpT[:,:,1])
        dNdmT[:,:,1] = np.sqrt(dNdmT[:,:,1])

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_pT).T,dNdpT.reshape((len(list_pT),len(particles)*2))),axis=1), columns=['pT']+label_part)
        dict_out.to_csv(path+out_str+'dNdpT_pT.csv', index=False, header=True)
        plot_quant(dict_out,r'$p_T$ [GeV]',f'$Ed^3N/d^3p|_{{|y|<{midrapy}}}\ [GeV^{{-2}}]$',plot_title,path+out_str+'dNdpT_pT',log=True)

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_mT).T,dNdmT.reshape((len(list_mT),len(particles)*2))),axis=1), columns=['mT']+label_part)
        dict_out.to_csv(path+out_str+'dNdmT_mT.csv', index=False, header=True)
        plot_quant(dict_out,r'$m_T-m_0$ [GeV]',f'$Ed^3N/d^3p|_{{|y|<{midrapy}}}\ [GeV^{{-2}}]$',plot_title,path+out_str+'dNdpT_mT',log=True)

        del dNchdpT,dNdpT,dNdmT,dict_out


    # calculate number of participants + error
    Nparts = np.zeros((len(list_b),2))
    for ib,xb in enumerate(list_b):
        Nparts[ib] = return_obs('Npart',xb)

    quant_Npart(Nparts)
    quant_y()
    stopping_y(Nparts)
    quant_pT()

########################################################################
def main():
    # list of particles to ouput
    particles = ['pi+','pi-','K+','K-','p','p~','Lambda','Lambda~','Xi-','Xi~+','Omega-','Omega~+']

    path_input, path_files = detect_files()
    inputf = read_input(path_input)
    dict_obs = read_data_F(path_files,inputf)
    #dict_obs = read_data(path_files,inputf)
    calculate_obs(dict_obs,inputf,particles)
    del particles,path_input,path_files,inputf,dict_obs

main()