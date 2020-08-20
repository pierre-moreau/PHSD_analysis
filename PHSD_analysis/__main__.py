import os # to get the path of the current directory
import re
import pandas as pd
import argparse
from math import pi
from scipy import stats
from . import *
from EoS_HRG.test.plot_HRG import plot_freezeout

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
print(f'Current directory: {dir_path}')
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
            inputf.update({list_line[1].replace(':',''): float(list_line[0].replace(',',''))})

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
    SRT=np.sqrt(2.*DMN*(TLAB+2.*DMN)) # collisional energy in the CM
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
    global etalim, ylim
    etalim = int(round(inputf['y'])+1)
    ylim = int(round(inputf['y'])+1)

    # list of y and eta for each bin
    global list_eta,list_y,edges_eta,edges_y
    list_eta = np.arange(start=-etalim,stop=etalim+0.0001,step=etabin)
    list_y = np.arange(start=-ylim,stop=ylim+0.0001,step=ybin)  
    edges_eta = np.arange(start=-etalim-etabin/2.,stop=etalim+etabin/2.+0.0001,step=etabin)
    edges_y = np.arange(start=-ylim-ybin/2.,stop=ylim+ybin/2.+0.0001,step=ybin)

    # max values of pT, mT
    global pTmax,mTmax
    pTmax = 4.
    mTmax = 4.
    # list of mT and pT for each bin
    global list_pT,list_mT,edges_pT,edges_mT
    list_mT = np.arange(start=pTbin/2.,stop=pTmax-pTbin/2.+0.0001,step=pTbin)
    list_pT = np.arange(start=mTbin/2.,stop=mTmax-mTbin/2.+0.0001,step=mTbin)
    edges_pT = np.arange(start=0.,stop=pTmax+0.0001,step=pTbin)
    edges_mT = np.arange(start=0.,stop=mTmax+0.0001,step=mTbin)

    # display information about the collision
    print(f'{N1}+{N2} at $\sqrt{{s_{{NN}}}}$ = {SRT:4.1f} GeV -- TLAB = {TLAB} GeV -- yproj = {yproj}')
    print(f'BMIN = {xBMIN} fm -- BMAX = {xBMAX} fm -- DELTAB = {inputf["DBIMP"]} fm')
    print(f'NUM = {inputf["NUM"]} \n')

    return inputf

########################################################################
def read_data(path_files,inputf):
    """
    read all the phsd.dat files specified in path_files
    """

    list_b_all = np.arange(start=inputf["BMIN"],stop=inputf["BMAX"]+0.0001,step=inputf['DBIMP'])

    ########################################################################
    def histogram(list,edges,weights=1):
        """
        compute histogram, to replace np.histogram
        histo = np.histogram(list,bins=nbin,range=(edges[0],edges[-1]),weights=weights)[0]
        """
        nbin = len(edges)-1
        histo = np.zeros(nbin)
        if(isinstance(weights,float)):
            weights = weights*np.ones(len(list))

        for iitem,item in enumerate(list):
            for i in range(nbin):
                if(edges[i]<item and item<edges[i+1]):
                    histo[i] += weights[iitem]
                    break
        return histo

    # create a list containing name of particles of interest
    name_part = []
    for part in list_part.values():
        name = part[0]
        if(name!='Sigma0' and name!='Sigma~0'): # count Lambda0 and Sigma0 together
            name_part.append(name)
    name_part += ['ch'] # add an item for charged particles

    # create dictionnary containing information about all observables of interests
    # dict_bimp[<Observable>(b=<impact parameter>;Nev=<event number>;<particle name>)]
    # Ex: dict_bimp['dNdpT(b=2;Nev=22;Omega-)']
    dict_bimp = {}
    dict_events = {}
    # loop over all phsd.dat files
    for xfile in path_files:
        print(f'Reading {xfile}')
        with open (xfile,"r") as data_file:

            # loop inside files over impact parameter
            for xb in list_b_all:
                # skip reading files if B > xBMAX
                if(xb>xBMAX):
                    break
                # loop inside impact parameters over ISUB
                for _ in range(int(inputf["ISUBS"])):
                    # loop inside ISUB over NUM
                    for _ in range(int(inputf["NUM"])):

                        try:
                            # header format (first line)
                            header1=data_file.readline()
                            list_header1=header1.split()
                            # N, ISUB, IRUN, BIMP, IBweight
                            N_particles=int(list_header1[0])
                            current_ISUB=int(list_header1[1])
                            current_NUM=int(list_header1[2])
                            current_b=float(list_header1[3])
                        except:
                            continue

                        # create new entry for each b in dict if it doesn't already exist
                        try:
                            dict_events[f'Nevents(b={current_b})']
                        except:
                            dict_events.update({f'Nevents(b={current_b})': 0})

                        # header format (2nd line)
                        header2=data_file.readline()
                        list_header2=header2.split()
                        # Np, phi2, epsi2, phi3, epsi3, phi4, epsi4, phi5, epsi5
                        Np = float(list_header2[0]) # number of participants

                        if(current_NUM==1):
                            if(current_b<xBMIN):
                                print(f'   - ISUB = {current_ISUB} ; B = {current_b} fm ; (not counted)')
                            else:
                                print(f'   - ISUB = {current_ISUB} ; B = {current_b} fm')
                    
                        # only count inelastic events
                        if(N_particles>(inputf["MASSTA"]+inputf["MASSPR"])):
                            count_event = True

                            # count number of events per impact parameter
                            dict_events[f'Nevents(b={current_b})'] += 1

                            # event ID number
                            Nev = dict_events[f'Nevents(b={current_b})']-1
                            # save Npart for each event
                            dict_events.update({f'Npart(b={current_b};Nev={Nev})': Np})

                            # initialize dict containing particle info for this event
                            dict_particles = {}
                            for part in name_part:
                                dict_particles.update({f'y({part})': [],\
                                    f'eta({part})': [],\
                                    f'pT({part})':[],\
                                    f'mT({part})': [],\
                                    f'phi({part})': []})
                        else:
                            count_event = False

                        for _ in range(N_particles):
                            # read line
                            line = data_file.readline()
                            # only count inelastic events, read and skip
                            if(not(count_event)):
                                continue
                            # if B < BMIN, don't fill arrays
                            if(current_b<xBMIN):
                                continue
                            # line format is ID,IDQ,PX,PY,PZ,P0,iHist
                            list_line = line.split()
                            ID = int(list_line[0])
                            IDQ = int(list_line[1])
                            PX = float(list_line[2])
                            PY = float(list_line[3])
                            PZ = float(list_line[4])
                            P0 = float(list_line[5])
                            iHist = int(list_line[6])

                            y = 0.5*np.log((P0+PZ)/(P0-PZ)) # rapidity
                            PP = np.sqrt(PX**2.+PY**2.+PZ**2.) # momentum
                            eta = 0.5*np.log((PP+PZ)/(PP-PZ)) # pseudo-rapidity
                            pT = np.sqrt(PX**2.+PY**2.) # transverse momentum
                            phi = np.arctan2(PY,PX) # angle phi
                        
                            # add info into list about charged particles
                            if(IDQ!=0):
                                dict_particles[f'y(ch)'].append(y)
                                dict_particles[f'pT(ch)'].append(pT)
                                dict_particles[f'eta(ch)'].append(eta)
                                dict_particles[f'phi(ch)'].append(phi)

                            # if ID is not the list, then skip this particle 
                            # and continue with the loop
                            try:
                                name,mass,_ = list_part[ID]
                            except:
                                continue

                            # don't count spectator neutrons and protons
                            if(name=='p' or name=='n0'):
                                # if |y| > yproj-0.5, don't count
                                # 0.5 is arbitrary, but works fine
                                if((iHist==-1) and (abs(y)>(inputf['y']-0.5))):
                                    continue

                            # count Lambda0 and Sigma0 together
                            if(name=='Sigma0'):
                                name = 'Lambda'
                            if(name=='Sigma~0'):
                                name = 'Lambda~'

                            mT = np.sqrt(mass**2.+PX**2.+PY**2.) # transverse mass

                            # add particle info into list
                            dict_particles[f'y({name})'].append(y)
                            dict_particles[f'pT({name})'].append(pT)
                            dict_particles[f'mT({name})'].append(mT)
                            dict_particles[f'eta({name})'].append(eta)
                            dict_particles[f'phi({name})'].append(phi)

                        #############################################################################
                        # calculate observables for this event and update corresponding dictionnary #
                        #############################################################################
                        if(count_event):
                            # transform each list to numpy array for analysis
                            for key in dict_particles:
                                dict_particles[key] = np.array(dict_particles[key])

                            # loop over all particles
                            for part in name_part:
                                # midrapidity cuts
                                cond_midrapeta = abs(dict_particles[f'eta({part})']) < midrapeta
                                cond_midrapy = abs(dict_particles[f'y({part})']) < midrapy

                                # dNdeta & dNdy
                                dict_bimp[f'dNdeta(b={current_b};Nev={Nev};{part})'] = len(dict_particles[f'eta({part})'][cond_midrapeta])/deta
                                dict_bimp[f'dNdy(b={current_b};Nev={Nev};{part})'] = len(dict_particles[f'y({part})'][cond_midrapy])/dy

                                # mean pT
                                if(len(dict_particles[f'pT({part})'][cond_midrapy])>0):
                                    dict_bimp[f'mean_pT(b={xb};Nev={Nev};{part})'] = np.mean(dict_particles[f'pT({part})'][cond_midrapy])
                                else:
                                    dict_bimp[f'mean_pT(b={xb};Nev={Nev};{part})'] = 0.

                                # dNdeta_eta & dNdy_y
                                dict_bimp[f'dNdeta_eta(b={current_b};Nev={Nev};{part})'] = \
                                    histogram(dict_particles[f'eta({part})'],edges_eta,weights=1./etabin)
                                dict_bimp[f'dNdy_y(b={current_b};Nev={Nev};{part})'] = \
                                    histogram(dict_particles[f'y({part})'],edges_y,weights=1./ybin)

                                # dNdpT_pT & dNdmT_mT
                                dict_bimp[f'dNdpT_pT(b={current_b};Nev={Nev};{part})'] = histogram(dict_particles[f'pT({part})'][cond_midrapy],edges_pT,\
                                        weights=np.array([1./(2.*pi*pT*pTbin*dy) for pT in dict_particles[f'pT({part})'][cond_midrapy]]))
                                if(part!='ch'):
                                    dict_bimp[f'dNdmT_mT(b={current_b};Nev={Nev};{part})'] = histogram(dict_particles[f'mT({part})'][cond_midrapy]-from_part(part)[1],edges_pT,\
                                        weights=np.array([1./(2.*pi*mT*mTbin*dy) for mT in dict_particles[f'mT({part})'][cond_midrapy]]))

                            # calculate dN/dy of baryons minus antibaryons
                            list_B = ['p','n','Lambda','Sigma-','Sigma+','Xi0','Xi-','Omega-']
                            list_antiB = ['p~','n~','Lambda~','Sigma~+','Sigma~-','Xi~0','Xi~+','Omega~+']
                            dict_bimp[f'dNBBBARdy_y(b={current_b};Nev={Nev})'] = \
                                np.sum(np.array([histogram(dict_particles[f'y({part})'],edges_y,weights=1./ybin) for part in list_B])\
                                -np.array([histogram(dict_particles[f'y({part})'],edges_y,weights=1./ybin) for part in list_antiB]),axis=0)

                            del dict_particles,cond_midrapeta,cond_midrapy

    return dict_events,dict_bimp

########################################################################
def calculate_obs(dict_events,dict_bimp,inputf,particles):

    print('\nCalculating outputs')

    # format for output files
    out_str = inputf['collstring']

    # list of impact parameters
    # which b to select for output?
    list_b_all = np.arange(start=inputf["BMIN"],stop=inputf["BMAX"]+0.0001,step=inputf['DBIMP'])
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
    def return_mean(gen,squared=False):
        """
        return mean and standard mean error from generator containing
        quantity for each event
        The confidence interval is given as 3.3 (\approx t-value for 99.9% confidence interval) times the standard mean error
        """
        xlist = list(gen)
        mean = np.mean(xlist,axis=0)
        t_value = 3.3
        sem = t_value*stats.sem(xlist,axis=0)
        # if mean is a float, just return floats
        if(isinstance(mean, float)):
            return mean,sem
        # if it's an array, return zipped array of mean and sem or sem**2.
        else:
            if(squared):
                return np.array(list(zip(mean,sem**2.)))
            else:
                return np.array(list(zip(mean,sem)))

    ####################################################################
    # quantities as a function of Npart
    def quant_Npart():
        """
        Export observables as a function of Npart
        dN_ch/deta & dN_ch/dy for charged particles
        dN/dy & <pT> for each particles
        Chemical freeze-out parameters as a function of Npart
        """
        print("   - observables as a function of Npart")
        
        # calculate number of participants + error
        Nparts = np.zeros((len(list_b),2))
        for ib,xb in enumerate(list_b):
            Nevtot = dict_events[f'Nevents(b={xb})'] # total number of events per b
            gen_Npart = np.array([dict_events[f'Npart(b={xb};Nev={Nev})'] for Nev in range(Nevtot)])
            Nparts[ib] = return_mean(gen_Npart)

        dict_out = pd.DataFrame(np.concatenate((np.atleast_2d(list_b).T,Nparts),axis=1), columns=['b','Npart','Npart_err'])
        dict_out.to_csv(path+out_str+'Npart_b.csv', index=False, header=True)
        if(len(Nparts)>1):
            plot_quant(dict_out,r'$b [fm]$',r'$N_{part}$',plot_title,path+out_str+'Npart_b')

        # dN_ch/deta & dN_ch/dy + error
        dNchdeta = np.zeros((len(list_b),2))
        dNchdy = np.zeros((len(list_b),2))
        for ib,xb in enumerate(list_b):
            Nevtot = dict_events[f'Nevents(b={xb})'] # total number of events per b

            # construct generator containing dNdchdeta at midrapidity for each event
            gen_dNchdeta = (dict_bimp[f'dNdeta(b={xb};Nev={Nev};ch)'] for Nev in range(Nevtot))
            # calculate average over all events
            dNchdeta[ib] = return_mean(gen_dNchdeta)

            # construct generator containing dNdchdy at midrapidity for each event
            gen_dNchdy = (dict_bimp[f'dNdy(b={xb};Nev={Nev};ch)'] for Nev in range(Nevtot))
            # calculate average over all events
            dNchdy[ib] = return_mean(gen_dNchdy)

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
            Nevtot = dict_events[f'Nevents(b={xb})'] # total number of events per b
            for ip,part in enumerate(particles):
                # construct generator containing dNdy at midrapidity for all events for the designated particle
                gen_dNdy = (dict_bimp[f'dNdy(b={xb};Nev={Nev};{part})'] for Nev in range(Nevtot))
                # calculate average over all events
                dNdy[ib,ip] = return_mean(gen_dNdy)
                # construct generator containing <pT> at midrapidity for all events for the designated particle
                gen_mean_pT = (dict_bimp[f'mean_pT(b={xb};Nev={Nev};{part})'] for Nev in range(Nevtot))
                # calculate average over all events
                mean_pT[ib,ip] = return_mean(gen_mean_pT)

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
            weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b
            Nevtot = dict_events[f'Nevents(b={xb})'] # total number of events per b

            # construct list containing the histograms of dNch/deta(eta) for each event
            gen_dNchdeta = (weightb*dict_bimp[f'dNdeta_eta(b={xb};Nev={Nev};ch)'] for Nev in range(Nevtot))
            # construct list containing the histograms of dNch/dy(y) for each event
            gen_dNchdy = (weightb*dict_bimp[f'dNdy_y(b={xb};Nev={Nev};ch)'] for Nev in range(Nevtot))

            # calculate average over all events
            dNchdeta += return_mean(gen_dNchdeta,squared=True)
            dNchdy += return_mean(gen_dNchdy,squared=True)

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
            Nevtot = dict_events[f'Nevents(b={xb})'] # total number of events per b
            for ip,part in enumerate(particles):
                # construct list containing the histograms of dN/deta(eta) for each event for the designated particle
                gen_dNdeta = np.array([weightb*dict_bimp[f'dNdeta_eta(b={xb};Nev={Nev};{part})'] for Nev in range(Nevtot)])
                # construct list containing the histograms of dN/dy(y) for each event for the designated particle
                gen_dNdy = np.array([weightb*dict_bimp[f'dNdy_y(b={xb};Nev={Nev};{part})'] for Nev in range(Nevtot)])
                
                # calculate average over all events
                dNdeta[:,ip] += return_mean(gen_dNdeta,squared=True)
                dNdy[:,ip] += return_mean(gen_dNdy,squared=True)

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
    def stopping_y():
        """
        Export observables as a function of rapidities y 
        dN/dy of net baryons
        stopping power
        """
        print("   - dNdy of net baryons as a function of y")

        # calculate number of participants + error
        Nparts = np.zeros((len(list_b),2))
        for ib,xb in enumerate(list_b):
            Nevtot = dict_events[f'Nevents(b={xb})'] # total number of events per b
            gen_Npart = np.array([dict_events[f'Npart(b={xb};Nev={Nev})'] for Nev in range(Nevtot)])
            Nparts[ib] = return_mean(gen_Npart)

        # initialize stopping
        delta_y = np.array([inputf['y'],0])
        # dN/dy
        dNdy = np.zeros((len(list_y),2))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for xb in list_b])
        for xb in list_b:
            # initialize dNdy for each b
            dNdyb = np.zeros((len(list_y),2))
            if(pp==True):
                weightb = 1.
            else:
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b
            Nevtot = dict_events[f'Nevents(b={xb})'] # total number of events per b

            # for each event, sum histogram for dN/dy(y) over baryons/antibaryons
            gen_dNdyb = (weightb*dict_bimp[f'dNBBBARdy_y(b={xb};Nev={Nev})'] for Nev in range(Nevtot))
            # calculate average over all events of dN/dy(y) baryons minus antibaryons
            dNdyb = return_mean(gen_dNdyb,squared=True)

            # add dNdy for each b
            dNdy +=  dNdyb

            dNdyb[:,1] = np.sqrt(dNdyb[:,1]) 
            # stopping (add contribution from this impact parameter)
            delta_y[0] -= sum([dNdyb[iy,0]/Nparts[ib,0]*abs(y)*ybin for iy,y in enumerate(list_y) if ((y<=inputf['y']) and (dNdyb[iy,0]>0))])
            # standard mean error on stopping
            # error in Npart and dNdyb should be taken into account as
            # \sigma**2(dNdyb/Npart) = (dNdyb/Npart)**2*((\sigma(dNdyb)/dNdyb)**2 + (\sigma(Npart)/Npart)**2) 
            delta_y[1] = sum([(dNdyb[iy,0]/Nparts[ib,0])**2.*((dNdyb[iy,1]/dNdyb[iy,0])**2. + (Nparts[ib,1]/Nparts[ib,0])**2.)*abs(y)*ybin \
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

        # dN_ch/dpT & dN_ch/dmT
        dNchdpT = np.zeros((len(list_pT),2))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for xb in list_b])
        # for each b, add the contribution to spectrum
        for xb in list_b:
            if(pp==True):
                weightb = 1.
            else:
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b
            Nevtot = dict_events[f'Nevents(b={xb})'] # total number of events per b
            # construct array containg the histograms dNch/dpT for each event
            gen_dNchdpT = (weightb*dict_bimp[f'dNdpT_pT(b={xb};Nev={Nev};ch)'] for Nev in range(Nevtot))
            # calculate average over all events
            dNchdpT += return_mean(gen_dNchdpT,squared=True)

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
                gen_dNdpT = (weightb*dict_bimp[f'dNdpT_pT(b={xb};Nev={Nev};{part})'] for Nev in range(Nevtot))
                gen_dNdmT = (weightb*dict_bimp[f'dNdmT_mT(b={xb};Nev={Nev};{part})'] for Nev in range(Nevtot))

                # calculate average over all events
                dNdpT[:,ip] += return_mean(gen_dNdpT,squared=True)
                dNdmT[:,ip] += return_mean(gen_dNdmT,squared=True)

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

    quant_Npart()
    quant_y()
    stopping_y()
    quant_pT()

########################################################################
def main():
    # list of particles to ouput
    particles = ['pi+','pi-','K+','K-','p','p~','Lambda','Lambda~','Xi-','Xi~+','Omega-','Omega~+']

    path_input, path_files = detect_files()
    inputf = read_input(path_input)
    dict_events,dict_bimp = read_data(path_files,inputf)
    calculate_obs(dict_events,dict_bimp,inputf,particles)
    del particles,path_input,path_files,inputf,dict_events,dict_bimp

main()