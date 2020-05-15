import numpy as np
import os # to get the path of the current directory
import re
import pandas as pd
import argparse
from math import pi
# import from __init__.py
from __init__ import *

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
# path where to look for files
path = args.folder

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
    path_files = []

    # try to find a match with phsd.dat first in the current folder
    for xfile in entries_all:
        if(xfile == 'phsd.dat'):
            path_files = [path+xfile]

    # if it didn't work
    if(len(path_files) == 0):
        # try to find a match with folders as job_#
        for xfile in entries_all:
            match = re.match('job_([0-9]+)', xfile)
            try:
                match = match.group(1)
                path_files += [path+xfile+'/phsd.dat']
            except:
                pass

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
    elif((xBMIN!=None) and (inputf['BMIN'] > xBMIN)):
        xBMIN = inputf['BMIN']
        print('Entered BMIN changed to min value in b')
    if(xBMAX==None):
        xBMAX = inputf['BMAX']
    elif((xBMAX!=None) and (inputf['BMAX'] > xBMAX)):
        xBMAX = inputf['BMAX']
        print('Entered BMAX changed to max value in b')

    # nuclei information
    N1 = nuclei[inputf["MASSTA"]]
    N2 = nuclei[inputf["MASSPR"]]

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

    list_b = np.arange(start=inputf["BMIN"],stop=inputf["BMAX"]+0.0001,step=inputf['DBIMP'])

    # create a list containing name of particles of interest
    name_part = []
    for part in list_part.values():
        name = part[0]
        if(name!='Sigma0' and name!='Sigmabar0'): # count Lambda0 and Sigma0 together
            name_part.append(name)
    name_part += ['ch'] # add an item for charged particles

    # create dictionnary containing information about all particles of interests
    # To get the list containing the rapidity values (y) of a particle for a value of impact parameter:
    # dict_bimp[<'y', 'pT', or 'mT'>(b=<impact parameter>;<particle name>)]
    # Ex: dict_bimp['y(b=2;Omega-)']
    dict_bimp = {}
    dict_events = {}
    for xb in list_b:
        dict_events= {f'Npart(b={xb})': 0., f'Nevents(b={xb})': 0.}
        for part in name_part:
            dict_bimp.update({f'y(b={xb};{part})': [], f'eta(b={xb};{part})': [], f'pT(b={xb};{part})':[], f'mT(b={xb};{part})': []})

    # loop over all phsd.dat files
    i = 0
    for xfile in path_files:
        i += 1
        if(i>2):
            continue
        print(f'Reading {xfile}')
        with open (xfile,"r") as data_file:

            # loop inside files over impact parameter
            for _ in list_b:
                
                # loop inside impact parameters over NUM
                for _ in range(int(inputf["NUM"])):   

                    # header format (first line)
                    header1=data_file.readline()
                    list_header1=header1.split()
                    # N, ISUB, IRUN, BIMP, IBweight
                    N_particles=int(list_header1[0])
                    current_ISUB=int(list_header1[1])
                    current_NUM=int(list_header1[2])
                    current_b=float(list_header1[3])

                    # header format (2nd line)
                    header2=data_file.readline()
                    list_header2=header2.split()
                    # Np, phi2, epsi2, phi3, epsi3, phi4, epsi4, phi5, epsi5
                    Np = float(list_header2[0]) # number of participants

                    if(current_NUM==1):
                        print(f'   - ISUB = {current_ISUB} ; B = {current_b} fm')

                    if(N_particles>0):
                        # count number of events per impact parameter
                        dict_events[f'Nevents(b={current_b})'] += 1.
                        # sum number of participants per impact paramter
                        dict_events[f'Npart(b={current_b})'] += Np

                    for _ in range(N_particles):
                        # read line
                        line = data_file.readline()
                        list_line = line.split()
                        # line format is ID,IDQ,PX,PY,PZ,P0,iHist
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
                        
                        # add info into list about charged particles
                        if(IDQ!=0):
                            dict_bimp[f'y(b={current_b};ch)'].append(y)
                            dict_bimp[f'pT(b={current_b};ch)'].append(pT)
                            dict_bimp[f'eta(b={current_b};ch)'].append(eta)

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
                            name = 'Lambda0'
                        if(name=='Sigmabar0'):
                            name = 'Lambdabar0'

                        mT = np.sqrt(mass**2.+PX**2.+PY**2.) # transverse mass

                        # add particle info into list
                        dict_bimp[f'y(b={current_b};{name})'].append(y)
                        dict_bimp[f'pT(b={current_b};{name})'].append(pT)
                        dict_bimp[f'mT(b={current_b};{name})'].append(mT)
                        dict_bimp[f'eta(b={current_b};{name})'].append(eta)

    # transform each list to numpy array for analysis
    for key in dict_bimp:
        dict_bimp[key] = np.array(dict_bimp[key])

    return dict_events,dict_bimp

########################################################################
def calculate_quant(dict_events,dict_bimp,inputf,particles):

    print('\nCalculating outputs')

    # format for output files
    out_str = inputf['collstring']

    # list of impact parameters
    list_b = np.arange(start=inputf["BMIN"],stop=inputf["BMAX"]+0.0001,step=inputf['DBIMP'])
    # which b to select for output?
    select_b = (xBMIN <= list_b) & (list_b <= xBMAX)

    # nuclei information
    N1 = nuclei[inputf["MASSTA"]]
    N2 = nuclei[inputf["MASSPR"]]
    plot_title = f"{N1}+{N2} at $\sqrt{{s_{{NN}}}}$ = {inputf['SRT']:4.1f} GeV"

    ####################################################################
    # quantities as a function of Npart
    def quant_Npart():
        """
        Export observables as a function of Npart
        dN_ch/deta & dN_ch/dy for charged particles
        dN/dy & <pT> for each particles
        """
        print("   - observables as a function of Npart")
        
        # calculate number of participants
        Nparts = np.zeros_like(list_b)
        for ib,xb in enumerate(list_b):
            Nparts[ib] =  dict_events[f'Npart(b={xb})']/dict_events[f'Nevents(b={xb})']

        # select particles in abs(y) < midrapy or abs(eta) < midrapeta
        deta = 2.*midrapeta
        dy = 2.*midrapy

        # dN_ch/deta & dN_ch/dy
        dNchdeta = np.zeros(len(list_b))
        dNchdy = np.zeros(len(list_b))
        for ib,xb in enumerate(list_b):
            
            mideta = abs(dict_bimp[f'eta(b={xb};ch)']) < midrapeta
            dNchdeta[ib] = len(dict_bimp[f'eta(b={xb};ch)'][mideta])/deta/dict_events[f'Nevents(b={xb})']

            midrap = abs(dict_bimp[f'y(b={xb};ch)']) < midrapy
            dNchdy[ib] = len(dict_bimp[f'y(b={xb};ch)'][midrap])/dy/dict_events[f'Nevents(b={xb})']

        dict_out = pd.DataFrame(zip(Nparts[select_b],dNchdeta[select_b]), columns=['Npart','dNchdeta'])
        dict_out.to_csv(path+out_str+'dNchdeta_Npart.csv', index=False, header=True)

        dict_out = pd.DataFrame(zip(Nparts[select_b],dNchdy[select_b]), columns=['Npart','dNchdy'])
        dict_out.to_csv(path+out_str+'dNchdy_Npart.csv', index=False, header=True)

        if(len(Nparts)>1):
            plot_quant(Nparts[select_b],dNchdeta[select_b],r'$N_{part}$',f'$dN_{{ch}}/d\eta|_{{|\eta|<{midrapeta}}}$',plot_title,path+out_str+'dNchdeta_Npart')
            plot_quant(Nparts[select_b],dNchdy[select_b],r'$N_{part}$',f'$dN_{{ch}}/dy|_{{|y|<{midrapy}}}$',plot_title,path+out_str+'dNchdy_Npart')

        # dN/dy & <pT>
        dNdy = np.zeros((len(particles),len(list_b)))
        mean_pT = np.zeros((len(particles),len(list_b)))
        for ib,xb in enumerate(list_b):
            for ip,part in enumerate(particles):
                midrap = abs(dict_bimp[f'y(b={xb};{part})']) < midrapy
                dNdy[ip,ib] = len(dict_bimp[f'y(b={xb};{part})'][midrap])/dy/dict_events[f'Nevents(b={xb})']
                mean_pT[ip,ib] = np.mean(dict_bimp[f'pT(b={xb};{part})'][midrap])

        dict_out = pd.DataFrame([np.concatenate(([Npart],dNdy[:,ib])) for ib,Npart in enumerate(Nparts) if select_b[ib]], columns=['Npart']+particles)
        dict_out.to_csv(path+out_str+'dNdy_Npart.csv', index=False, header=True)

        dict_out = pd.DataFrame([np.concatenate(([Npart],mean_pT[:,ib])) for ib,Npart in enumerate(Nparts) if select_b[ib]], columns=['Npart']+particles)
        dict_out.to_csv(path+out_str+'pT_Npart.csv', index=False, header=True)

        if(len(Nparts)>1):
            plot_quant(Nparts[select_b],dNdy[:,select_b],r'$N_{part}$',r'$dN/dy$',plot_title,path+out_str+'dNdy_Npart',partplot=particles,log=True)
            plot_quant(Nparts[select_b],mean_pT[:,select_b],r'$N_{part}$',r'$\langle p_T \rangle$ [GeV]',plot_title,path+out_str+'pT_Npart',partplot=particles)

    ####################################################################
    # quantities as a function of y and eta
    def quant_y():
        """
        Export observables as a function of rapidities y and eta
        dN/dy & dN/deta for charged particles
        dN/dy & dN/deta for each particles
        """
        print("   - observables as a function of y & eta")

        # select particles in abs(y) < ylim or abs(eta) < etalim
        etalim = int(round(inputf['y'])+1)
        ylim = int(round(inputf['y'])+1)
        # list of y and eta for each bin
        list_eta = np.arange(start=-etalim,stop=etalim+0.0001,step=etabin)
        list_y = np.arange(start=-ylim,stop=ylim+0.0001,step=ybin)

        # dN_ch/deta & dN_ch/dy
        dNchdeta = np.zeros_like(list_eta)
        dNchdy = np.zeros_like(list_y)
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for ib,xb in enumerate(list_b) if select_b[ib]])
        # for each b, add the contribution to spectrum
        for ib,xb in enumerate(list_b):
            if(select_b[ib]):
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b
                dNchdeta += weightb*np.histogram(dict_bimp[f'eta(b={xb};ch)'],bins=len(list_eta),range=(-etalim-etabin/2.,etalim+etabin/2.))[0]/etabin/dict_events[f'Nevents(b={xb})']
                dNchdy += weightb*np.histogram(dict_bimp[f'y(b={xb};ch)'],bins=len(list_y),range=(-ylim-ybin/2.,ylim+ybin/2.))[0]/ybin/dict_events[f'Nevents(b={xb})']

        dict_out = pd.DataFrame(zip(list_eta,dNchdeta), columns=['eta','dNchdeta'])
        dict_out.to_csv(path+out_str+'dNchdeta_eta.csv', index=False, header=True)

        dict_out = pd.DataFrame(zip(list_y,dNchdy), columns=['y','dNchdy'])
        dict_out.to_csv(path+out_str+'dNchdy_y.csv', index=False, header=True)

        plot_quant(list_eta,dNchdeta,r'$\eta$',f'$dN_{{ch}}/d\eta$',plot_title,path+out_str+'dNchdeta_eta')
        plot_quant(list_y,dNchdy,r'$y$',f'$dN_{{ch}}/dy$',plot_title,path+out_str+'dNchdy_y')

        # dN/dy & dN/deta
        dNdeta = np.zeros((len(particles),len(list_eta)))
        dNdy = np.zeros((len(particles),len(list_y)))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for ib,xb in enumerate(list_b) if select_b[ib]])
        for ib,xb in enumerate(list_b):
            if(select_b[ib]):
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b
                for ip,part in enumerate(particles):
                    dNdeta[ip] += weightb*np.histogram(dict_bimp[f'eta(b={xb};{part})'],bins=len(list_eta),range=(-etalim-etabin/2.,etalim+etabin/2.))[0]/etabin/dict_events[f'Nevents(b={xb})']
                    dNdy[ip] += weightb*np.histogram(dict_bimp[f'y(b={xb};{part})'],bins=len(list_y),range=(-ylim-ybin/2.,ylim+ybin/2.))[0]/ybin/dict_events[f'Nevents(b={xb})']

        dict_out = pd.DataFrame([np.concatenate(([eta],dNdeta[:,ieta])) for ieta,eta in enumerate(list_eta)], columns=['eta']+particles)
        dict_out.to_csv(path+out_str+'dNdeta_eta.csv', index=False, header=True)

        dict_out = pd.DataFrame([np.concatenate(([y],dNdy[:,iy])) for iy,y in enumerate(list_y)], columns=['y']+particles)
        dict_out.to_csv(path+out_str+'dNdy_y.csv', index=False, header=True)

        plot_quant(list_eta,dNdeta,r'$\eta$',f'$dN/d\eta$',plot_title,path+out_str+'dNdeta_eta',partplot=particles,log=True)
        plot_quant(list_y,dNdy,r'$y$',f'$dN/dy$',plot_title,path+out_str+'dNdy_y',partplot=particles,log=True)

    ####################################################################
    # stopping as a function of y
    def stopping_y():
        """
        Export observables as a function of rapidities y 
        dN/dy of net baryons
        stopping power
        """
        print("   - dNdy of net baryons as a function of y")

        # select particles in abs(y) < ylim
        ylim = int(round(inputf['y'])+1)
        # list of y and eta for each bin
        list_y = np.arange(start=-ylim,stop=ylim+0.0001,step=ybin)

        # calculate number of participants
        Nparts = np.zeros_like(list_b)
        for ib,xb in enumerate(list_b):
            Nparts[ib] =  dict_events[f'Npart(b={xb})']/dict_events[f'Nevents(b={xb})']

        # initialize stopping
        delta_y = inputf['y']
        # dN/dy
        dNdy = np.zeros(len(list_y))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for ib,xb in enumerate(list_b) if select_b[ib]])
        for ib,xb in enumerate(list_b):
            if(select_b[ib]):
                # initialize dNdy for each b
                dNdyb = np.zeros(len(list_y))
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b
                for part in ['p','n0','Lambda0','Sigma-','Sigma+','Xi0','Xi-','Omega-']:
                    dNdyb += weightb*np.histogram(dict_bimp[f'y(b={xb};{part})'],bins=len(list_y),range=(-ylim-ybin/2.,ylim+ybin/2.))[0]/ybin/dict_events[f'Nevents(b={xb})']
                for part in ['pbar','nbar0','Lambdabar0','Sigmabar+','Sigmabar-','Xibar0','Xibar+','Omegabar+']:
                    dNdyb -= weightb*np.histogram(dict_bimp[f'y(b={xb};{part})'],bins=len(list_y),range=(-ylim-ybin/2.,ylim+ybin/2.))[0]/ybin/dict_events[f'Nevents(b={xb})']

                dNdy +=  dNdyb # add dNdy for each b
                # stopping
                delta_y -= 1./Nparts[ib]*sum([dNdyb[iy]*abs(y)*ybin for iy,y in enumerate(list_y) if y <= inputf['y']])

        dict_out = pd.DataFrame(zip(list_y,dNdy), columns=['y','dNdy'])
        dict_out.to_csv(path+out_str+'dNdyBBAR_y.csv', index=False, header=True)

        dict_out = pd.DataFrame(np.array([[inputf['SRT'],inputf['y'],delta_y,delta_y/inputf['y']]]), columns=['sqrt(s)','y','delta_y','delta_y/y'])
        dict_out.to_csv(path+out_str+'stopping.csv', index=False, header=True)

        plot_quant(list_y,dNdy,r'$y$',r'$dN_{B-\bar{B}}/dy$',plot_title,path+out_str+'dNdyBBAR_y')

    ####################################################################
    # quantities as a function of pT and mT
    def quant_pT():
        """
        Export observables as a function of pT & mT
        dN/dpT & dN/dmT for charged particles
        dN/dpT & dN/dmT for each particles
        """
        print("   - observables as a function of pT & mT")

        # gap at midrapidity
        dy = 2.*midrapy
        # max values of pT, mT
        pTmax = 4.
        mTmax = 4.
        # list of y and eta for each bin
        list_mT = np.arange(start=pTbin/2.,stop=pTmax-pTbin/2.+0.0001,step=pTbin)
        list_pT = np.arange(start=mTbin/2.,stop=mTmax-mTbin/2.+0.0001,step=mTbin)

        # dN_ch/dpT & dN_ch/dmT
        dNchdpT = np.zeros_like(list_pT)
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for ib,xb in enumerate(list_b) if select_b[ib]])
        # for each b, add the contribution to spectrum
        for ib,xb in enumerate(list_b):
            if(select_b[ib]):
                # select midrapidity
                midrap = abs(dict_bimp[f'y(b={xb};ch)']) < midrapy
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b
                dNchdpT += weightb*np.histogram(dict_bimp[f'pT(b={xb};ch)'][midrap],bins=len(list_pT),range=(0,pTmax),weights=np.array([1./(2.*pi*pT*pTbin*dy)/dict_events[f'Nevents(b={xb})'] for pT in dict_bimp[f'pT(b={xb};ch)'][midrap]]))[0]

        dict_out = pd.DataFrame(zip(list_pT,dNchdpT), columns=['pT','dNchdpT'])
        dict_out.to_csv(path+out_str+'dNchdpT_pT.csv', index=False, header=True)

        plot_quant(list_pT,dNchdpT,r'$p_T$ [GeV]',f'$Ed^3N_{{ch}}/d^3p|_{{|y|<{midrapy}}}\ [GeV^{{-2}}]$',plot_title,path+out_str+'dNchdpT_pT',log=True)

        # dN/dpT & dN/dmT
        dNdpT = np.zeros((len(particles),len(list_pT)))
        dNdmT = np.zeros((len(particles),len(list_mT)))
        # normalization for sum over b
        Anormb = sum([2.*pi*xb*inputf['DBIMP'] for ib,xb in enumerate(list_b) if select_b[ib]])
        for ib,xb in enumerate(list_b):
            if(select_b[ib]):
                weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b
                for ip,part in enumerate(particles):
                    # select midrapidity
                    midrap = abs(dict_bimp[f'y(b={xb};{part})']) < midrapy
                    weightb = 2.*pi*xb*inputf['DBIMP']/Anormb # weight for this b
                    dNdpT[ip] += weightb*np.histogram(dict_bimp[f'pT(b={xb};{part})'][midrap],bins=len(list_pT),range=(0,pTmax),weights=np.array([1./(2.*pi*pT*pTbin*dy)/dict_events[f'Nevents(b={xb})'] for pT in dict_bimp[f'pT(b={xb};{part})'][midrap]]))[0]
                    dNdmT[ip] += weightb*np.histogram(dict_bimp[f'mT(b={xb};{part})'][midrap],bins=len(list_mT),range=(0,mTmax),weights=np.array([1./(2.*pi*mT*mTbin*dy)/dict_events[f'Nevents(b={xb})'] for mT in dict_bimp[f'mT(b={xb};{part})'][midrap]]))[0]

        dict_out = pd.DataFrame([np.concatenate(([pT],dNdpT[:,ipT])) for ipT,pT in enumerate(list_pT)], columns=['pT']+particles)
        dict_out.to_csv(path+out_str+'dNdpT_pT.csv', index=False, header=True)

        dict_out = pd.DataFrame([np.concatenate(([mT],dNdmT[:,imT])) for imT,mT in enumerate(list_mT)], columns=['mT']+particles)
        dict_out.to_csv(path+out_str+'dNdmT_mT.csv', index=False, header=True)

        plot_quant(list_pT,dNdpT,r'$p_T$ [GeV]',f'$Ed^3N/d^3p|_{{|y|<{midrapy}}}\ [GeV^{{-2}}]$',plot_title,path+out_str+'dNdpT_pT',partplot=particles,log=True)
        plot_quant(list_mT,dNdmT,r'$m_T$ [GeV]',f'$Ed^3N/d^3p|_{{|y|<{midrapy}}}\ [GeV^{{-2}}]$',plot_title,path+out_str+'dNdpT_mT',partplot=particles,log=True)

    quant_Npart()
    quant_y()
    stopping_y()
    quant_pT()
    
########################################################################
def main():
    # list of particles to ouput
    particles = ['pi+','pi-','K+','K-','p','pbar','Lambda0','Lambdabar0','Xi-','Xibar+','Omega-','Omegabar+']

    path_input, path_files = detect_files()
    inputf = read_input(path_input)
    dict_events,dict_bimp = read_data(path_files,inputf)
    calculate_quant(dict_events,dict_bimp,inputf,particles)

########################################################################
main()