#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import math 
import astropy
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import scipy.stats as sps

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import statsmodels.api as sm


# In[2]:


#SciServer Modules
import SciServer
from SciServer import CasJobs


# In[3]:


Candidates = """select drp.MaNGAid,
                drp.nsa_elpetro_mass as GalMass,
                drp.z,
                drp.nsa_elpetro_absmag_u as uMag, 
                drp.nsa_elpetro_absmag_r as rMag,
                firefly.LW_AGE_1RE as LWStellarAge
                
                
                from mangaDRPall as drp
                
                JOIN mangaDAPall as dap on dap.mangaid = drp.mangaid

                JOIN mangaFirefly_mastar as firefly on drp.mangaid = firefly.MANGAID
                                
                WHERE (drp.z < 0.15) and (drp.z > 0.001) 
                and dap.daptype = 'HYB10-MILESHC-MASTARHC2' 
                and (drp.nsa_elpetro_mass > 5e+9)"""

Massives = CasJobs.executeQuery(Candidates, 'dr17')
print(Massives)

CandidateIDs = Massives['MaNGAid']
CandidateMasses = Massives['GalMass']
CandidateZs = Massives['z']
CandidateMags = Massives['uMag'] - Massives['rMag']
lwCandidateAges = Massives['LWStellarAge']


# In[4]:


#List of ID, Galaxy Mass, z, Magnitude, and Average Stellar Age for all Candidates

ID = []
GalMass = []
Redshift =[]
GalMag = []
lwAvgStelAge = []
GALINFO_RAW = []

for i in range(len(CandidateIDs)):
    ID.append([CandidateIDs[i]])

for i in range(len(CandidateMasses)):
    GalMass.append([CandidateMasses[i]])
    
for i in range(len(CandidateZs)):
    Redshift.append([CandidateZs[i]])

for i in range(len(CandidateMags)):
    GalMag.append([CandidateMags[i]])

for i in range(len(lwCandidateAges)):
    lwAvgStelAge.append([lwCandidateAges[i]])
    
print('No. of MaNGA IDs: ' + str(len(ID)))
print('No. of Galaxy Mass Values: ' + str(len(GalMass)))
print('No. of Redshift Values: ' + str(len(Redshift)))
print('No. of Magnitude Values: ' + str(len(GalMag)))
print('No. of Stellar Age Values: ' + str(len(lwAvgStelAge)) + '\n')

for i in range(len(CandidateIDs)):
    GALINFO_RAW.append([CandidateIDs[i]])
    GALINFO_RAW.append([CandidateMasses[i]])
    GALINFO_RAW.append([CandidateZs[i]])
    GALINFO_RAW.append([CandidateMags[i]])
    GALINFO_RAW.append([lwCandidateAges[i]])

print(GALINFO_RAW)


# In[5]:


##========================================================AGN SORTING=========================================================##


# In[6]:


#Creating list of massive galaxy AGN IDs
AGNIDs_DF = pd.read_excel('Massive_AGN_ID.xlsx')
AGNIDs = pd.DataFrame(AGNIDs_DF)
AGNIDs = AGNIDs.values.tolist()

AGNID = []

for i in range(len(AGNIDs)):
    AGNID.append(AGNIDs[i])
    
print('Target number of AGN Galaxies: ' + str(len(AGNID)) + '\n')
print('MaNGA IDs: ' + str(AGNID))


# In[7]:


AGN_ID = []
AGN_GalMass = []
AGN_Redshift = []
AGN_Magnitude = []
AGN_lwAvgStelAge = []

AGN_GalCount = 0 #Number of galaxies

for i in range(len(AGNID)):
    for x in range(len(ID)):
        if (AGNID[i] == ID[x]):
            AGN_ID.append(ID[x])
            AGN_GalMass.append(GalMass[x])
            AGN_Redshift.append(Redshift[x])
            AGN_Magnitude.append(GalMag[x])
            AGN_lwAvgStelAge.append(lwAvgStelAge[x])
            AGN_GalCount+=1
            break
            
print('AGN Dominated Galaxy Count: ' + str(AGN_GalCount))
print('IDs: ' + str(AGN_ID) + '\n')
print('Masses: ' + str(AGN_GalMass) + '\n')
print('Redshifts: ' + str(AGN_Redshift) + '\n')
print('Magnitudes: ' + str(AGN_Magnitude) + '\n')
print('Ages: ' + str(AGN_lwAvgStelAge) + '\n')


# In[8]:


#Removing brackets from element

for i in range(len(AGN_ID)):
    AGN_ID[i] = str(AGN_ID[i])[1:]
    AGN_ID[i] = str(AGN_ID[i])[:-1]
    
for i in range(len(AGN_GalMass)):
    AGN_GalMass[i] = str(AGN_GalMass[i])[1:]
    AGN_GalMass[i] = str(AGN_GalMass[i])[:-1]
    
for i in range(len(AGN_Redshift)):
    AGN_Redshift[i] = str(AGN_Redshift[i])[1:]
    AGN_Redshift[i] = str(AGN_Redshift[i])[:-1]
    
for i in range(len(AGN_Magnitude)):
    AGN_Magnitude[i] = str(AGN_Magnitude[i])[1:]
    AGN_Magnitude[i] = str(AGN_Magnitude[i])[:-1]

for i in range(len(AGN_lwAvgStelAge)):
    AGN_lwAvgStelAge[i] = str(AGN_lwAvgStelAge[i])[1:]
    AGN_lwAvgStelAge[i] = str(AGN_lwAvgStelAge[i])[:-1]
    
print(AGN_ID)
print(AGN_GalMass)
print(AGN_Redshift)
print(AGN_Magnitudes)
print(AGN_lwAvgStelAge)


# In[9]:


#Information is stored in a 4D array detailed below

                             #**SO FAR ONLY MANGA ID, GALAXY MASS, AND REDSHIFT ARE IN THE LIST**# 
                    ########################################################################################                 
                    #                                                                                      #
                    # #==========INDEX-0=====INDEX-1======INDEX-2=======INDEX-3=============INDEX-4        #
                    #                                                                                      #
                    # #INDEX-0  MaNGA-ID   Galaxy Mass   Redshift   Galaxy Magnitude   Average Stellar Age #
                    # #INDEX-1  MaNGA-ID   Galaxy Mass   Redshift   Galaxy Magnitude   Average Stellar Age #
                    # #INDEX-2  MaNGA-ID   Galaxy Mass   Redshift   Galaxy Magnitude   Average Stellar Age #
                    # #INDEX...                                                                            #
                    ########################################################################################           
                        
AGN_GALAXIES_CONDENSED = np.array(list(zip(AGN_ID, AGN_GalMass, AGN_Redshift, AGN_Magnitude, AGN_lwAvgStelAge)))
AGN_GALAXIES_CONDENSED = pd.DataFrame(AGN_GALAXIES_CONDENSED)
AGN_GALAXIES_CONDENSED


# In[10]:


#Changing values from strings to floats

for i in range(len(AGN_GALAXIES_CONDENSED)):
    AGN_GALAXIES_CONDENSED[1][i] = float(AGN_GALAXIES_CONDENSED[1][i])
    AGN_GALAXIES_CONDENSED[2][i] = float(AGN_GALAXIES_CONDENSED[2][i])
    AGN_GALAXIES_CONDENSED[3][i] = float(AGN_GALAXIES_CONDENSED[3][i])
    AGN_GALAXIES_CONDENSED[4][i] = float(AGN_GALAXIES_CONDENSED[4][i])
    
AGN_GALAXIES_CONDENSED


# In[11]:


##======================================================NON-AGN SORTING=======================================================##


# In[12]:


#Creating list of massive galaxy NON-AGN IDs
SFIDs_DF = pd.read_excel('Massive_SF_ID.xlsx')
SFIDs = pd.DataFrame(SFIDs_DF)
SFIDs = SFIDs.values.tolist()

SFID = []

for i in range(len(SFIDs)):
    SFID.append(SFIDs[i])
    
print('Target number of SF Galaxies: ' + str(len(SFID)) + '\n')
print('MaNGA IDs: ' + str(SFID))


# In[13]:


nonAGN_ID = []
nonAGN_GalMass = []
nonAGN_Redshift = []
nonAGN_Magnitude = []
nonAGN_lwAvgStelAge = []


nonAGN_GalCount = 0 #Number of galaxies

for i in range(len(SFID)):
    for x in range(len(ID)):
        if (SFID[i] == ID[x]):
            nonAGN_ID.append(ID[x])
            nonAGN_GalMass.append(GalMass[x])
            nonAGN_Redshift.append(Redshift[x])
            nonAGN_Magnitude.append(GalMag[x])
            nonAGN_lwAvgStelAge.append(lwAvgStelAge[x])
            nonAGN_GalCount+=1
            break
            
print('SF Dominated Galaxy Count: ' + str(nonAGN_GalCount))
print('IDs' + str(nonAGN_ID) + '\n')
print('Masses' + str(nonAGN_GalMass) + '\n')
print('Redshifts' + str(nonAGN_Redshift) + '\n')
print('Magnitudes: ' + str(nonAGN_Magnitude) + '\n')
print('Ages: ' + str(nonAGN_lwAvgStelAge) + '\n')


# In[14]:


#Removing brackets from element

for i in range(len(nonAGN_ID)):
    nonAGN_ID[i] = str(nonAGN_ID[i])[1:]
    nonAGN_ID[i] = str(nonAGN_ID[i])[:-1]
    
for i in range(len(nonAGN_GalMass)):
    nonAGN_GalMass[i] = str(nonAGN_GalMass[i])[1:]
    nonAGN_GalMass[i] = str(nonAGN_GalMass[i])[:-1]
    
for i in range(len(nonAGN_Redshift)):
    nonAGN_Redshift[i] = str(nonAGN_Redshift[i])[1:]
    nonAGN_Redshift[i] = str(nonAGN_Redshift[i])[:-1]
    
for i in range(len(nonAGN_Magnitude)):
    nonAGN_Magnitude[i] = str(nonAGN_Magnitude[i])[1:]
    nonAGN_Magnitude[i] = str(nonAGN_Magnitude[i])[:-1]

for i in range(len(nonAGN_lwAvgStelAge)):
    nonAGN_lwAvgStelAge[i] = str(nonAGN_lwAvgStelAge[i])[1:]
    nonAGN_lwAvgStelAge[i] = str(nonAGN_lwAvgStelAge[i])[:-1]
    
print(nonAGN_ID)
print(nonAGN_GalMass)
print(nonAGN_Redshift)
print(nonAGN_Magnitude)
print(nonAGN_lwAvgStelAge)


# In[15]:


#Information is stored in a 4D array detailed below

                             #**SO FAR ONLY MANGA ID, GALAXY MASS, AND REDSHIFT ARE IN THE LIST**# 
                    ########################################################################################                 
                    #                                                                                      #
                    # #==========INDEX-0=====INDEX-1======INDEX-2=======INDEX-3=============INDEX-4        #
                    #                                                                                      #
                    # #INDEX-0  MaNGA-ID   Galaxy Mass   Redshift   Galaxy Magnitude   Average Stellar Age #
                    # #INDEX-1  MaNGA-ID   Galaxy Mass   Redshift   Galaxy Magnitude   Average Stellar Age #
                    # #INDEX-2  MaNGA-ID   Galaxy Mass   Redshift   Galaxy Magnitude   Average Stellar Age #
                    # #INDEX...                                                                            #
                    ########################################################################################           
                        

nonAGN_GALAXIES_CONDENSED = np.array(list(zip(nonAGN_ID, nonAGN_GalMass, nonAGN_Redshift, nonAGN_Magnitude, nonAGN_lwAvgStelAge)))
nonAGN_GALAXIES_CONDENSED = pd.DataFrame(nonAGN_GALAXIES_CONDENSED)
nonAGN_GALAXIES_CONDENSED


# In[16]:


#Changing values from strings to floats

for i in range(len(nonAGN_GALAXIES_CONDENSED)):
    nonAGN_GALAXIES_CONDENSED[1][i] = float(nonAGN_GALAXIES_CONDENSED[1][i])
    nonAGN_GALAXIES_CONDENSED[2][i] = float(nonAGN_GALAXIES_CONDENSED[2][i])
    nonAGN_GALAXIES_CONDENSED[3][i] = float(nonAGN_GALAXIES_CONDENSED[3][i])
    nonAGN_GALAXIES_CONDENSED[4][i] = float(nonAGN_GALAXIES_CONDENSED[4][i])
    
AGN_GALAXIES_CONDENSED


# In[17]:


print((AGN_GALAXIES_CONDENSED[1][0]))


# In[18]:


##===================================================MATCHING AGN TO NON-AGN==================================================##


# In[23]:


AGN_PAIR_HALFS_LW = []
nonAGN_PAIR_HALFS_LW = []

MassThreshold = 0.1
zThreshold = 0.05
MagThreshold = 0.25

PairCount = 0

for i in range(len(AGN_GALAXIES_CONDENSED)):
    for x in range(len(nonAGN_GALAXIES_CONDENSED)):
        if ((nonAGN_GALAXIES_CONDENSED[4][x] not in nonAGN_PAIR_HALFS_LW)
                and
            ((AGN_GALAXIES_CONDENSED[4][i] > -1) and (nonAGN_GALAXIES_CONDENSED[4][x] > -1))
                and
            (((AGN_GALAXIES_CONDENSED[1][i]) - (AGN_GALAXIES_CONDENSED[1][i] * MassThreshold)) <= nonAGN_GALAXIES_CONDENSED[1][x] <= ((AGN_GALAXIES_CONDENSED[1][i]) + (AGN_GALAXIES_CONDENSED[1][i] * MassThreshold)))
                and 
            (((AGN_GALAXIES_CONDENSED[2][i]) - (AGN_GALAXIES_CONDENSED[2][i] * zThreshold)) <= nonAGN_GALAXIES_CONDENSED[2][x] <= ((AGN_GALAXIES_CONDENSED[2][i]) + (AGN_GALAXIES_CONDENSED[2][i] * zThreshold)))
                and
            (((AGN_GALAXIES_CONDENSED[3][i]) - (AGN_GALAXIES_CONDENSED[3][i] * MagThreshold)) <= nonAGN_GALAXIES_CONDENSED[3][x] <= ((AGN_GALAXIES_CONDENSED[3][i]) + (AGN_GALAXIES_CONDENSED[3][i] * MagThreshold)))
           ):
            
            AGN_PAIR_HALFS_LW.append(AGN_GALAXIES_CONDENSED[4][i])
            nonAGN_PAIR_HALFS_LW.append(nonAGN_GALAXIES_CONDENSED[4][x])
            PairCount +=1
            break

PercentMatched = (PairCount/AGN_GalCount)*100
            
print('Total AGN/Non-AGN Galaxy Pairs: ' + str(PairCount))
print('Percentage of AGN Galaxies Paried: %.3f' % PercentMatched)


# In[27]:


COMPLETE_PAIRS_LW = (AGN_PAIR_HALFS_LW, nonAGN_PAIR_HALFS_LW)

print('No. of AGN Pair Halfs: ' + str(len(AGN_PAIR_HALFS_LW)))
print('No. of SF Pair Halfs: ' + str(len(nonAGN_PAIR_HALFS_LW)) + '\n')
print('List of Both Halfs: ' + str(COMPLETE_PAIRS_LW))


# In[28]:


#Creating Excel files for Complete Pair Data
from openpyxl import Workbook
import xlsxwriter

#Light Weighted AGN Pairs
m_AGNLWExcel = pd.DataFrame({"Massive AGN LW Stellar Population Age Data": AGN_PAIR_HALFS_LW})
m_AGNLWExcel.to_excel("MASSIVE_AGN_LW_StelAgeData.xlsx", index=False)

#Light Weighted SF Pairs
m_SFLWExcel = pd.DataFrame({"Massive SF LW Stellar Population Age Data": nonAGN_PAIR_HALFS_LW})
m_SFLWExcel.to_excel("MASSIVE_SF_LW_StelAgeData.xlsx", index=False)

#Mass Weighted SF Pairs
#m_AGNMWExcel = pd.DataFrame({"Massive AGN MW Stellar Population Age Data": AGN_PAIRS_MW})
#m_AGNMWExcel.to_excel("MASSIVE_AGN_MW_StelAgeData.xlsx", index=False)

#Mass Weighted SF Pairs
#m_SFMWExcel = pd.DataFrame({"Massive SF MW Stellar Population Age Data": SF_PAIRS_MW})
#m_SFMWExcel.to_excel("MASSIVE_SF_MW_StelAgeData.xlsx", index=False)


# In[34]:


alphas = [0.8, 0.8]
facecolor = ['#D9E7FA','#D9E7FA']
edgecolors = ['#02BEF7','#0295F7']

sns.set(style = 'whitegrid') 

vlnplt = sns.violinplot(data = COMPLETE_PAIRS_LW)
plt.yticks([-1.0, -0.5, 0, 0.5, 1.0], ['-1.0', '-0.5','0', '0.5', '1.0'])

for violin, alpha in zip(vlnplt.collections[::2], alphas):
    violin.set_alpha(alpha)
    
for pc, color in zip(vlnplt.collections[::2], facecolor):
    pc.set_facecolor(color)

for pc, color in zip(vlnplt.collections[::2], edgecolors):
    pc.set_edgecolor(color)
    
    
srmplt = sns.swarmplot(data = COMPLETE_PAIRS_LW, alpha = 0.4, s = 2.5, color = '#E041D8')
srmplt.set_ylabel("Mean Stellar Age " + "$(log(Gyr))$")
srmplt.set_title("Massive Galaxy Mean Stellar Ages")
srmplt.set_xticklabels(['AGN-Dominated\nMassive Galaxies', 'SF-Dominated\nMassive Galaxies'])

plt.savefig('Massive Galaxy Stellar Population Age', dpi = 600, bbox_inches = 'tight')


# In[ ]:


##=====================================================STATISTICAL ANALYSIS===================================================##


# In[31]:


sps.ks_2samp(AGN_PAIR_HALFS_LW, nonAGN_PAIR_HALFS_LW)


# In[32]:


sps.ttest_i(AGN_PAIR_HALFS_LW, nonAGN_PAIR_HALFS_LW)
