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
import matplotlib.colors as colors


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


#SciServer Modules
import SciServer
from SciServer import CasJobs


# In[3]:


#Query of fluxes for different emission lines
EmLFQry= """select drp.MaNGAid, drp.objRA, drp.objDEC, drp.z, 
drp.nsa_elpetro_absmag_r as rMAG, 
drp.nsa_elpetro_absmag_g as gMAG, 
drp.nsa_elpetro_mass as GalMass, 

dap.emline_sflux_cen_oiii_5008 as OIII,  
dap.emline_sflux_cen_hb_4862 as HB, 
dap.emline_sflux_cen_nii_6585 as NII, 
emline_sflux_cen_ha_6564 as Ha, 

dap.daptype
from mangaDRPall as drp
JOIN mangaDAPall as dap on dap.mangaid = drp.mangaid
WHERE (drp.z < 0.15) and (drp.z > 0.001) 
and dap.daptype = 'HYB10-MILESHC-MASTARHC2'
and (5e9 > drp.nsa_elpetro_mass)
"""
           
Selection1 = CasJobs.executeQuery(EmLFQry, 'dr17')
print(Selection1)

GalMass = []
for i in range(len(Selection1['GalMass'])):
    GalMass.append([Selection1['GalMass'][i]])


# In[4]:


print('Max Mass Index: ' + str(GalMass.index(max(GalMass))))
print('Min Mass Index: ' + str(GalMass.index(min(GalMass))))


# In[13]:


#Max Massive Mass
i = 1324

print('Most Massive Galaxy: \n')
print('ID: ' + str(Selection1['MaNGAid'][i]))
print('Mass: ' + str(Selection1['GalMass'][i]))
print('z: ' + str(Selection1['z'][i]))
print('OIII: ' + str(Selection1['OIII'][i]))
print('HB: ' + str(Selection1['HB'][i]))
print('NII: ' + str(Selection1['NII'][i]))
print('Ha: ' + str(Selection1['Ha'][i]) + '\n')
print('\n')

#Mass Massive Mass
i = 46

print('Least Massive Galaxy: \n')
print('ID: ' + str(Selection1['MaNGAid'][i]))
print('Mass: ' + str(Selection1['GalMass'][i]))
print('z: ' + str(Selection1['z'][i]))
print('OIII: ' + str(Selection1['OIII'][i]))
print('HB: ' + str(Selection1['HB'][i]))
print('NII: ' + str(Selection1['NII'][i]))
print('Ha: ' + str(Selection1['Ha'][i]))


# In[14]:


#Omitting NaN values from the raw data
BPTX = abs(Selection1['NII']/Selection1['Ha'])
BPTY = abs(Selection1['OIII']/Selection1['HB'])
ID = Selection1['MaNGAid']

BPTX = BPTX.dropna()
BPTY = BPTY.dropna()

BPTX = list(filter(None, BPTX))
BPTY = list(filter(None, BPTY))

print (len(BPTX))
print (len(BPTY))


# In[15]:


#Separating the data based on the equations provided by Kauffman and Trou
agnBPTx = []
agnBPTy = []

sfBPTx = []
sfBPTy = []

gBPTx = []
gBPTy = []

#MaNGA ID Lists
AGN_ID = []

SF_ID = []

MaNGAID = Selection1['MaNGAid']

for i in range(len(BPTX)):
    if (BPTY[i] > ((-0.61 * (np.log(BPTX[i]) - 0.05)) + 1.3)) and (BPTY[i] > ((-1.2 * np.log(BPTX[i])) - 0.4)):
        agnBPTx.append(BPTX[i])
        agnBPTy.append(BPTY[i])
        AGN_ID.append(ID[i])
    elif (BPTY[i] < (-1.2 * np.log(BPTX[i])) - 0.4) and (BPTY[i] < ((-0.61 * (np.log(BPTX[i]) - 0.05)) + 1.3)):
        sfBPTx.append(BPTX[i])
        sfBPTy.append(BPTY[i])
        SF_ID.append(ID[i])
    elif (((BPTY[i] > (-1.2 * np.log(BPTX[i])) - 0.4) and (BPTY[i] < (-0.61 * (np.log(BPTX[i]) - 0.05) + 1.3 ))) or ((BPTY[i] < ((-0.61 * (np.log(BPTX[i]) - 0.05)) + 1.3)) and (BPTY[i] > (-1.2 * np.log(BPTX[i])) - 0.4))):
        gBPTx.append(BPTX[i])
        gBPTy.append(BPTY[i])
    else:
        print(str(BPTX[i]) + ', ' + str(BPTY[i]) + ': Index = ' + str(i))
        print('Something is broken :( \n')

        
print('Current Index: ' + str(i))
print(str(MaNGAID[i]) + '\n')

print('AGN Galaxies: ' + str(len(agnBPTx)))
print('Star Forming Galaxies: ' + str(len(sfBPTx)))
print('Grey Galaxies: ' + str(len(gBPTx)) + '\n')

print('AGN x: ' + str(agnBPTx))
print('AGN y: ' + str(agnBPTy) + '\n')

print('Star Forming x:' + str(sfBPTx))
print('Star Forming y:' + str(sfBPTy) + '\n')

print('Grey Galaxies x:' + str(gBPTx))
print('Grey Galaxies y:' + str(gBPTy) + '\n') 


# In[22]:


#Creating Excel file for MaNGA IDs
from openpyxl import Workbook
import xlsxwriter

#AGN IDs
AGNExcel = pd.DataFrame({"MaNGA IDs: AGN": AGN_ID})
AGNExcel.to_excel("Dwarf_AGN_ID.xlsx", index=False)

#Star Forming IDs
SFExcel = pd.DataFrame({"MaNGA IDs: Star Forming": SF_ID})
SFExcel.to_excel("Dwarf_SF_ID.xlsx", index=False)


# In[17]:


#Equations separating star forming and AGN populations
#Kaufmann et al. 2003
AGN = (-0.61 * (np.log(BPTX) - 0.05)) + 1.3
#Trouille et al. 2011
SF = (-1.2 * np.log(BPTX)) - 0.4


# In[18]:


#Correcting order for pop separated lines
AGN_Line = np.array(list(zip(BPTX, AGN)))
AGN_Line = sorted(AGN_Line, key=lambda x: x[0])

SF_Line = np.array(list(zip(BPTX, SF)))
SF_Line = sorted(SF_Line, key=lambda x: x[0])


# In[19]:


#Splitting elements back into two lists
BPTX_plt = []
AGN_plt = []
SF_plt = []

for i in range(len(AGN_Line)):
    BPTX_plt.append([AGN_Line[i][0]])
    AGN_plt.append([AGN_Line[i][1]])

for i in range(len(SF_Line)):
    SF_plt.append([SF_Line[i][1]])


# In[21]:


plt.figure(figsize = (8, 8))
plt.title('BPT Dwarf Galaxies', fontsize = 16)

#AGN
plt.scatter(agnBPTx, agnBPTy, color = 'maroon', alpha = 1, s = 1.5, label = "AGN")
#SF
plt.scatter(sfBPTx, sfBPTy, color = 'royalblue', alpha = 1, s = 1.5, label = "Star Forming")
#Grey
plt.scatter(gBPTx, gBPTy, color = 'grey', alpha = 1, s = 1.5)


plt.ylabel('log ([OIII] ${\u03BB}$5008/$H{\u03B2}$)', fontsize = 16)
plt.xlabel('log ([NII] ${\u03BB}$6585/$H{\u03B1}$)', fontsize = 16)
plt.xscale('log')
plt.yscale('log')
plt.xlim(.5e-1, 0.5e1)
plt.ylim(1e-1, 1.75e1)

plt.plot(BPTX_plt, AGN_plt, color = 'black', linestyle = ':', linewidth = 3, label = "Kaufmann et al. 2003")
plt.plot(BPTX_plt, SF_plt, color = 'black', linestyle = '--', linewidth = 2, label = "Trouille et al. 2011")
legend = plt.legend(loc = 'upper left')

plt.savefig('Dwarf Galaxy BPT Diagram', dpi = 600, bbox_inches = 'tight')
plt.show()

