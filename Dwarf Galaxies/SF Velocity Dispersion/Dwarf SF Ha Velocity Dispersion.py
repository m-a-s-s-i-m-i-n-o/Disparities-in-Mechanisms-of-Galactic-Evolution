#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import math 
import astropy
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as sps

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import statsmodels.api as sm


# In[3]:


#SciServer Modules
import SciServer
from SciServer import CasJobs


# In[4]:


AGNIDs_DF = pd.read_excel('Dwarf_SF_ID.xlsx')
AGNIDs = pd.DataFrame(AGNIDs_DF)
AGNIDs = AGNIDs.values.tolist()

AGNID = []

for i in range(len(AGNIDs)):
    AGNID.append(AGNIDs[i])
    
print('Target number of AGN Galaxies: ' + str(len(AGNID)) + '\n')
print('MaNGA IDs: ' + str(AGNID))


# In[5]:


Properties = """select drp.MaNGAid,
                drp.z,
                drp.nsa_elpetro_mass as GalMass,
                dap.ha_gsigma_1re as Ha_VelDisp,
                dap.nsa_sersic_th50 as Distance,
                
                dap.daptype from mangaDRPall as drp
                JOIN mangaDAPall as dap on dap.mangaid = drp.mangaid
                WHERE (drp.z < 0.15) and (drp.z > 0.001)
                and dap.daptype = 'HYB10-MILESHC-MASTARHC2'
                and (drp.nsa_elpetro_mass < 5e+9)"""

Selection1 = CasJobs.executeQuery(Properties, 'dr17')
print(Selection1)

MaNGAIDs_raw = Selection1['MaNGAid']
GalMass_raw = Selection1['GalMass']
Radius_raw = Selection1['Distance']
Ha_VelDisp_raw = Selection1['Ha_VelDisp']


# In[6]:


MaNGAIDs = []
GalMass = []
Radius = []

Ha_VelDisp = []

for i in range(len(MaNGAIDs_raw)):
    MaNGAIDs.append([MaNGAIDs_raw[i]])

for i in range(len(GalMass_raw)):
    GalMass.append([GalMass_raw[i]])
    
for i in range(len(Radius_raw)):
    Radius.append([Radius_raw[i]])
    
for i in range(len(Ha_VelDisp_raw)):
    Ha_VelDisp.append([Ha_VelDisp_raw[i]])
    
print('No. of MaNGA IDs: ' + str(len(MaNGAIDs)))
print('No. of Galaxy Mass Values: ' + str(len(GalMass)))
print('No. of Radius Values: ' + str(len(Radius)))
print('No. of Ha Velocity Dispersion Values: ' + str(len(Ha_VelDisp)))


# In[7]:


GalIDs_fin = []
GalMass_fin = []
Radius_fin = []

Ha_VelDisp_fin = []

for i in range(len(AGNID)):
    for x in range(len(MaNGAIDs)):
        if (AGNID[i] == MaNGAIDs[x]):
            GalIDs_fin.append(MaNGAIDs[x])
            GalMass_fin.append(GalMass[x])
            Radius_fin.append(Radius[x])
            Ha_VelDisp_fin.append(Ha_VelDisp[x])
            break

#Target quantity is 531
print('Galaxies: ' + str(len(GalIDs_fin)))
print('Galaxy Mass Values: ' + str(len(GalMass_fin)))
print('Radius Values: ' + str(len(Radius_fin)) + '\n')
print('Ha Velocity Dispersion Values: ' + str(len(Ha_VelDisp_fin)))


# In[8]:


#Removing brackets from each element
for i in range(len(Ha_VelDisp_fin)):
    Ha_VelDisp_fin[i] = str(Ha_VelDisp_fin[i])[1:]
    Ha_VelDisp_fin[i] = str(Ha_VelDisp_fin[i])[:-1]
    
for i in range(len(GalMass_fin)):
    GalMass_fin[i] = str(GalMass_fin[i])[1:]
    GalMass_fin[i] = str(GalMass_fin[i])[:-1]


# In[9]:


#Merging values into a 2D array to select points graphed
PltPts2D = np.array(list(zip(Ha_VelDisp_fin, GalMass_fin)))

print('Length: ' + str(len(PltPts2D)) + '\n')
PltPts2D = pd.DataFrame(PltPts2D)

PltPts2D


# In[10]:


for i in range(2372):
    PltPts2D[0][i] = float(PltPts2D[0][i])
    PltPts2D[1][i] = float(PltPts2D[1][i])
    
print(PltPts2D)


# In[11]:


#Omitting values less than zero
PltPts2D_Ha =[]
PltPts2D_Rad =[]

for i in range(len(PltPts2D)):
    if((0 < PltPts2D[0][i] < 65) and (0 < PltPts2D[1][i])):
        PltPts2D_Ha.append(PltPts2D[0][i])
        PltPts2D_Rad.append(PltPts2D[1][i])
        
PltPts2D_z = np.array(list(zip(PltPts2D_Ha, PltPts2D_Rad)))

print('2D Plot Point Array Length: ' + str(len(PltPts2D_z)) + '\n')
print(min(PltPts2D_Rad))
print(max(PltPts2D_Rad))
print(PltPts2D_z)


# In[12]:


#Finding min and max Ha values
#MinHa = min(PltPts2D_Ha)
#MaxHa = max(PltPts2D_Ha)

#print('Minimum: ' + str(MinHa))
#print('Maximum: ' + str(MaxHa))


# In[13]:


#Filtering based on IQR
#OutliersX = []
#OutliersY = []
#PltPtsIQR_Ha =[]
#PltPtsIQR_Rad =[]

#for i in range(len(PltPts2D_z)):
    #if(((abs(PltPts2D_z[i][0]) - MinHa) < RNG) or ((abs(PltPts2D_z[i][1]) - MaxHa) > RNG)):
        #OutliersX.append(PltPts2D_z[i][0])
        #OutliersY.append(PltPts2D_z[i][1])
    #else:
        #PltPtsIQR_Ha.append(PltPts2D_z[i][0])
        #PltPtsIQR_Rad.append(PltPts2D_z[i][1])
        
#PltPts2D_fin = np.array(list(zip(PltPtsIQR_Ha, PltPtsIQR_Rad)))

#print('2D Plot Point Array Length: ' + str(len(PltPts2D_fin)) + '\n')
#print(PltPts2D_fin)


# In[14]:


#Splitting elements back into two lists for plotting
Ha_VelDisp_plt = []
GalMass_plt = []

for i in range(len(PltPts2D_z)):
    Ha_VelDisp_plt.append([PltPts2D_z[i][0]])
    GalMass_plt.append([PltPts2D_z[i][1]])
    
print('Ha Velocity Dispersion Values: ' + str(len(Ha_VelDisp_plt)))
print('Radius Values: ' + str(len(GalMass_plt)))

print(Ha_VelDisp_plt)
print(GalMass_plt)


# In[18]:


plt.figure(figsize = (8, 8))
plt.title('$H{\u03B1}$ ${\u03C3}$: SF-Dominated Dwarf Galaxies', fontsize = 16)

plt.scatter(Ha_VelDisp_plt, 
            GalMass_plt, 
            color = 'white', 
            edgecolor = '#fb7d00', 
            marker = 'v', 
            alpha = 1, 
            s = 30, 
            label = "MaNGA SF Dwarf Galaxy")

plt.xlabel('${\u03C3}$: $H{\u03B1}$ (kms\u207B\u00B9)', fontsize = 16)
plt.ylabel('Galaxy Mass ($M_{\odot}$)', fontsize = 16)

#plt.xlim(10, 65)
#plt.ylim(0, 25)

#Regression Line
poly_features = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_features.fit_transform(Ha_VelDisp_plt)

Reg = LinearRegression()
Reg.fit(X_poly, GalMass_plt)

XValues_lin = np.linspace(10, 65, 1882).reshape(-1, 1)
XValues_poly = poly_features.transform(XValues_lin)

YValues = Reg.predict(XValues_poly)

plt.plot(XValues_lin, YValues, c ='red')
legend = plt.legend(loc = 'upper left')
plt.savefig('Ha Velocity Dispersion: Dwarf SF', dpi = 600, bbox_inches = 'tight')
plt.show()


# In[19]:


#Calculating p-value
mod = sm.OLS(GalMass_plt, Ha_VelDisp_plt)
fii = mod.fit()
p_value = fii.summary2().tables[1]['P>|t|']

print(p_value)
#p < .0001


# In[22]:


#Calculating r-value
r2 = r2_score(GalMass_plt, YValues)
r = (np.sqrt(abs(r2)))
print('r = ' + str(r))

