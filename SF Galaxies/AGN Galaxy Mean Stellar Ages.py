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
import seaborn as sns
import scipy.stats as sps


# In[3]:


##===========================================================SORTING==========================================================##


# In[4]:


#Opening Excel Files with Data
DWARF_AGN_PAIR_HALFS_LW_raw = pd.read_excel('DWARF_AGN_LW_StelAgeData.xlsx')
DWARF_AGN_PAIR_HALFS_LW_df = pd.DataFrame(DWARF_AGN_PAIR_HALFS_LW_raw)
DWARF_AGN_PAIR_HALFS_LW = DWARF_AGN_PAIR_HALFS_LW_df.values.tolist()

d_AGN_PAIR_HALFS_LW = []

for i in range(len(DWARF_AGN_PAIR_HALFS_LW)):
    d_AGN_PAIR_HALFS_LW.append(DWARF_AGN_PAIR_HALFS_LW[i])

MASSIVE_AGN_PAIR_HALFS_LW_raw = pd.read_excel('MASSIVE_AGN_LW_StelAgeData.xlsx')
MASSIVE_AGN_PAIR_HALFS_LW_df = pd.DataFrame(MASSIVE_AGN_PAIR_HALFS_LW_raw)
MASSIVE_AGN_PAIR_HALFS_LW = MASSIVE_AGN_PAIR_HALFS_LW_df.values.tolist()

m_AGN_PAIR_HALFS_LW = []

for i in range(len(MASSIVE_AGN_PAIR_HALFS_LW)):
    m_AGN_PAIR_HALFS_LW.append(MASSIVE_AGN_PAIR_HALFS_LW[i])
    
print('Dwarf AGN LW Stellar Age Data (' + str(len(d_AGN_PAIR_HALFS_LW)) + '): ' + str(d_AGN_PAIR_HALFS_LW) + '\n')

print('Massive AGN LW Stellar Age Data(' + str(len(m_AGN_PAIR_HALFS_LW)) + '): ' + str(m_AGN_PAIR_HALFS_LW))


# In[12]:


dm_COMPLETE_PAIRS_LW = (d_AGN_PAIR_HALFS_LW, m_AGN_PAIR_HALFS_LW)
dm_COMPLETE_PAIRS_LW


# In[6]:


##===========================================================PLOTING==========================================================##


# In[14]:


alphas = [0.8, 0.8]
facecolor = ['#FAEFD9','#D9E7FA']
edgecolors = ['#FB7D00','#02BEF7']

sns.set(style = 'whitegrid') 

vlnplt = sns.violinplot(data = dm_COMPLETE_PAIRS_LW)
plt.yticks([-1.0, -0.5, 0, 0.5, 1.0], ['-1.0', '-0.5','0', '0.5', '1.0'])

for violin, alpha in zip(vlnplt.collections[::2], alphas):
    violin.set_alpha(alpha)
    
for pc, color in zip(vlnplt.collections[::2], facecolor):
    pc.set_facecolor(color)

for pc, color in zip(vlnplt.collections[::2], edgecolors):
    pc.set_edgecolor(color)
    
    
vlnplt.set_ylabel("Mean Stellar Age " + "$(log(Gyr))$")
vlnplt.set_title("AGN Galaxy Mean Stellar Ages")
vlnplt.set_xticklabels(['AGN-Dominated\nDwarf Galaxies', 'AGN-Dominated\nMassive Galaxies'])

plt.savefig('AGN Dwarf and Massive Galaxy Stellar Population Age', dpi = 600, bbox_inches = 'tight')


# In[8]:


##=====================================================STATISTICAL ANALYSIS===================================================##


# In[13]:


sps.ttest_ind(d_AGN_PAIR_HALFS_LW, m_AGN_PAIR_HALFS_LW)


# In[ ]:




