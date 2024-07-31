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


# In[ ]:


##===========================================================SORTING==========================================================##


# In[3]:


#Opening Excel Files with Data
DWARF_SF_PAIR_HALFS_LW_raw = pd.read_excel('DWARF_SF_LW_StelAgeData.xlsx')
DWARF_SF_PAIR_HALFS_LW_df = pd.DataFrame(DWARF_SF_PAIR_HALFS_LW_raw)
DWARF_SF_PAIR_HALFS_LW = DWARF_SF_PAIR_HALFS_LW_df.values.tolist()

d_SF_PAIR_HALFS_LW = []

for i in range(len(DWARF_SF_PAIR_HALFS_LW)):
    d_SF_PAIR_HALFS_LW.append(DWARF_SF_PAIR_HALFS_LW[i])

MASSIVE_SF_PAIR_HALFS_LW_raw = pd.read_excel('MASSIVE_SF_LW_StelAgeData.xlsx')
MASSIVE_SF_PAIR_HALFS_LW_df = pd.DataFrame(MASSIVE_SF_PAIR_HALFS_LW_raw)
MASSIVE_SF_PAIR_HALFS_LW = MASSIVE_SF_PAIR_HALFS_LW_df.values.tolist()

m_SF_PAIR_HALFS_LW = []

for i in range(len(MASSIVE_SF_PAIR_HALFS_LW)):
    m_SF_PAIR_HALFS_LW.append(MASSIVE_SF_PAIR_HALFS_LW[i])
    
print('Dwarf SF LW Stellar Age Data (' + str(len(d_SF_PAIR_HALFS_LW)) + '): ' + str(d_SF_PAIR_HALFS_LW) + '\n')

print('Massive SF LW Stellar Age Data(' + str(len(m_SF_PAIR_HALFS_LW)) + '): ' + str(m_SF_PAIR_HALFS_LW))


# In[4]:


dm_COMPLETE_PAIRS_LW = (d_SF_PAIR_HALFS_LW, m_SF_PAIR_HALFS_LW)


# In[ ]:


##===========================================================PLOTING==========================================================##


# In[10]:


alphas = [0.8, 0.8]
facecolor = ['#FAEFD9','#D9E7FA']
edgecolors = ['#FBA300','#0295F7']

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
vlnplt.set_title("SF Galaxy Mean Stellar Ages")
vlnplt.set_xticklabels(['SF-Dominated\nDwarf Galaxies', 'SF-Dominated\nMassive Galaxies'])

plt.savefig('SF Dwarf and Massive Galaxy Stellar Population Age', dpi = 600, bbox_inches = 'tight')


# In[ ]:


##=====================================================STATISTICAL ANALYSIS===================================================##


# In[9]:


sps.ttest_ind(d_SF_PAIR_HALFS_LW, m_SF_PAIR_HALFS_LW)

