
# coding: utf-8

# In[572]:

# Nils Napp
# Example network for doing disgnosis 

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.sampling import GibbsSampling
from pgmpy.factors.discrete import State
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# The random variables in the BN have strings as names
# Edges are directed (FROm, TO) and the Bayes net automatically
# adds the noeds that are assoiated with edges that it has not seen
# This definition is equivalent to first adding the 
# rancom variables 'Healthy', 'Flu', 'FluShot', 'Lyme','Fatigue','Voit',
#'Fever', and 'Numbness', and then adding the associated edges

disease_model=BayesianModel([('Healthy','Flu'),
                           ('Healthy','Lyme'), 
                           ('FluShot','Flu'), 
                           ('Flu', 'Fatigue'), 
                           ('Flu', 'Vomit'),
                           ('Flu', 'Fever'),
                           ('Lyme', 'Fatigue'),
                           ('Lyme', 'Numbness')])
                           


# In[573]:

#For these binary RV, assume that the first entry is P(RV=true) and the second one is P(RV=false)

pH=TabularCPD(variable='Healthy',
             variable_card=2,
              values=[[0.7, 0.3]])

pS=TabularCPD(variable='FluShot',
             variable_card=2,
              values=[[0.3, 0.7]])

pF_HS=TabularCPD(variable='Flu',
             variable_card=2,
              evidence=['Healthy','FluShot'],
              evidence_card=[2,2],
              # this means each row shoule have four entries orderd as
              # Health^FluShot Health^!FluShot !Health^FluShot !Health^!FluShot
              values=[[0.000, 0.00, 0.2, 0.8 ],
                      [1, 1, 0.8, 0.2]])

pL_H=TabularCPD(variable='Lyme',
             variable_card=2,
              evidence=['Healthy'],
              evidence_card=[2],
              values=[[0.0, 0.1],
                        [1, 0.9]])

pVomit_F=TabularCPD(variable='Vomit',
             variable_card=2,
              evidence=['Flu'],
              evidence_card=[2],
                    values=[[0.9, 0.2],
                            [0.1, 0.8]])

pFever_F=TabularCPD(variable='Fever',
             variable_card=2,
              evidence=['Flu'],
              evidence_card=[2],
                    values=[[0.9, 0.3],
                            [0.1, 0.7]])

pNumbness_L=TabularCPD(variable='Numbness',
                       variable_card=2,
                       evidence=['Lyme'],
                       evidence_card=[2],
                       values=[[0.8, 0.3],
                               [0.2, 0.7]])

pFatigue_FL=TabularCPD(variable='Fatigue',
                         variable_card=2,
                         evidence=['Flu','Lyme'],
                         evidence_card=[2,2],
                         # F^L , F^!L, !F^L, !F^!L Health&!FluShot !Health&FluShot !Health&!FluShot
                         values=[[0.99, 0.90, 0.7, 0.1],
                                 [0.01, 0.1, 0.3, 0.9]])


disease_model.add_cpds(pH,pS,pF_HS,pL_H,pFatigue_FL,pVomit_F,pFever_F,pNumbness_L)
                       

infer=VariableElimination(disease_model)

lyme=infer.query(['Lyme'])['Lyme']
print(infer.query(['Healthy'])['Healthy'])


# In[ ]:




# In[574]:

n=infer.query(['Flu'],evidence={'Fatigue':0,'Fever':0,'FluShot':0})['Flu']


# In[ ]:




# In[575]:

def gibb_sam(n):
    gib_chain=GibbsSampling(disease_model)
    # gib_chain.sample(size=30)

    gen=gib_chain.generate_sample(size=n)
    l=[sample for sample in gen]
    r=0
    for c,x in enumerate(l):
        for j,(var,st) in enumerate(x):
            if (var,st)==('Flu',0):
                r=r+1
                plt.plot(c,(r/n),'bo')
    plt.show()

    
      


# In[576]:

gibb_sam(1000)

