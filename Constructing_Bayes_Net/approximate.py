
# coding: utf-8

# In[327]:

# Nils Napp
# Example network for doing disgnosis 

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
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
                           


# In[328]:

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


# In[329]:

# print(lyme)
#
#
# print(infer.query(['Flu'],evidence={'Healthy':1,'FluShot':0})['Flu'])
#
# q1=infer.query(['Fatigue'],evidence={'Flu':0})['Fatigue']
# q2=infer.query(['Fatigue'],evidence={'Flu':0,'FluShot':1})['Fatigue']
# q3=infer.query(['Fatigue'],evidence={'Flu':0,'FluShot':0})['Fatigue']
#
# q4=infer.query(['Fatigue']['Fever'],evidence={'FluShot':0})['Fatigue']['Fever']
#
# q5=infer.query(['Fatigue'],evidence={'Lyme':0,'FluShot':1})['Fatigue']
#
# print(q1)
# print(q2)
# print(q3)
# print(q5)
# 1= not true.


# In[330]:

n=infer.query(['Flu'],evidence={'Fatigue':0,'Fever':0,'FluShot':0})['Flu']


# In[331]:

def rejection_estimate(n):
    inferences=BayesianModelSampling(disease_model)
    evidences = [State(var='Fatigue', state=0),State(var='Fever', state=0),State(var='FluShot', state=0)]

    p=inferences.rejection_sample(evidences,n)
    i=0


    for t in range(n):
        if p['Flu'][t]==float(0):
            i=i+1
            plt.plot(t,(i/n),'bo')
    plt.ylabel('Evolving esimate')
    plt.xlabel('Number of samples')
    plt.show()


# In[332]:

rejection_estimate(200)

