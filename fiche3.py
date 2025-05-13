import pandas as pd 
import numpy as np
import os 
from mlxtend.frequent_patterns import apriori, association_rules 
os.chdir("C:\Cours\2emeAnnee\4emeSemestre\R4.04")
dataset = pd.read_table('panier2.csv', sep=';', encoding='ASCII')
tabc= pd.crosstab(dataset.trans, dataset.produit)
tabc.replace(0,False)
tabc.replace(1,True)
def apriori(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0, low_memory=False):
    return