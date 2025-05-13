import pandas as pd 
import numpy as np
import os 
from mlxtend.frequent_patterns import apriori, association_rules 
os.chdir("C:\\Cours\\2emeAnnee\\4emeSemestre\\R4.04")
dataset = pd.read_table('panier2.csv', sep=';', encoding='ANSI')
tabc= pd.crosstab(dataset.trans, dataset.produit)
tabc.replace(0,False)
tabc.replace(1,True)
freq_itemsets = apriori(tabc, min_support=0.3, max_len=4, use_colnames=True)