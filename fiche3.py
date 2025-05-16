import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def load_and_filter_data(file_path, country_filter = "", separator=';', encoding='latin1'):
    dataset = pd.read_csv(file_path, sep=separator, encoding=encoding)
    return dataset

def prepare_data(data):

    tabc = pd.crosstab(data['trans'], data['produit'])
    
    tabc = tabc > 0
    return tabc

def analyze_apriori(dataframe, min_support=0.3, metric="confidence", min_threshold=0.8):

    frequent_itemsets = apriori(dataframe, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold, num_itemsets=4)
    return frequent_itemsets, rules



file_path = "panier2.csv"  


data = load_and_filter_data(file_path)


tableau_bool = prepare_data(data)


frequent_itemsets, rules = analyze_apriori(tableau_bool, min_support=0.5, metric="confidence", min_threshold=0.8)

print("Itemsets fréquents :")
print(frequent_itemsets)

print("\nRègles d'association :")
print(rules)
