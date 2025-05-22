import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def load_and_filter_data(file_path, country_filter = "", separator=';', encoding='ANSI'):
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


def prepare_data_online_retail(data):
    data = data[(data['Country'] == 'France')] #garde uniquement pour la france
    data = data[~data['Invoice'].str.startswith('C')]
    tabc = pd.crosstab(data['Invoice'], data['Description'])
    
    tabc = tabc.applymap(lambda x: True if x > 0 else False)
    return tabc



data = load_and_filter_data("panier2.csv" )


tableau_bool = prepare_data(data)


frequent_itemsets, rules = analyze_apriori(tableau_bool, min_support=0.5, metric="confidence", min_threshold=0.8)

print("Itemsets fréquents :")
print(frequent_itemsets)

print("\nRègles d'association :")
print(rules)

data2009 = load_and_filter_data("online_retail_2009-2010.csv", "France")
data2010 = load_and_filter_data("online_retail_2010_2011.csv", "France")
dataRetail = pd.concat([data2009, data2010])
tableau_boolean = prepare_data_online_retail(dataRetail)

frequent_itemsets_retail, rules_retail = analyze_apriori(tableau_boolean, min_support=0.1, metric="confidence", min_threshold=0.8)
print(f"itemset 2009: ${frequent_itemsets_retail}")
print(f"rules 2009: ${rules_retail}")