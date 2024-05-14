###RULES FILTER AND ANOMALY DETECTION

import pandas as pd

att = pd.read_csv('C:/Users/User/Desktop/THESIS/Association rule mining/fpgrowth/Delay fpgrowth/delay_fpgrowthRUN1.csv')
saf = pd.read_csv('C:/Users/User/Desktop/THESIS/Association rule mining/fpgrowth/Safe fpgrowth/safe_fpgrowthRUN1.csv')

#removing the columns with unnecessary metrics
att = att.iloc[:, :2]
#removing the repetitive rules and only leaving the ones with 1 antecedent and 1 consequent
mask = ~att.astype(str).apply(lambda x: x.str.contains(',')).any(axis=1)
fnd = att[mask].copy()
#filtering out rules with confidence metric over 40%
fnd = fnd[fnd['confidence'] < 0.4]

#doing the same with safe data
saf = saf.iloc[:, :2]
masky = ~saf.astype(str).apply(lambda x: x.str.contains(',')).any(axis=1)
sfd = saf[masky].copy()
sfd = sfd[sfd['confidence'] < 0.4]

from fuzzywuzzy import fuzz
#iterating over normal and attacked data to compare the rules based on the similarity score
norm=[]
for index1, row1 in sfd.iterrows():
    for index2, row2 in fnd.iterrows():
        if fuzz.token_set_ratio(row1["antecedents"], row2["antecedents"])>95 and fuzz.token_set_ratio(row1["consequents"], row2["consequents"])>95:    
            if index1!=index2:
                norm.append({"antecedents":row2["antecedents"],"consequents":row2["consequents"]})
        elif fuzz.token_set_ratio(row1["antecedents"], row2["consequents"])>95 and fuzz.token_set_ratio(row1["consequents"], row2["antecedents"])>95:
            if index1!=index2:
                norm.append({"antecedents":row2["antecedents"],"consequents":row2["consequents"]})

normies= pd.DataFrame(norm)
normies=normies.drop_duplicates()

#creating the dataframe for anomalous rules
merged = pd.merge(fnd, normies, on=['antecedents', 'consequents'], how='left', indicator=True)

#filter the rows that are only in 'atfil'
anomi = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
#reset index
anomi.reset_index(drop=True, inplace=True)
print(anomi[["antecedents","consequents"]])
