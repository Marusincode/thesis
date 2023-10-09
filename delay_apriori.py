#apriori on 'run_1'

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('C:/Users/User/desktop/apriori/run_1.csv')

te = TransactionEncoder()
te_ary = te.fit(df).transform(df)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
frequent_itemsets.to_csv('association_rules.csv', index=False)
