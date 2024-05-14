#DELAY/LATITUDE/LONGITUDE/SAFE APRIORI

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import csv
import time

#to calculate processing time
t1=time.time()
#load the dataset
df = pd.read_csv('C:/Users/User/Desktop/THESIS/Datasets/run1_delay.csv')
#df = pd.read_csv('C:/Users/User/Desktop/THESIS/Datasets/run1_latitude_.csv')
#df = pd.read_csv('C:/Users/User/Desktop/THESIS/Datasets/run1_longitude.csv')
#df = pd.read_csv('C:/Users/User/Desktop/THESIS/Datasets/run1_safe.csv')

#remove attack indicators and time attributes
df = df.drop('is_attack', axis=1)
df = df.drop('Time', axis=1)
df = df.drop('Cyber_weight', axis=1)

#select specific columns for mining associations
data = df[['AV_x', 'AV_y', 'AV_steer', 'AV_vel', 'AV_yaw', 'npc_x', 'npc_y', 'Rollout_num']]

#perform one-hot encoding on the selected columns
df_encoded = pd.get_dummies(data, columns=['AV_x', 'AV_y', 'AV_steer', 'AV_vel', 'AV_yaw', 'npc_x', 'npc_y', 'Rollout_num'])

#apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

#derive association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)
rules.to_csv('C:/Users/User/Desktop/THESIS/Association rule mining/apriori/Delay apriori/rulesrun1delay.csv', index=False)
#rules.to_csv('C:/Users/User/Desktop/THESIS/Association rule mining/apriori/Latitude apriori/rulesrun1lat.csv', index=False)
#rules.to_csv('C:/Users/User/Desktop/THESIS/Association rule mining/apriori/Longitude apriori/rulesrun1lon.csv', index=False)
#rules.to_csv('C:/Users/User/Desktop/THESIS/Association rule mining/apriori/Safe apriori/rulesrun1safe.csv', index=False)

#time calculation
t2=time.time()
print(t2-t1)

#DELAY/LATITUDE/LONGITUDE/SAFE FP-GROWTH

import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
import csv
import time

t1=time.time()
df = pd.read_csv('C:/Users/User/Desktop/THESIS/Datasets/run1_delay.csv')
#df = pd.read_csv('C:/Users/User/Desktop/THESIS/Datasets/run1_latitude_.csv')
#df = pd.read_csv('C:/Users/User/Desktop/THESIS/Datasets/run1_longitude.csv')
#df = pd.read_csv('C:/Users/User/Desktop/THESIS/Datasets/run1_safe.csv')

data = df[['AV_x', 'AV_y', 'AV_steer', 'AV_vel', 'AV_yaw', 'npc_x', 'npc_y', 'Rollout_num']]
df_encoded = pd.get_dummies(data, columns=['AV_x', 'AV_y', 'AV_steer', 'AV_vel', 'AV_yaw', 'npc_x', 'npc_y', 'Rollout_num'])

frequent_itemsets=fpgrowth(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

rules.to_csv('C:/Users/User/Desktop/THESIS/Association rule mining/fpgrowth/Delay fpgrowth/delay_fpgrowthRUN1.csv', index = False)
#rules.to_csv('C:/Users/User/Desktop/THESIS/Association rule mining/fpgrowth/Latitude fpgrowth/latitude_fpgrowthRUN1.csv', index = False)
#rules.to_csv('C:/Users/User/Desktop/THESIS/Association rule mining/fpgrowth/Longitude fpgrowth/longitude_fpgrowthRUN2.csv', index = False)
#rules.to_csv('C:/Users/User/Desktop/THESIS/Association rule mining/fpgrowth/Safe fpgrowth/safe_fpgrowthRUN1.csv', index = False)

t2=time.time()
print(t2-t1)


