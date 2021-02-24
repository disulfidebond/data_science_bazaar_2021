import pandas as pd
import numpy as np
import xgboost
import random
from sklearn.metrics import accuracy_score

df = pd.read_csv('COVID-19_Historical_Data_by_State.orig.csv')

df = df.sort_values('DATE')
df = df.reset_index(drop=True)

# repeat for the values "POS_FEM_NEW", "POS_MALE_NEW", "HOSP_UNK_NEW",
df['HOSP_NEW'] = df['HOSP_YES'].diff()

# setup labels
df['SCAN'] = df['POS_NEW'].diff(periods=-1)
df['INCREASE'] = -1
df.loc[df['SCAN'] < 0, ['INCREASE']] = 1
df.loc[df['SCAN'] >= 0, ['INCREASE']] = 0

df = df.iloc[20:]
# df_filtered = df[["POS_NEW", "POS_7DAYAVG", "NEG_NEW", "NEG_7DAYAVG", "DTH_NEW", "DTH_7DAYAVG", "TEST_NEW", "TEST_7DAYAVG", "HOSP_NEW", "HOSP_NO_NEW", "HOSP_UNK_NEW", "POS_FEM_NEW", "POS_MALE_NEW", "INCREASE"]]
df_filtered.drop(df.tail(1).index, inplace=True)
df_filtered = df[["POS_NEW", "POS_7DAYAVG", "NEG_NEW", "NEG_7DAYAVG", "DTH_NEW", "DTH_7DAYAVG", "TEST_NEW", "TEST_7DAYAVG", "HOSP_NEW", "INCREASE"]]
df_filtered



df_split = np.random.rand(len(df_filtered)) < 0.8))
df_training = df_filtered[df_split]
df_testing = df_filtered[~df_split]
df_training_y = df_training['INCREASE']
df_training_x = df_training
del df_training_x['INCREASE']
df_training_y = df_training_y.tolist()

df_testing_y = df_testing['INCREASE']
df_testing_x = df_testing
del df_testing_x['INCREASE']
df_testing_y = df_testing_y.tolist()

model = xgboost.XGBClassifier()
model.fit(df_training_x, df_training_y)

y_pred = model.predict(df_testing_x)


predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(df_testing_y, predictions)
print("Accuracy is %.2f%%" % (accuracy * 100.0))
