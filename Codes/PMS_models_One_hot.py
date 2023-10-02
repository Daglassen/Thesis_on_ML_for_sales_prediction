# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
LogReg=LogisticRegression()
LinReg=LinearRegression()

from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import janitor

############################################################################################################################################
################           1. DATA CLEANING           ######################################################################################
############################################################################################################################################


# IMPORT DATA
df = pd.read_excel('D:/Angelo/DTU/Thesis/Data/PMS_Data_SMC_IN.xlsx')

# CHECK COLUMNS
df.columns
df = df.drop(columns=['TBO acc. MAN (hrs)', 'Manager', 'JOB_TITLE', 'Comment', 'NEXT_DUE_DATE', 'NEXT_DUE_HRS'])

'''
#CHECK TYPES AND CORRECT THEM
df.dtypes
df['NEXT_DUE_DATE'] = pd.to_datetime(df['NEXT_DUE_DATE'], format="%Y-%m-%d %H:%M:%S", errors = 'coerce' )
'''
# TRIM TEXT
cols = df.select_dtypes(object).columns
df[cols] = df[cols].apply(lambda x: x.str.strip())

'''
### 1. TAKE ROWS WITH NEXT DUE DATE IN 2022 AND CREATE NEW DATA
df_temp = df[(df['NEXT_DUE_DATE'] < '2022-12-31')]
# DROP COLUMNS LAST DATE
df_temp = df_temp.drop(columns=['LAST_DONE_DATE', 'LAST_DONE_HRS'])
# RENAME COLUMNS
df_temp = df_temp.rename(columns={"NEXT_DUE_DATE": "LAST_DONE_DATE", "NEXT_DUE_HRS": "LAST_DONE_HRS"})
'''

'''
### 2. APPEND NEW DATA TO MAIN DATAFRAME
# DELETE UNWANTED COLUMNS
df = df.drop(columns=['NEXT_DUE_DATE', 'NEXT_DUE_HRS'])
# APPEND TWO DATASETS
df = df.append(df_temp)
'''
# Rename date column
df = df.rename(columns={"LAST_DONE_DATE": "Maintenance_Date", "LAST_DONE_HRS": "Running_Hours"})


### 3. DROP NON-OVERHAULS
df = (df[df['JOB_TYPE'] == 'Overhaul'] )
df = df.drop(columns=['JOB_TYPE'])


### 4. CREATE CONDITIONAL COLUMN (MAINTENANCE KIT) BASED ON EQUIPMENT
df['Maintenance_Kit'] = pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.03."), "Piston Ring CR", 
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.02."), "Stuff. Box MK",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.09.01.01."), "Fuel Pump",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.10.01.01"), "Alpha Lub",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.04."), "Cyl. Liner",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.01"), "Cyl. Cover",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.12.01"), "Exh. Valve Act",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.09.01.02"), "Press. Booster",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.09.02."), "Fuel Valve (16.000 hrs)",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.20.01.04."), "Accumulators ME",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.20.01.01."), "Proportional Valve HydrPump ME",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.20.01.03."), "Hydr. Cyl. Unit",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("651."), "Auxiliary Engine MK",
                                    pd.np.where(df.EQUIPMENT_CODE.str.contains("601.01.15.01"), "Turbocharger",
                                    "N/A"))))))))))))))


### 5: CREATE CYLINDER NUMBER AND INJECTOR NUMBER COLUMN
# The column "EQUIPMENT_NAME" contains spaces sometimes after NO. - We remove those spaces
df['EQUIPMENT_NAME'] = df['EQUIPMENT_NAME'].str.replace('NO. ','NO.')

# Take the 2 numbers after NO.
df['TEMP_COL'] = df.EQUIPMENT_NAME.str.extract('\.(.{,2})') 
# split temporary column
df[['A', 'Cylinder_Number', 'Injector_Number', 'D']] = df['TEMP_COL'].str.split('',expand=True)
# Drop unwanted columns
df = df.drop(columns=['A', 'D', 'TEMP_COL'])

# The number shown in turbochargers are the engine numbers
# Move those number to the correct column
#df = df.reset_index(drop=True)
#df.loc[df['EQUIPMENT_NAME'].str.contains('CHARGER'), 'Engine_Number'] = df['Cylinder_Number']
# Change formatting to the correct:
#df = df.replace({'Engine_Number': {'A': '1', 'B': '2', 'C':'3'}})    
# Replace blank engine number with A
#df['Engine_Number'] = df['Engine_Number'].replace(r'^\s*$', '1', regex=True)

# Specificaly for Zita and Zoe, remove C from cylinder cover rows
#df = df.loc[(df['VESSEL_NAME'] == 'Zita Schulte') & (df['Injector_Number'] == C), 'Injector_Number'] = 'NULL'
#df["Injector_Number"] = np.where((df["Injector_Number"] == "C") &  (df['VESSEL_NAME'] == 'Zita Schulte'), 0, df["Injector_Number"])
#df["Injector_Number"] = np.where((df["Injector_Number"] == "C") &  (df['VESSEL_NAME'] == 'Zoe Schulte'), 0, df["Injector_Number"])

df.isna().sum()



### 6. REMOVE ROWS THAT CONTAIN INDICATOR VALVE OR SAFETY VALVE (THEY ARE CONSIDERED OBSOLETE)
df = df[df["EQUIPMENT_NAME"].str.contains("INDICATOR VALVE") == False]
df = df[df["EQUIPMENT_NAME"].str.contains("SAFETY VALVE") == False]


### 7. CREATE MAINTENANCE YEAR
df['maintainance_year'] = df['Maintenance_Date'].dt.year

### 8. REMOVE DUPLICATE ROWS
df = df.drop_duplicates(['VESSEL_NAME','Maintenance_Kit','Cylinder_Number', 'Injector_Number','maintainance_year', 'EQUIPMENT_CODE'], keep='last')

### 9. THERE ARE A FEW ROWS WITH NAN (4) - WE REMOVE THEM 
df.isna().sum().sum()
df=df.dropna(axis=0)

### 10. VESSEL 'WESER STAHL' ONLY HAS 5 OBSERVVATIONS FROM 2015, REMOVE IT?
#df = df[df['VESSEL_NAME'] != 'Weser Stahl']

# SAVE DATAFRAME AS df1
df_preprocessed = df.copy()

############################################################################################################################################
################           2. PREPARE DATA FOR MODELS           ######################################################################################
############################################################################################################################################
df = df_preprocessed.copy()
df.dtypes
df.Injector_Number.unique()
df.nunique()
vessel_values = df.VESSEL_NAME.value_counts()
print(vessel_values)


### 0. WE WILL PREDICT EACH VALVE MAINTENANCE SEPARATELY
# ADD INJECTOR NUMBER TO MAINTENANCE KIT
df["Maintenance_Kit"] = df['Maintenance_Kit'].astype(str) +" "+ df["Injector_Number"]
# TRIM TEXT
cols = df.select_dtypes(object).columns
df[cols] = df[cols].apply(lambda x: x.str.strip())

### 1. DROP NOT NEEDED COLUMNS
df = df.drop(columns=['EQUIPMENT_NAME', 'EQUIPMENT_CODE', 'Injector_Number'])

### 2. ADD OUTPUT COLUMN (IF A MAINTENANCE OCCURED)
df['done'] = 1

### 3. DELETE MAINTENANCE FROM 2015 AND BEFORE (if I keep them I will have ca.190000 rows)
df = df[df['maintainance_year'] > 2015]  


### 4. REMOVE ALPHA LUB FROM PREDICTION (THEY OCCUR TOO FEW (2) TIMES - WE WON'T PREDICT THIS TYPE)
df = df[df['Maintenance_Kit'] != 'Alpha Lub']


### 5. ADD ROWS WITH ALL THE POSSIBLE MAINTENANE TYPES
df = df.complete('VESSEL_NAME', 'Maintenance_Kit', 'Cylinder_Number', 'maintainance_year').fillna(0, downcast='infer')
# Sort the data to have better view
df = df.sort_values(['VESSEL_NAME', 'Cylinder_Number', 'maintainance_year'],
             ascending = [True, True, True])

df = df.reset_index(drop=True)


### 6. THE CREATED ROWS EXCEED THE CORRECT AMOUNT OF ENGINES/CYLIDERS, WE HAVE TO DELETE THEM MANULLY
df = df.astype({"Cylinder_Number": int})
df = df[(df.VESSEL_NAME == 'Aqua Bonanza') & (df.Cylinder_Number <= 6) | 
        (df.VESSEL_NAME == 'Central') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Clamor Schulte') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Clover') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Donata Schulte') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Dorothea Schulte') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Hans Schulte') & (df.Cylinder_Number <= 8) |
        (df.VESSEL_NAME == 'Hedwig Schulte') & (df.Cylinder_Number <= 8) |
        (df.VESSEL_NAME == 'Hyde') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Jasper Dream') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Key Sonority') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Konrad Schulte') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'LILA II') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Largo Mariner') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'London Courage') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'MSC Agadir') & (df.Cylinder_Number <= 9) |
        (df.VESSEL_NAME == 'MSC Antigua') & (df.Cylinder_Number <= 9) |
        (df.VESSEL_NAME == 'Margarete Schulte') & (df.Cylinder_Number <= 8) |
        (df.VESSEL_NAME == 'Moritz Schulte') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'PPS Luck') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'PPS Salmon') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Thalea Schulte') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Thekla Schulte') & (df.Cylinder_Number <= 5) |
        (df.VESSEL_NAME == 'Theresa Schulte') & (df.Cylinder_Number <= 5) |
        (df.VESSEL_NAME == 'United Crown') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Weser Stahl') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Zita Schulte') & (df.Cylinder_Number <= 6) |
        (df.VESSEL_NAME == 'Zoe Schulte') & (df.Cylinder_Number <= 6) ]


df.describe()

### 7. CREATE COLUMN WITH AVG RUNNING HOURS
# WE ADD THE RUNNING HOUR INSTANCES PER VESSEL, PER YEAR AND DIVIDE BY THEIR NUMBER
df['Running_Hours'] = df.replace(0, np.nan).groupby(['VESSEL_NAME', 'maintainance_year'])['Running_Hours'].transform('mean') 

df['Running_Hours'].isna().sum()


### 8. FILL NA VALUES WITH AVG OF PREVIOUS AND NEXT
# rearrange the dataframe
df = df.sort_values(['VESSEL_NAME', 'maintainance_year'],
              ascending = [True, True])



# fill na in running hours with the mean of previous-next years, per vessel
df = (df.assign(ffill = df.groupby('VESSEL_NAME').Running_Hours.ffill(),
                 bfill = df.groupby('VESSEL_NAME').Running_Hours.bfill(),
                 both  = df.groupby('VESSEL_NAME').Running_Hours.ffill().add(df.Running_Hours.bfill()),
                 fin = df.groupby('VESSEL_NAME').Running_Hours.ffill().add(df.Running_Hours.bfill()).div(2)))
df['Running_Hours'] = df['Running_Hours'].fillna(df['fin'])



#  for the vessels that still don't have, we apply the running hours from the next year 
df.loc[df['maintainance_year'] == 2016 ,'Running_Hours'] = df.loc[df['maintainance_year'] == 2016,'Running_Hours'].fillna(value = (df['bfill']))
df.loc[df['maintainance_year'] == 2017 ,'Running_Hours'] = df.loc[df['maintainance_year'] == 2017,'Running_Hours'].fillna(value = df['bfill'])
df.loc[df['maintainance_year'] == 2018 ,'Running_Hours'] = df.loc[df['maintainance_year'] == 2018,'Running_Hours'].fillna(value = df['bfill'])
df.loc[df['maintainance_year'] == 2019 ,'Running_Hours'] = df.loc[df['maintainance_year'] == 2019,'Running_Hours'].fillna(value = df['bfill'])

df.dtypes
### 9. DROP NOT NEEDED COLUMNS
df = df.drop(columns=['ffill', 'bfill', 'both', 'fin', 'Maintenance_Date', 'FREQUENCY'])

# SAVE PROGRESS
df1 = df.copy()


############################################################################################################################################
################           3. PRE-PROCESSING (ONE HOT ENCODING)        ######################################################################################
############################################################################################################################################
'''
delete weser Stahl???
'''


df = df1.copy()
df.dtypes
# change data types
df = df.astype({"Cylinder_Number": object, "maintainance_year": object})



### 1. ONE-OUT-OF K ENCODING
df = pd.get_dummies(df, prefix = '', prefix_sep='')


### 2. SPLIT AND SHUFFLE DATA
test_proportion = 0.25;
data_train, data_test = train_test_split(df,test_size=test_proportion, random_state=2, shuffle=True);
X_train = data_train.drop('done', axis=1)
X_test = data_test.drop('done', axis=1)
y_train = data_train['done']
y_test = data_test['done']


### 3. OFFSET


### 4. STANDARDIZE TEST/TRAIN SETS
X_train['Running_Hours'] = MinMaxScaler().fit_transform(np.array(X_train['Running_Hours']).reshape(-1,1))
X_test['Running_Hours'] = MinMaxScaler().fit_transform(np.array(X_test['Running_Hours']).reshape(-1,1))



############################################################################################################################################
################           3. TRAIN-TEST MODELS        ######################################################################################
############################################################################################################################################


### 1. LINEAR REGRESSION
LogReg.fit(X_train, y_train)
y_pred_log = LogReg.predict(X_test)

ScoreLogReg = LogReg.score(X_test,y_test)
cmLogReg = metrics.confusion_matrix(y_test, y_pred_log)
metricsLogReg = metrics.classification_report(y_test, y_pred_log)



### 2. RANDOM FOREST
clf=RandomForestClassifier(n_estimators=300)
clf.fit(X_train,y_train)
y_pred_RF=clf.predict(X_test)

ScoreRF = metrics.accuracy_score(y_test, y_pred_RF)
cmRF = metrics.confusion_matrix(y_test, y_pred_RF)
metricsRF = metrics.classification_report(y_test, y_pred_RF)



### 3. BERNOULI NAIVE BAYES
clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred_BNB = clf.predict(X_test)

ScoreBNB = metrics.accuracy_score(y_test, y_pred_BNB)
cmBNB = metrics.confusion_matrix(y_test, y_pred_BNB)
metricsRandForest = metrics.classification_report(y_test, y_pred_RF)



### 3. FOREST WITH ADABOOST
clf = AdaBoostClassifier(n_estimators=300)
clf.fit(X_train,y_train)
y_pred_ADA = clf.predict(X_test)

scores = cross_val_score(clf, X_train, y_train, cv=5)
ScoreADA = scores.mean()
cmADA = metrics.confusion_matrix(y_test, y_pred_ADA)
metricsADA = metrics.classification_report(y_test, y_pred_ADA)



### 4. GRADIENT BOOST CLASSIFIER
clf = GradientBoostingClassifier(n_estimators=300)
clf.fit(X_train, y_train)
y_pred_GBC = clf.predict(X_test)

ScoreGBC = metrics.accuracy_score(y_test, y_pred_GBC)
cmGBC = metrics.confusion_matrix(y_test, y_pred_GBC)
metricsGBC = metrics.classification_report(y_test, y_pred_GBC)



### 5. XGBOOST CLASSIFIER
clf = xgb.XGBClassifier(objective="binary:logistic")

clf.fit(X_train, y_train)
y_pred_XGB = clf.predict(X_test)

ScoreXGB = metrics.accuracy_score(y_test, y_pred_XGB)
cmXGB = metrics.confusion_matrix(y_test, y_pred_XGB)
metricsXGB = metrics.classification_report(y_test, y_pred_XGB)



### 6. NEURAL NETWORK

# SET INPUT/OUTPUT
Y = y_train
X = X_train


# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)



########  BUILD THE MODEL
model = Sequential()
model.add(Dense(500, input_shape=(X.shape[1],), activation='relu')) # Add an input shape! (features,)
model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary() 

# compile the model
model.compile(optimizer='Adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# early stopping callback
# This callback will stop the training when there is no improvement in the validation loss for 10 consecutive epochs.  
es = EarlyStopping(monitor='val_accuracy', 
                                   mode='max', 
                                   patience=10,
                                   restore_best_weights=True)

# now we just update our model fit call
history = model.fit(X,
                    Y,
                    callbacks=[es],
                    epochs=80, # I can set this higher
                    batch_size=10,
                    validation_split=0.25,
                    shuffle=True,
                    verbose=1)




######## EVALUATE THE MODEL
history_dict = history.history
# Learning curve(Loss) training and validation loss by epoch

# loss
loss_values = history_dict['loss'] # you can change this
val_loss_values = history_dict['val_loss'] # you can also change this

# range of X (no. of epochs)
epochs = range(1, len(loss_values) + 1) 

# plot
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

######## PLOT ACCURACY
# Learning curve(accuracy)
# accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# this is the max value - should correspond to the HIGHEST train accuracy
np.max(val_acc)



######## PRINT CONFUSION MATRIX AND CLASSIFICATION REPORT
model.predict(X_test) # prob of successes (done)
np.round(model.predict(X_test),0) # 1 and 0 (done or not)


# Round predictions to a whole number (0 or 1)
preds = np.round(model.predict(X_test),0)

# confusion matrix
cmNN = metrics.confusion_matrix(y_test, preds) 
# [3851	 142]
# [197	 340]

# classification report
print(metrics.classification_report(y_test, preds))













