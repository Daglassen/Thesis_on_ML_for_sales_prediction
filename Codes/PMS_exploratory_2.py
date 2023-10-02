# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import janitor

############################################################################################################################################
################           1. DATA CLEANING           ######################################################################################
############################################################################################################################################


# IMPORT DATA
df = pd.read_excel('D:/Angelo/DTU/Thesis/Data/PMS_Data_SMC_IN.xlsx')

# CHECK COLUMNS
df.columns
df = df.drop(columns=['TBO acc. MAN (hrs)', 'Manager', 'JOB_TITLE', 'Comment'])

#CHECK TYPES AND CORRECT THEM
df.dtypes
df['NEXT_DUE_DATE'] = pd.to_datetime(df['NEXT_DUE_DATE'], format="%Y-%m-%d %H:%M:%S", errors = 'coerce' )

# TRIM TEXT
cols = df.select_dtypes(object).columns
df[cols] = df[cols].apply(lambda x: x.str.strip())


### 1. TAKE ROWS WITH NEXT DUE DATE IN 2022 AND CREATE NEW DATA
df_temp = df[(df['NEXT_DUE_DATE'] < '2022-12-31')]
# DROP COLUMNS LAST DATE
df_temp = df_temp.drop(columns=['LAST_DONE_DATE', 'LAST_DONE_HRS'])
# RENAME COLUMNS
df_temp = df_temp.rename(columns={"NEXT_DUE_DATE": "LAST_DONE_DATE", "NEXT_DUE_HRS": "LAST_DONE_HRS"})



### 2. APPEND NEW DATA TO MAIN DATAFRAME
# DELETE UNWANTED COLUMNS
df = df.drop(columns=['NEXT_DUE_DATE', 'NEXT_DUE_HRS'])
# APPEND TWO DATASETS
df = df.append(df_temp)


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


### 5: CREATE CYLINDER NUMBER AND ENGINE NUMBER COLUMN
# The column "EQUIPMENT_NAME" contains spaces sometimes after NO. - We remove those spaces
df['EQUIPMENT_NAME'] = df['EQUIPMENT_NAME'].str.replace('NO. ','NO.')

# Take the 2 numbers after NO.
df['TEMP_COL'] = df.EQUIPMENT_NAME.str.extract('\.(.{,2})') 
# split temporary column
df[['A', 'Cylinder_Number', 'Engine_Number', 'D']] = df['TEMP_COL'].str.split('',expand=True)




# Replace blank engine number with A
df['Engine_Number'] = df['Engine_Number'].replace(r'^\s*$', 'A', regex=True)
# Drop unwanted columns
df = df.drop(columns=['A', 'D', 'TEMP_COL'])


### 6. REMOVE ROWS THAT CONTAIN INDICATOR VALVE OR SAFETY VALVE (THEY ARE CONSIDERED OBSOLETE)
df = df[df["EQUIPMENT_NAME"].str.contains("INDICATOR VALVE") == False]
df = df[df["EQUIPMENT_NAME"].str.contains("SAFETY VALVE") == False]


### 7. CREATE MAINTENANCE YEAR
df['maintainance_year'] = df['LAST_DONE_DATE'].dt.year

### 6. REMOVE DUPLICATE ROWS
df = df.drop_duplicates(['VESSEL_NAME','Maintenance_Kit','Cylinder_Number', 'Engine_Number','maintainance_year', 'EQUIPMENT_CODE'], keep='last')


# SAVE DATAFRAME AS df1
df_preprocessed = df.copy()

############################################################################################################################################
################           1. PREPARE DATA FOR MODELS           ######################################################################################
############################################################################################################################################

df.dtypes


### 1. DROP NOT NEEDED COLUMNS
df = df.drop(columns=['EQUIPMENT_NAME', 'EQUIPMENT_CODE', 'LAST_DONE_DATE'])

### 2. ADD OUTPUT COLUMN (IF A MAINTENANCE OCCURED)
df['done'] = 1



### 3. ADD ROWS WITH ALL THE POSSIBLE MAINTENANE TYPES
df1 = df.copy()

df1 = df1.complete('VESSEL_NAME', 'Maintenance_Kit', 'Cylinder_Number','Engine_Number', 'maintainance_year').fillna(0, downcast='infer')
df1 = df1.sort_values(['VESSEL_NAME', 'Engine_Number', 'Cylinder_Number', 'maintainance_year'],
              ascending = [True, True, True, True])


#df1['Maintenance_Kit'] = pd.Categorical(df['Maintenance_Kit'], categories=df['Maintenance_Kit'].unique())
#df1 = df.groupby(['VESSEL_NAME', 'FREQUENCY','LAST_DONE_HRS','Maintenance_Kit', 'Cylinder_Number','Engine_Number', 'maintainance_year'], as_index=False).first()

#components = np.array(('Piston Ring CR', 'Stuff. Box MK','Fuel Pump', 'Alpha Lub', "Cyl. Liner", "Cyl. Cover",
#                      "Exh. Valve Act", "Press. Booster", "Fuel Valve (16.000 hrs)", "Accumulators ME", 
#                      "Proportional Valve HydrPump ME", "Hydr. Cyl. Unit", "Auxiliary Engine MK", "Turbocharger"))







