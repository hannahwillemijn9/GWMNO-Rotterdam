#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 08:39:17 2024

@author: willemijnelzinga
"""

#%% Task: Excel to Dataframe. 

import os 
import pandas as pd
import numpy as np
import geopandas as gpd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors
import contextily as ctx
import pysensors as ps 

file = 'Heijplaat meetgegevens1.xlsx'
df = pd.read_excel(file, sheet_name = 'PRW_Peilbuis_Meetgegevens')
df_hdr = pd.read_excel(file, sheet_name = 'PRW_Peilbuizen')
 
df01 = pd.DataFrame()
for i, col in enumerate(df):
    if df.loc[0, col] == 'DATUM_METING':
        df02 = pd.DataFrame(df.loc[1:, col].values, columns = ['date'])
        df02['date'] = pd.to_datetime(df02['date'], format = '%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')
        df02['meting'] = df.loc[1:,df.columns[df.columns.get_indexer([col])+1][0]].values
        df02['meetpunt'] = df.columns[df.columns.get_indexer([col])+1][0]
        df02 = df02[df02['date'].notna()]
        df02['x'] = df_hdr['X_COORDINAAT'][df_hdr['PEILBUIS'] == 
                                           df.columns[df.columns.get_indexer([col])+1][0]].values[0]
        df02['y'] = df_hdr['Y_COORDINAAT'][df_hdr['PEILBUIS'] == 
                                           df.columns[df.columns.get_indexer([col])+1][0]].values[0]
        df02 = df02[['meetpunt', 'x', 'y', 'date', 'meting']]
        df01 = pd.concat([df01, df02])

df = df01.copy()

del file, i, col, df02

#%% Task: CSV to Dataframe. 

import pandas as pd

df01 = pd.read_csv('prw_meetgegevens1.csv', dtype = str, sep = ';') 
df01 = df01.rename(columns = {'ID' : 'ID',
                              'Peilbuis-volgnummer' : 'meetpunt',
                              'Waarneming' : 'waarneming',
                              'Metingsdatum / tijd' : 'date',
                              'Meetwaarde(m NAP)' : 'meting',
                              'Hoogte meetmerk(m NAP)' : 'meethoogte'}) 
df01['date'] = pd.to_datetime(df01['date'], format = '%d-%m-%Y %H:%M:%S').dt.strftime('%Y-%m-%d')
df01 = df01[['ID', 'meetpunt', 'date', 'meting', 'meethoogte', 'waarneming']]

df01['meting'] = pd.to_numeric(df01['meting'].str.replace(',','.'), errors='coerce') 
df01['meethoogte'] = pd.to_numeric(df01['meethoogte'].str.replace(',','.'), errors='coerce') 
df01 = df01[['ID','meetpunt','date','meting','meethoogte', 'waarneming']]

prw_meetgegevens = df01.copy()
del df01

merge_df = pd.merge(df, prw_meetgegevens, how='left', on=['meetpunt', 'date', 'meting'])
print(merge_df)

merge_df['date'] = pd.to_datetime(merge_df['date'], format='%Y-%m-%d')
start_date = '2010-01-01'
end_date = '2024-01-01'
merge_df = merge_df[(merge_df['date'] >= start_date) & (merge_df['date'] <= end_date)]

#%% Task: Pastas basic model. Figures: all variables, recharge, stress models. 

#Step 1: Figure with unique colors corresponding to monitoring wells. 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

merge_df['date'] = pd.to_datetime(merge_df['date'])
if not isinstance(merge_df.index, pd.DatetimeIndex):
    merge_df = merge_df.set_index('date')

unique_wells = merge_df['meetpunt'].unique()

custom_palette = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Lime Green
    "#17becf",  # Cyan
    "#aec7e8",  # Light Blue
    "#ffbb78",  # Light Orange
    "#98df8a",  # Light Green
    "#ff9896",  # Light Red
]

palette = sns.color_palette(custom_palette)[:len(unique_wells)]


plt.figure(figsize=(12, 6))
sns.scatterplot(data=merge_df, x='date', y='meting', hue='meetpunt', style='meetpunt', markers=True, s=20, palette=palette)

plt.ylabel('Groundwater level [m MSL]')
plt.xlabel('Time [days]')
plt.title('Monitoring Wells Heijplaat')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Monitoring Well')

plt.show()


# Step 2 

f = open('.//KNMI Rotterdam.txt', 'r')
while True: 
    line = f.readline()
    if '# STN,' in line: 
        break 
line = line.replace('#', '').replace('\n','').replace(' ','').split(',')

df_knmi = pd.read_csv('.//KNMI Rotterdam.txt', names=line, sep=',', comment='#')

print("Columns in df_knmi:", df_knmi.columns)

df_knmi['date'] = pd.to_datetime(df_knmi['YYYYMMDD'], format='%Y%m%d')
df_knmi = df_knmi[df_knmi['date'] > '2010-01-01']

if 'RH' in df_knmi.columns:
    df_knmi['RH'] = pd.to_numeric(df_knmi['RH'], errors='coerce')

if 'EV' in df_knmi.columns:
    df_knmi['EV24'] = pd.to_numeric(df_knmi['EV24'], errors='coerce')

df_knmi = df_knmi.set_index(pd.DatetimeIndex(df_knmi['date']))

precip = df_knmi[['RH']].copy()
precip['RH'][precip['RH'] <0.] = 0 
precip['RH'] = precip['RH']/10000 

evap = df_knmi[['EV24']].copy()
evap['EV24'] = pd.to_numeric(evap['EV24'], errors='coerce')
evap['EV24'] = evap['EV24']/10000


recharge = precip['RH'] - evap['EV24']
recharge = recharge.to_frame()
recharge.rename(columns={0: 'Recharge'}, inplace=True)

recharge.plot(label='Recharge', figsize=(10,4), color='#1f77b4')
plt.ylabel('Recharge [m/day]')
plt.xlabel('Time [days]')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Recharge Weather Station 344 Rotterdam')
plt.tight_layout()
plt.show()

## EVAP and P fig

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(precip.index, precip['RH'], label='Precipitation', color='#1f77b4')

ax.set_ylabel('Precipitation [m/day]')
ax.set_xlabel('Time [days]')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title('Precipitation Weather Station 344 Rotterdam')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(evap.index, evap['EV24'], label='Evaporation', color='#1f77b4')

ax.set_ylabel('Precipitation [m/day]')
ax.set_xlabel('Time [days]')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title('Precipitation Weather Station 344 Rotterdam')
plt.tight_layout()
plt.show()


# Step 3
import pastas as ps 

heijplaat = merge_df.copy()

for meetpunt in heijplaat['meetpunt'].unique():
    heijplaat1 = heijplaat[heijplaat['meetpunt'] == meetpunt]
    
    if not heijplaat1.empty:
        heijplaat1['meting'] = pd.to_numeric(heijplaat1['meting'], errors='coerce').fillna(method='bfill')
        print(heijplaat1[heijplaat1['meting'].isna()])
        
        heijplaat1.sort_index(inplace=True)
        heijplaat1 = heijplaat1.drop(columns=['x', 'y', 'ID', 'meethoogte'])
        
        ml1 = ps.Model(heijplaat1['meting'], name=f'ml1_{meetpunt}')
        
        sm1 = ps.StressModel(recharge['Recharge'], ps.Gamma(), name='recharge', settings='evap')
        ml1.add_stressmodel(sm1)
        
        ml1.solve()
        ml1.plot()
        plt.xlabel('Date [days]')
        plt.ylabel('Groundwater Level [m MSL]')
        plt.title(f'Ml1: Stress model based on recharge {meetpunt}')
        plt.show()
        
        
        ml1.plots.results(figsize=(10, 6))
        plt.suptitle(f'Ml1: Stress model based on recharge for {meetpunt}', y=1.02)
        plt.show()
        print(ml1.stats.summary())
        
        ml2 = ps.Model(heijplaat1['meting'], name=f'ml2_{meetpunt}')
        
        sm2 = ps.RechargeModel(precip['RH'], evap['EV24'], ps.Gamma(), name='Recharge', 
                               recharge=ps.rch.Linear(), settings=('prec', 'evap'))
        ml2.add_stressmodel(sm2)
        
        ml2.solve()
        ml2.plot()
        plt.xlabel('Time [days]')
        plt.ylabel('Groundwater Level [m MSL]')
        plt.title(f'Ml2: Stress model based on evaporation factor for {meetpunt}')
        plt.show()
    

#%% Task: Backcasting and plotting for unique variables only for Datalogger. 

import pandas as pd
import matplotlib.pyplot as plt
import pastas as ps

# Veronderstel dat merge_df, recharge, etc. al gedefinieerd zijn

hp = merge_df[merge_df['waarneming'] != 'Gemeten met datalogger']
dl = merge_df[merge_df['waarneming'] == 'Gemeten met datalogger']

combi = list(set(hp['meetpunt'].unique()).union(set(dl['meetpunt'].unique())))

start_backcast = pd.to_datetime('2010-01-01')
end_backcast = pd.to_datetime('2024-01-01')

all_summaries1 = pd.DataFrame()

for meetpunt in combi:
    dl1 = dl[dl['meetpunt'] == meetpunt].copy()
    if not dl1.empty:
        # Zet de index om naar datetime, als deze nog niet in dat formaat is.
        # Dit is alleen nodig als de index nog geen DatetimeIndex is.
        dl1.index = pd.to_datetime(dl1.index)

        # Aannemend dat je de overbodige kolommen wilt verwijderen.
        dl1 = dl1.drop(columns=['meetpunt', 'x', 'y', 'ID', 'meethoogte'], errors='ignore')

        # Zet 'meting' om naar numerieke waarden
        dl1['meting'] = pd.to_numeric(dl1['meting'], errors='coerce')

        # Maak en los het model op met pastas
        ml9 = ps.Model(dl1['meting'], name=f'meting_ml9 {meetpunt}')
        sm9 = ps.StressModel(recharge['Recharge'], ps.Gamma(), name='recharge', settings='evap')
        ml9.add_stressmodel(sm9)
        ml9.solve()

        # Simuleer en plot de backcasting
        backcast_datalogger = ml9.simulate(tmin=start_backcast, tmax=end_backcast)
        backcast_dates = pd.date_range(start=start_backcast, end=end_backcast, freq='D')
        plt.figure(figsize=(12, 6))
        plt.plot(backcast_dates, backcast_datalogger.reindex(backcast_dates), label='Datalogger', color='#1f77b4')
        plt.xlabel('Time [days]')
        plt.ylabel('Groundwater Level [m MSL]')
        plt.title(f'Backcasting Datalogger: {meetpunt}')
        plt.legend()
        plt.show()

        # Verzamel en voeg de samenvattingen toe aan het overzichts DataFrame
        summl9 = ml9.stats.summary()
        summl9['Model'] = 'Datalogger'
        summl9['meetpunt'] = meetpunt  # Gebruik de werkelijke meetpuntwaarde
        summl9 = summl9.reset_index()
        all_summaries1 = pd.concat([all_summaries1, summl9], ignore_index=True)


            
#%% Task: Backcasting and plotting only for handpeiling/manual. 


import pandas as pd
import matplotlib.pyplot as plt
import pastas as ps


hp = merge_df[merge_df['waarneming'] != 'Gemeten met datalogger']
dl = merge_df[merge_df['waarneming'] == 'Gemeten met datalogger']

combi = list(set(hp['meetpunt'].unique()).union(set(dl['meetpunt'].unique())))

start_backcast = pd.to_datetime('2010-01-01')
end_backcast = pd.to_datetime('2024-01-01')

all_summaries2 = pd.DataFrame()

for meetpunt in combi:
    hp1 = hp[hp['meetpunt'] == meetpunt].copy()
    if not hp1.empty:
        hp1.index = pd.to_datetime(hp1.index) if 'date' not in hp1.columns else pd.to_datetime(hp1['date'])
        if 'date' in hp1.columns:
            hp1.set_index('date', inplace=True)
        
        hp1 = hp1.drop(columns=['meetpunt', 'x', 'y', 'ID', 'meethoogte'], errors='ignore')

        hp1['meting'] = pd.to_numeric(hp1['meting'], errors='coerce')

        ml7 = ps.Model(hp1['meting'], name=f'meting_ml7{meetpunt}')
        sm7 = ps.StressModel(recharge['Recharge'], ps.Gamma(), name='recharge', settings='evap')
        ml7.add_stressmodel(sm7)
        ml7.solve()

        summl7 = ml7.stats.summary().reset_index()
        summl7['Model'] = 'Manual'
        summl7['meetpunt'] = meetpunt  

        backcast_manual = ml7.simulate(tmin=start_backcast, tmax=end_backcast)
        backcast_dates = pd.date_range(start=start_backcast, end=end_backcast, freq='D')
        plt.figure(figsize=(12, 6))
        plt.plot(backcast_dates, backcast_manual.reindex(backcast_dates), label='Manual', color='orange')
        plt.xlabel('Time [days]')
        plt.ylabel('Groundwater Level [m MSL]')
        plt.title(f'Backcasting Manual: {meetpunt}')
        plt.legend()
        plt.show()

        all_summaries2 = pd.concat([all_summaries2, summl7], ignore_index=True)


#%% Task: Backcasting and plotting for unique variables both Manual and Datalogger with statistics to Excel. No figure. 

import pandas as pd
import matplotlib.pyplot as plt
import pastas as ps

def prepare_and_model(data, meetpunt, recharge, model_name):
    """Bereidt data voor en past model toe."""
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    else:
        data.index = pd.to_datetime(data.index)
    
    data = data.drop(columns=['meetpunt', 'x', 'y', 'ID', 'meethoogte', 'waarneming'], errors='ignore')
    data['meting'] = pd.to_numeric(data['meting'], errors='coerce')
    
    model = ps.Model(data['meting'], name=model_name)
    stress_model = ps.StressModel(recharge['Recharge'], ps.Gamma(), name='recharge', settings='evap')
    model.add_stressmodel(stress_model)
    model.solve()
    
    return model

all_summaries = pd.DataFrame()

for meetpunt in combi:
    if meetpunt in hp['meetpunt'].values:
        model_hp = prepare_and_model(hp[hp['meetpunt'] == meetpunt].copy(), meetpunt, recharge, f'meting_ml7{meetpunt}')
        summl_hp = model_hp.stats.summary().reset_index()
        summl_hp['Model'] = 'Manual'
        summl_hp['meetpunt'] = meetpunt
        all_summaries = pd.concat([all_summaries, summl_hp])
    
    if meetpunt in dl['meetpunt'].values:
        model_dl = prepare_and_model(dl[dl['meetpunt'] == meetpunt].copy(), meetpunt, recharge, f'meting_ml9 {meetpunt}')
        summl_dl = model_dl.stats.summary().reset_index()
        summl_dl['Model'] = 'Datalogger'
        summl_dl['meetpunt'] = meetpunt
        all_summaries = pd.concat([all_summaries, summl_dl])
    
    # Combineer simulaties en plot indien beide datasets bestaan
    if meetpunt in hp['meetpunt'].values and meetpunt in dl['meetpunt'].values:
        backcast_handpeiling = model_hp.simulate(tmin=start_backcast, tmax=end_backcast)
        backcast_datalogger = model_dl.simulate(tmin=start_backcast, tmax=end_backcast)
        backcast_dates = pd.date_range(start=start_backcast, end=end_backcast, freq='D')
        
        plt.figure(figsize=(12, 6))
        plt.plot(backcast_dates, backcast_handpeiling, label='Manual', color='orange')
        plt.plot(backcast_dates, backcast_datalogger, label='Datalogger', color='#1f77b4')
        plt.xlabel('Time [days]')
        plt.ylabel('Groundwater Level [m MSL]')
        plt.title(f'Comparison backcasting manual and datalogger: {meetpunt}')
        plt.legend()
        plt.show()


#%% Task: Bar plot of 'all_summaries' dataframe, based on simulated data. Figure: Bar plot. 

# Bar plot: RMSE and R2 with datalogger and manual
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools  

barplot = all_summaries[all_summaries['Statistic'].isin(['rmse', 'rsq'])]

plt.figure(figsize=(12, 6))

dodge_val = 0.2

unique_meetpunt = barplot['meetpunt'].unique()
unique_models = barplot['Model'].unique()
unique_stats = barplot['Statistic'].unique()

positions = np.arange(len(unique_meetpunt))

# Create a pivot table for proper alignment of data
pivot_data = barplot.pivot_table(index='meetpunt', columns=['Model', 'Statistic'], values='Value', fill_value=0)

for i, (model, stat) in enumerate(itertools.product(unique_models, unique_stats)):
    model_stat_data = pivot_data[(model, stat)].reindex(unique_meetpunt, fill_value=0)
    plt.bar(positions + i * dodge_val, model_stat_data, label=f'{model} - {stat}', width=dodge_val)

plt.xticks(positions + dodge_val * (len(unique_models) * len(unique_stats) - 1) / 2, unique_meetpunt, rotation=45)

plt.xlabel('Monitoring Well')
plt.ylabel('Value')
plt.title('RMSE and R2 by Monitoring Well: Datalogger and Manual')
plt.legend(title='Model - Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()


    
# Bar plot: EVP with datalogger and manual 
    
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools  

barplot2 = all_summaries[all_summaries['Statistic'].isin(['evp'])]

plt.figure(figsize=(12, 6))

dodge_val = 0.2

unique_meetpunt2 = barplot2['meetpunt'].unique()
unique_models2 = barplot2['Model'].unique()
unique_stats2 = barplot2['Statistic'].unique()

positions = np.arange(len(unique_meetpunt2))

pivot_data = barplot2.pivot_table(index='meetpunt', columns=['Model', 'Statistic'], values='Value', fill_value=0)

for i, (model, stat) in enumerate(itertools.product(unique_models2, unique_stats2)):
    model_stat_data = pivot_data[(model, stat)].reindex(unique_meetpunt2, fill_value=0)
    plt.bar(positions + i * dodge_val, model_stat_data, label=f'{model} - {stat}', width=dodge_val)

plt.xticks(positions + dodge_val * (len(unique_models2) * len(unique_stats2) - 1) / 2, unique_meetpunt2, rotation=45)

plt.xlabel('Monitoring Well')
plt.ylabel('Value')
plt.title('EVP by Monitoring Well: Datalogger and Manual')
plt.legend(title='Model - Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()



# Bar plot: RMSE and R2 for every well

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools

barplot = all_summaries[all_summaries['Statistic'].isin(['rmse', 'rsq'])]

plt.figure(figsize=(12, 6))

dodge_val = 0.2  

unique_meetpunt = barplot['meetpunt'].unique()
unique_stats = barplot['Statistic'].unique()

positions = np.arange(len(unique_meetpunt))

pivot_data = barplot.pivot_table(index='meetpunt', columns='Statistic', values='Value', aggfunc=np.mean, fill_value=0)

for i, stat in enumerate(unique_stats):
    stat_data = pivot_data[stat].reindex(unique_meetpunt, fill_value=0)
    plt.bar(positions + i * dodge_val, stat_data, label=stat, width=dodge_val)

plt.xticks(positions + dodge_val * (len(unique_stats) - 1) / 2, unique_meetpunt, rotation=45)

plt.xlabel('Monitoring Well')
plt.ylabel('Value')
plt.title('RMSE and R2 by Monitoring Well')
plt.legend(title='Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()



# Bar plot: EVP for every well 

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools 

barplot2 = all_summaries[all_summaries['Statistic'].isin(['evp'])]

plt.figure(figsize=(12, 6))

dodge_val = 0.2

unique_meetpunt2 = barplot2['meetpunt'].unique()
# unique_models2 = barplot2['Model'].unique()
unique_stats2 = barplot2['Statistic'].unique()

positions = np.arange(len(unique_meetpunt2))

pivot_data = barplot2.pivot_table(index='meetpunt', columns=['Model', 'Statistic'], values='Value', fill_value=0)

for i, (model, stat) in enumerate(itertools.product(unique_models2, unique_stats2)):
    model_stat_data = pivot_data[(model, stat)].reindex(unique_meetpunt2, fill_value=0)
    plt.bar(positions + i * dodge_val, model_stat_data, label=f'{model} - {stat}', width=dodge_val)

plt.xticks(positions + dodge_val * (len(unique_models2) * len(unique_stats2) - 1) / 2, unique_meetpunt2, rotation=45)

plt.xlabel('Monitoring Well')
plt.ylabel('Value')
plt.title('EVP by Monitoring Well')
plt.legend(title='Model - Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

#%% Task: Statistical test to determine the significant difference between 'Datalogger' and 'Manual'. No figure. 

# Adjust alpha to 0.05, meaning a 10% chance of significance 

import pandas as pd
from scipy import stats

statint = ['rmse', 'rsq', 'evp']  

alpha = 0.05  

for statspec in statint:
    filtered_summaries = all_summaries[all_summaries['Statistic'] == statspec]

    dlstat = filtered_summaries[filtered_summaries['Model'] == 'Datalogger']['Value']
    hpstat = filtered_summaries[filtered_summaries['Model'] == 'Manual']['Value']

    t_stat, p_value = stats.ttest_ind(dlstat, hpstat, equal_var=False)  # Welch's t-test

    print(f"\nT-test for {statspec} statistic:")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    if p_value < alpha:
        print(f"There is a significant difference between Datalogger and Manual for the {statspec} statistic (p < {alpha}).")
    else:
        print(f"There is no significant difference between Datalogger and Manual for the {statspec} statistic (p >= {alpha}).")

                
#%% Task: Revised merging observation and simulation data only for DATALOGGER. 


import pandas as pd
import pastas as ps


dl = merge_df[merge_df['waarneming'] == 'Gemeten met datalogger']
combi = dl['meetpunt'].unique()
start_backcast = pd.to_datetime('2010-01-01')
end_backcast = pd.to_datetime('2024-01-01')

combined_data = pd.DataFrame()

for meetpunt in combi:
    data = dl[dl['meetpunt'] == meetpunt].copy()
    if not data.empty:
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        else:
            data.index = pd.to_datetime(data.index)
        
        data = data.drop(columns=['meetpunt', 'x', 'y', 'ID', 'meethoogte', 'waarneming'], errors='ignore')
        data['meting'] = pd.to_numeric(data['meting'], errors='coerce')
        
        model = ps.Model(data['meting'], name=f'meting {meetpunt}')
        stress_model = ps.StressModel(recharge['Recharge'], ps.Gamma(), name='recharge', settings='evap')
        model.add_stressmodel(stress_model)
        model.solve()
        
        observed = data.reset_index()[['date', 'meting']].rename(columns={'meting': 'observed'})
        observed['meetpunt'] = meetpunt
        observed['source'] = 'datalogger'
        
        backcast = model.simulate(tmin=start_backcast, tmax=end_backcast).reset_index()
        backcast = backcast.rename(columns={0: 'Simulation', 'index': 'date'})
        backcast['meetpunt'] = meetpunt
        
        combined = pd.merge(observed, backcast, on=['date', 'meetpunt'], how='outer')
        combined['combination'] = combined['observed'].fillna(combined['Simulation'])
        combined['source'] = combined.apply(lambda row: row['source'] if pd.notnull(row['observed']) else 'Simulation', axis=1)
        
        combined_data = pd.concat([combined_data, combined[['date', 'meetpunt', 'combination', 'source']]], ignore_index=True)


#%% Task: Remove duplicate values, only keep 'handpeiling' in the column 'source'. Plot the dataframe in 1 plot.
# When only using datalogger data, removing duplicates is not necessary. 


filter_data = combined_data.copy()
filter_data['date'] = pd.to_datetime(filter_data['date'])
filter_data = filter_data.sort_values(by='date')

#%% Task: Revised plotting of merged df in subplots.

import matplotlib.pyplot as plt
import seaborn as sns

simulation_data = combined_data[combined_data['source'] == 'Simulation']
datalogger_data = combined_data[combined_data['source'] == 'datalogger']

unique_meetpunts = combined_data['meetpunt'].unique()

output_dir = "scatter_plots"
os.makedirs(output_dir, exist_ok=True)

for meetpunt in unique_meetpunts:
    plt.figure(figsize=(10, 6))  

    sim_data_mp = simulation_data[simulation_data['meetpunt'] == meetpunt]
    dl_data_mp = datalogger_data[datalogger_data['meetpunt'] == meetpunt]

    sns.scatterplot(data=dl_data_mp, x='date', y='combination', color='#1f77b4', label='Data Logger', s=10)

    sns.scatterplot(data=sim_data_mp, x='date', y='combination', color='orange', label='Simulation', s=10)

    plt.title(f'Groundwater Level for Monitoring Well: {meetpunt}', fontsize=12)
    plt.xlabel('Time [days]', fontsize=12)
    plt.ylabel('Groundwater Level [m MSL]', fontsize=12)
    plt.legend()
    
    filename = f"{meetpunt}_scatter_plot.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=300)  

    plt.show()  


#%% Task: Figure with distinction between data loggers and manual collection.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if 'date' in merge_df.columns:
    merge_df['date'] = pd.to_datetime(merge_df['date'])
else:
    merge_df.index = pd.to_datetime(merge_df.index)

sns.set_theme()
sns.set_style("white")

palette = sns.color_palette("muted")

fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

gemeten_df = merge_df[merge_df['waarneming'] == 'Gemeten']
sns.scatterplot(ax=axs[0], data=gemeten_df, x='date', y='meting', hue='meetpunt', s=10, palette=palette)
axs[0].set_ylabel('Groundwater Level [m MSL]')
axs[0].set_title('Overview Manual Measurements')

gemeten_datalogger_df = merge_df[merge_df['waarneming'] == 'Gemeten met datalogger']
sns.scatterplot(ax=axs[1], data=gemeten_datalogger_df, x='date', y='meting', hue='meetpunt', s=10, palette=palette)
axs[1].set_ylabel('Groundwater Level [m MSL]')
axs[1].set_xlabel('Date [days]')
axs[1].set_title('Overview Datalogger')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title='Meetpunt')
axs[0].get_legend().remove()  
axs[1].get_legend().remove()  

plt.subplots_adjust(right=0.85)  

plt.show()












