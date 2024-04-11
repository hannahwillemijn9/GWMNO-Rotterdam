#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:56:44 2024

@author: willemijnelzinga


"""



#%% Sequel of corrected script 'rozcorr' on January 10. 
# Consists: Using 1D hydrograph data, setup, QR based ranking of monitoring wells, define reduction: gwn reduction, reconstruction,
# results: get error scores, plotting hydrograph reconstruction.
#%% Task: Prepare the new dataframe for the script. 

import os
import pandas as pd 
import numpy as np
import geopandas as gpd
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.colors
import numpy as np 
import contextily as ctx
import pysensors as ps 



filterheij=filter_data.pivot(index='meetpunt', columns='date', values='combination').reset_index()
filterheij.sample(frac=1, random_state=1)
filterheij.reset_index(inplace=True)

key = filterheij.iloc[:, 1:2]
filterheij.index.name = 'sensors'

filterheij = filterheij.drop(columns=['index', 'meetpunt'])
filterheij = filterheij.T
filterheij.describe().round(2)

filterheij.index = pd.to_datetime(filterheij.index)
filterheij = filterheij.loc['2020-01-01': '2024-01-01']

#%% Task: Geodataframes ! Download the shp files of Heijplaat/Charlois

import geopandas as gpd

gdf = gpd.read_file('Heijplaat.shp')
print("Original CRS:", gdf.crs)
gdf.crs = 'EPSG:28992'
gdf = gdf.to_crs(epsg=28992)

gdf2 = gpd.read_file('xyzheij.shp')
print("Original CRS:", gdf.crs)
gdf.crs = 'EPSG:28992'
gdf = gdf.to_crs(epsg=28992)

gdf2['meetpunt'] = gdf2['BUISCODE'].astype(str) + '-' + gdf2['VOLGNUMMER'].astype(str)
print(gdf2.columns)
print(gdf2['meetpunt'].dtype)

gdf2 = pd.merge(gdf2, key, left_on='meetpunt', right_on='meetpunt', how='left').to_crs(epsg=28992)

gdf2 = gdf2.drop(columns=['HOOGTE_MAA', 'LAST_UPDAT', 'LAST_UPD_1', 'CREATED_BY', 'CREATION_D', 'MAT_CODE'])
gdf2 = gdf2.drop(columns=['ID', 'BUISCODE', 'VOLGNUMMER', 'BUISCODE_P', 'INW_DIAMET', 'HOOGTE_MEE', 'NUL_METING', 'BOVENKANT_', 'LENGTE_BUI', 'HOOGTE_BOV', 'TOEL_AFWIJ', 'BTP_CODE', 'MEETMERK', 'PLAATSBEPA', 'DATUM_STAR', 'DATUM_EIND', 'DATUM_VERV', 'IND_PLAATS'])

gdf2.rename(columns={'X_COORDINA': 'X'}, inplace=True)
gdf2.rename(columns={'Y_COORDINA': 'Y'}, inplace=True)

#

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry as sg
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib


modelname = 'Heijplaat'
output_dir = pathlib.Path(f"data/5-visualization/{modelname}/validate_heads")
output_dir.mkdir(exist_ok=True, parents=True)

df = gdf2.drop('geometry', axis=1)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["X"], df["Y"]), crs="EPSG:28992").to_crs("EPSG:3857")


levels = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de']

cmap = matplotlib.colors.ListedColormap(colors)
cmap.set_under(colors[0])
cmap.set_over(colors[-1])
norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
gdf.sort_values(by="Z").plot(column="Z", ax=ax, legend=False, cmap=cmap, norm=norm, markersize=50, edgecolor='black', linewidth=1.0)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Adjust colorbar
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad="5%")
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, ticks=levels, extend="both", label="[m]")

# Set titles and labels
ax.set_title(f"Groundwater Level [m MSL] Monitoring Wells {modelname}")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.tight_layout()

plt.show()


#%% Task: Dataframe in array - global centering - train and test data 

import numpy as np

X = filterheij.to_numpy()
n_samples, n_features = X.shape
xmin, xmax= X.min(), X.max()

unit = "[m]"

print('Sampling period: ', filterheij.T.columns[1], filterheij.T.columns[-1])
print('Number of samples:', n_samples)
print('Number of features (sensors):', n_features)
print('Shape of X (samples, sensors):', X.shape)
print('Min. and max. value:', xmin, unit, ';', xmax, unit)

print('Min. and max. centered value:', X.min().round(3), unit, X.max().round(3), unit)
print(f"Min. and max. centered value: {X.min()} {unit}, {X.max()} {unit}")
print('Mean centered data:', X.mean().round(5), unit)

# Ratio 80/20
tdata = int(len(X)*0.80)
X_train, X_test = X[:tdata], X[tdata:]
print('Train data:', len(X_train), ', Test data:', len(X_test))
print('Train data:', round(((tdata/len(X))*100)), '%', ', Test data:', 100-round(((tdata/len(X))*100)), '%')



#%% Task: QR-based ranking 

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt 
import matplotlib.colors as mplcolors
import numpy as np
import contextily as ctx
import cartopy.crs as ccrs 
import cartopy.io.img_tiles as cimgt 


seed = 42 

model = ps.SSPOR(
     basis=ps.basis.Identity(),
     n_sensors= n_features).fit(X_train,seed=seed)
model

sensors_all = model.get_selected_sensors()

s = sensors_all.tolist()
s = pd.DataFrame(s,columns=['sensors'])

sensors_ID= pd.merge(s, key, left_on='sensors', right_on='sensors', how='left')
sensors_ID.insert(0,'rank', range(1, 1+len(sensors_ID)))
sensors_ID.set_index('rank', inplace=True, drop=False)

gdf = gdf.to_crs(epsg=28992)
gdf2 = gdf2.to_crs(epsg=28992)

bounds= [int(n_features*0), int(n_features*0.1), int(n_features*0.25), int(n_features*0.5),int(n_features*0.75),int(n_features*0.9), int(n_features*1)]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#103F6E','#5AA2CC','#DEEEF7','#FBD9CA','#EF8A62','#941F2D'])
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

gdf3 = pd.merge(gdf2, sensors_ID, left_on='meetpunt', right_on='meetpunt', sort=False, how='right').to_crs(epsg=28992)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

df = gdf3.copy()

gdf = gpd.GeoDataFrame(df, geometry='geometry')

gdf = gdf.drop(['Unnamed: 0'], axis=1, errors='ignore')

gdf.crs = "EPSG:28992"
gdf = gdf.to_crs("EPSG:3857")

modelname = 'Heijplaat'
pathlib.Path(f"data/5-visualization/{modelname}/validate_ranks").mkdir(parents=True, exist_ok=True)

list = ["rank"]
levels = [0, 2, 4, 6, 8, 10, 12, 14] 

n_colors = len(levels) - 1  
cmap = plt.get_cmap('viridis', n_colors)  
colors = [cmap(i) for i in range(n_colors)]

for i in list:
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.sort_values(by=i).plot(
        column=i, ax=ax, legend=False, cmap=cmap, norm=norm,
        markersize=50, edgecolor='black', linewidth=1.0,
    )
    
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    settings_cbar = {"ticks": levels, "extend": "both", "label": "Rank"}
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(cbar, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)
    
    ax.set_title(f"Ranking Visualization - {modelname}")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()

# Add a label to the wells 

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib import patheffects


modelname = 'Heijplaat'
pathlib.Path(f"data/5-visualization/{modelname}/validate_ranks").mkdir(parents=True, exist_ok=True)

# Visualization for 'rank'
levels = [0, 2, 4, 6, 8, 10, 12, 14]  
n_colors = len(levels) - 1
cmap = plt.get_cmap('viridis', n_colors)
colors = [cmap(i) for i in range(n_colors)]

cmap = matplotlib.colors.ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

fig, ax = plt.subplots(figsize=(10, 10))
gdf.sort_values(by='rank').plot(
    column='rank', ax=ax, legend=False, cmap=cmap, norm=norm,
    markersize=50, edgecolor='black', linewidth=1.0,
)

text_path_effects = [patheffects.withStroke(linewidth=3, foreground="white")]

for idx, row in gdf.iterrows():
    point = row.geometry.centroid if row.geometry.geom_type == 'MultiPoint' else row.geometry
    
    ax.text(
        x=point.x, y=point.y,  
        s=row['rank'], 
        fontsize=14,
        ha='right', va='bottom',
        path_effects=text_path_effects  
    )

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

settings_cbar = {"ticks": levels, "extend": "both", "label": "Rank"}
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad="5%")
fig.colorbar(cbar, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)

ax.set_title(f"Ranking Visualization - {modelname}")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.tight_layout()
plt.show()


#%% Task: Reduction

### 10% reduction 
reduction = 0.10
n_sensors = int(n_features * (1-reduction))
print('wells removed', int(n_features-n_sensors))
print('wells remaining', n_sensors)


sensor_range = np.arange(1, n_features + 1)
errors = model.reconstruction_error(X_test, sensor_range = sensor_range)

plt.figure(figsize=(10, 6))

if n_sensors < len(errors):
    plt.plot(sensor_range, errors, color='tab:blue', linestyle='-', marker='')
    plt.scatter(n_sensors, errors[n_sensors - 1], color='tab:orange', zorder=5)
    plt.vlines(n_sensors, ymin=-0.5, ymax=errors[n_sensors - 1], colors='tab:orange', linestyle='--', linewidth=2.0, label='Selected number of wells')
    plt.hlines(errors[n_sensors -1], xmin=-5, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 15])

else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')
    
#Set labels and title 
plt.xlabel('Number of Monitoring Wells')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error Heijplaat 10%')

#Grid 
plt.grid(True)

offset = 0.1 * (max(errors) - min(errors))
plt.ylim(bottom=0, top=max(errors))

plt.show()

### 25% reduction

reduction = 0.25
n_sensors = int(n_features * (1-reduction))
print('wells removed', int(n_features-n_sensors))
print('wells remaining', n_sensors)


sensor_range = np.arange(1, n_features + 1)
errors = model.reconstruction_error(X_test, sensor_range = sensor_range)

plt.figure(figsize=(10, 6))

if n_sensors < len(errors):
    plt.plot(sensor_range, errors, color='tab:blue', linestyle='-', marker='')
    plt.scatter(n_sensors, errors[n_sensors - 1], color='tab:orange', zorder=5)
    plt.vlines(n_sensors, ymin=-0.5, ymax=errors[n_sensors - 1], colors='tab:orange', linestyle='--', linewidth=2.0, label='Selected number of wells')
    plt.hlines(errors[n_sensors -1], xmin=0, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 15])
else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')
    
#Set labels and title 
plt.xlabel('Number of Monitoring Wells')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error Heijplaat 25%')

#Grid 
plt.grid(True)

offset = 0.1 * (max(errors) - min(errors))
plt.ylim(bottom=0, top=max(errors))

plt.show()

### 50% reduction
reduction = 0.50
n_sensors = int(n_features * (1-reduction))
print('wells removed', int(n_features-n_sensors))
print('wells remaining', n_sensors)

sensor_range = np.arange(1, n_features + 1)
errors = model.reconstruction_error(X_test, sensor_range = sensor_range)

plt.figure(figsize=(10, 6))

if n_sensors < len(errors): 
    plt.plot(sensor_range, errors, color='tab:blue', linestyle='-', marker='')
    plt.scatter(n_sensors, errors[n_sensors - 1], color='tab:orange', zorder=5)
    plt.vlines(n_sensors, ymin=-0.5, ymax=errors[n_sensors - 1], colors='tab:orange', linestyle='--', linewidth=2.0, label='Selected number of wells')
    plt.hlines(errors[n_sensors -1], xmin=0, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 15])

else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')

plt.xlabel('Number of Monitoring Wells ')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error Heijplaat 50%')

plt.grid(True)

offset = 0.1 * (max(errors) - min(errors))
plt.ylim(bottom=0, top=max(errors))
plt.show()


### 75% reduction

reduction = 0.75
n_sensors = int(n_features * (1-reduction))
print('wells removed', int(n_features-n_sensors))
print('wells remaining', n_sensors)

sensor_range = np.arange(1, n_features + 1)
errors = model.reconstruction_error(X_test, sensor_range = sensor_range)

plt.figure(figsize=(10, 6))

if n_sensors < len(errors): 
    plt.plot(sensor_range, errors, color='tab:blue', linestyle='-', marker='')
    plt.scatter(n_sensors, errors[n_sensors - 1], color='tab:orange', zorder=5)
    plt.vlines(n_sensors, ymin=-0.5, ymax=errors[n_sensors - 1], colors='tab:orange', linestyle='--', linewidth=2.0, label='Selected number of wells')
    plt.hlines(errors[n_sensors -1], xmin=0, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 15])

else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')

plt.xlabel('Number of Monitoring Wells ')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error Heijplaat 75%')

plt.grid(True)

offset = 0.1 * (max(errors) - min(errors))
plt.ylim(bottom=0, top=max(errors))
plt.show()

### 90% reduction 

reduction = 0.90
n_sensors = int(n_features * (1-reduction))
print('wells removed', int(n_features-n_sensors))
print('wells remaining', n_sensors)

sensor_range = np.arange(1, n_features + 1)
errors = model.reconstruction_error(X_test, sensor_range = sensor_range)

plt.figure(figsize=(10, 6))

if n_sensors < len(errors): 
    plt.plot(sensor_range, errors, color='tab:blue', linestyle='-', marker='')
    plt.scatter(n_sensors, errors[n_sensors - 1], color='tab:orange', zorder=5)
    plt.vlines(n_sensors, ymin=-0.5, ymax=errors[n_sensors - 1], colors='tab:orange', linestyle='--', linewidth=2.0, label='Selected number of wells')
    plt.hlines(errors[n_sensors -1], xmin=0, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 15])

else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')

plt.xlabel('Number of Monitoring Wells ')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error Heijplaat 90%')

plt.grid(True)

offset = 0.1 * (max(errors) - min(errors))
plt.ylim(bottom=0, top=max(errors))
plt.show()


# Combined figure 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable


reductions = [0.10, 0.25, 0.50, 0.75, 0.90]
sensor_ranges = np.arange(1, n_features + 1)

plt.figure(figsize=(10, 6))
modelname = 'Heijplaat'
pathlib.Path(f"data/5-visualization/{modelname}/validate_ranks").mkdir(parents=True, exist_ok=True)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for reduction, color in zip(reductions, colors):
    n_sensors = int(n_features * (1 - reduction))
    errors = model.reconstruction_error(X_test, sensor_range=sensor_ranges)

    
    if n_sensors < len(errors):
        plt.scatter(n_sensors, errors[n_sensors - 1], color=color, zorder=5)
        plt.vlines(n_sensors, ymin=0, ymax=errors[n_sensors - 1], colors=color, linestyle='--', linewidth=2.0, label=f'{int(reduction*100)}% Reduction')
        # plt.hlines(errors[n_sensors - 1], xmin=1, xmax=n_sensors, colors=color, linestyle='--', linewidth=2.0)
        plt.text(n_sensors + 0.5, errors[n_sensors -1], f'{errors[n_sensors -1]:.3f}', color=color, fontsize=12, va='bottom', ha='right')

plt.xlabel('Number of Monitoring Wells')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error with a 10-90% Reduction')

plt.grid(True)
plt.legend()
offset = 0.1 * (max(errors) - min(errors)) 
plt.ylim(bottom=0.0)

plt.show()



#%% Task: Reconstruction

# Set number of sensors after fitting
model.set_number_of_sensors(n_sensors)
print(model)
sensors = model.get_selected_sensors()

# Subsample data so we only have measurements at chosen sensor locations
X_test_subsampled = X_test[:, sensors]; X_train_subsampled = X_train[:, sensors]
X_test_reconstructed = model.predict(X_test_subsampled); X_train_reconstructed = model.predict(X_train_subsampled)


#Rename Index and columns 
recon= pd.DataFrame(X_test_reconstructed)
recon.columns=filterheij.columns # df is filterroz 
obs=filterheij[len(X_train):len(filterheij)]
recon.index=obs.index 

# Creates translation
s_red = sensors.tolist()
s_red = pd.DataFrame(s_red,columns=['sensors'])

s_red_list = sensors.tolist()
s_red_df = pd.DataFrame(s_red_list, columns=['sensors'])

# Merge sensor list and key
sensors_ID_red= pd.merge(s_red_df, key, left_on='sensors', right_on='sensors', how='left')
sensors_ID_red.insert(0,'rank', range(1, 1+len(sensors)))
sensors_ID_red.set_index('rank', inplace=True,drop=False)

#Rename columns
# Reconstructed: transpose, merge with another df, set index, transpose df back to original. 
recon=recon.T
recon= pd.merge(recon, key, left_on='sensors', right_on='sensors', how='left')
recon.set_index('meetpunt', inplace=True, drop=True)
recon=recon.T

# Observed: transpose, merge with another df, set index, transpose df back to original. 
obs=obs.T
obs= pd.merge(obs, key, left_on='sensors', right_on='sensors', how='left')
obs.set_index('meetpunt', inplace=True,drop=True)
obs=obs.T

# sensor_list= list(sensors_ID_red.meetpunt) #replace GW_Nummer with sensor or meetpunt 
sensor_list = sensors_ID_red['meetpunt'].tolist()

#redundant
recon_red=recon.drop(sensor_list, axis = 1)
obs_red=obs.drop((sensor_list), axis = 1)
#optimal
recon_opt=recon.filter(sensor_list)
obs_opt=obs.filter(sensor_list)

#%% Task: Results 

#Get Error Scores
err = recon_red-obs_red
err_rel = err/((np.max(obs_red)-np.min(obs_red)))
err_nash = obs_red - np.mean(obs_red)
MAE = np.mean(abs(err))
NSE = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))  
r=recon_red.corrwith(obs_red, axis = 0)
R2=r ** 2
RMSE =  np.sqrt(np.mean(err ** 2))
rRMSE = np.sqrt(np.mean(err_rel ** 2)) * 100
Bias = np.mean(err)
rBias = np.mean(err_rel) * 100
alpha = np.std(recon_red)/np.std(obs_red)
beta = np.mean(recon_red)/np.mean(obs_red)
KGE = 1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)

scores = pd.DataFrame({'MAE': MAE, 'NSE' : NSE, 'R2': R2, 'RMSE' : RMSE, 'rRMSE' :  rRMSE, 'Bias' : Bias, 'rBias' : rBias, 'KGE' : KGE, 'alpha': alpha, 'beta': beta, 'r_score' :r})



#Plot hydrograph with selected metrics
    
plt.rcParams['figure.figsize'] = [10, 3]

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

columns_list = sorted(recon_red.columns.values)
for i in columns_list:
    plt.plot(obs_red[i], '-', color='tab:blue', linewidth=2)
    plt.plot(recon_red[i], '-', color='tab:orange', linewidth=1.5)
    plt.legend(labels=['Observed', 'Reconstructed'])
    plt.title('GWM ID: '+str(i)+': reconstructed with ' +
              str(n_sensors) + ' from ' + str(n_features) + ' GMWs', fontsize=12)
    plt.ylabel('GWL [m NAP]')
    plt.autoscale(enable=True, axis='x', tight=True)
    textstr1 = ('MAE [m]: ', scores.MAE[i].round(2), ', R² []: ', scores.R2[i].round(2),
                ',   RMSE [m] :', scores.RMSE[i].round(2), ',   rBias [%] :', scores.rBias[i].round(3))
    # Gebruik ''.join(map(str, textstr1)) om de tuple om te zetten naar een string
    plt.figtext(0.12, -0.035, ''.join(map(str, textstr1)), color='white',
                fontsize=10, bbox={'fc': 'black', 'ec': 'black'})
    
    reconstructed = os.path.join(output_dir, f'GWM_reconstructed{i}.png')
    plt.savefig(reconstructed, dpi=300, bbox_inches='tight')
  
    plt.show()
    

#Test if there is a significant difference between the reconstructed and measured data.
import os
import pandas as pd
import scipy.stats as stats

# Assuming obs_red, recon_red, and scores are defined as per your context
# Ensure to define 'output_dir', 'n_sensors', and 'n_features' as per your script

output_dir = "rozfigures"
os.makedirs(output_dir, exist_ok=True)

columns_list = sorted(recon_red.columns.values)

# Iterate through each GWM to calculate and print the paired t-test results
for i in columns_list:
    # Perform a paired t-test for measured vs. reconstructed data
    t_stat, p_value = stats.ttest_rel(obs_red[i], recon_red[i])
    
    # Determine if the difference is statistically significant
    alpha = 0.05  # Significance level
    if p_value < alpha:
        print(f"GWM ID: {i}: Significant difference (p-value = {p_value:.3f})")
    else:
        print(f"GWM ID: {i}: No significant difference (p-value = {p_value:.3f})")
        

# Boxplot with text labels 
import pandas as pd 
import matplotlib.pyplot as plt 

metricsplot = scores[['R2']]
fig, ax = plt.subplots(figsize=(8, 5))

boxplot = metricsplot.boxplot(ax=ax, medianprops=dict(color='orange', linewidth=2))

quantiles = np.quantile(metricsplot, np.array([0.00, 0.25, 0.50, 0.75, 1.00])).squeeze()
ax.hlines(quantiles, [0] * quantiles.size, [1] * quantiles.size, 
          color='#1f77b4', ls=':', lw=0.5, zorder=0)

# ax.scatter([0.5] * quantiles.size, quantiles, marker='o', color='blue', zorder=1)

for i, q in enumerate(quantiles): 
    ax.text(0.5, q, f'q{i+1}: {q:.3f}', ha='right', va='center', color='#1f77b4', fontsize=12)

y_min = metricsplot.min().min()
y_max = metricsplot.max().max()

padding = 0.1 * (y_max - y_min)

ax.set_ylim(bottom=y_min - padding, top=y_max + padding)

ax.set_title('Reconstruction Metric: $R^2$ - Heijplaat', size=12)
# plt.xlabel('Metric: $R^2$')
plt.ylabel('[ ]')

plt.show()

  

# Error-based ranking
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

s1= abs(obs-recon).mean().sort_values(ascending=False)
s1=s1[:(n_features-n_sensors)]
rank=pd.DataFrame(s1, columns=['error']).sort_values(by=['error'])

rank.insert(0,'error_rank', range(1, 1+len(rank)))
rank.reset_index(level=0, inplace=True)
rank.set_index('error_rank', inplace=True, drop=False)

gdf4= pd.merge(gdf2, rank, left_on='meetpunt', right_on='meetpunt', sort=False, how='right').to_crs(epsg=28992)



# MAE of eliminated monitoring wells 

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
import matplotlib
import os

# Function to format the colorbar ticks with two decimal places
def format_tick(value, tick_number):
    return f'{value:.2f}'

gdf4 = gdf4.to_crs(epsg=3857)

# Setup output directory
output_dir = "MAEheij"
os.makedirs(output_dir, exist_ok=True)

# Assuming the gdf4 is properly set
modelname = 'Heijplaat'
output_dir = pathlib.Path(f"data/5-visualization/{modelname}/validate_ranks")
output_dir.mkdir(parents=True, exist_ok=True)

# Define visualization parameters
column = "error"
error_ranks = [-0.2, 0, 0.02, 0.04, 0.06, 0.08, 0.10]
n_colors = len(error_ranks) - 1
cmap = plt.get_cmap('viridis', n_colors)
cmap = matplotlib.colors.ListedColormap([cmap(i) for i in range(n_colors)])
norm = matplotlib.colors.BoundaryNorm(error_ranks, cmap.N)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

bounds = gdf4.total_bounds 
buffer_size = 250

xmin, ymin, xmax, ymax = bounds
xmin -= buffer_size  # Increase left boundary
xmax += buffer_size  # Increase right boundary
ymin -= buffer_size / 2  # Increase bottom boundary
ymax += buffer_size / 2  # Increase top boundary

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])


gdf4.plot(ax=ax, column=column, cmap=cmap, norm=norm, legend=False)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Setting up the color bar
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar.set_label('MAE')
cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_tick))

text_path_effects = [patheffects.withStroke(linewidth=3, foreground='white')]


# Add labels for the error values
for idx, row in gdf4.iterrows():
    ax.text(
        x=row.geometry.centroid.x,
        y=row.geometry.centroid.y,
        s=f'{row[column]:.2f}',
        fontsize=9,
        ha='center',
        va='bottom'
    )

# Add titles and labels
ax.set_title(f"MAE Visualization - {modelname}")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Save the figure
errorsheijplaat = os.path.join(output_dir, 'GWM_errors.png')
plt.savefig(errorsheijplaat, dpi=300, bbox_inches='tight')

plt.show()



#Task: Remaining monitoring wells 


# Error-based ranking of remaining monitoring wells
s1 = abs(obs-recon).mean().sort_values(ascending=False)
s1=s1[:(n_features-n_sensors)]
rank=pd.DataFrame(s1, columns=['error']).sort_values(by=['error'])

rank.insert(0,'error_rank', range(1, 1+len(rank)))
rank.reset_index(level=0, inplace=True)
rank.set_index('error_rank', inplace=True, drop=False)

#create gdf of best sensors based on error
gdf5= pd.merge(gdf2, rank, left_on='meetpunt', right_on='meetpunt', sort=False, how='right').to_crs(epsg=28992)


# Plot the locations of the remaining monitoring wells 

import pandas as pd 
import geopandas as gpd 

gdf5 = gdf3[~gdf3['meetpunt'].isin(gdf4['meetpunt'])]
gdf5 = gdf5.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))


bounds = gdf5.total_bounds 
buffer_size = 250

xmin, ymin, xmax, ymax = bounds
xmin -= buffer_size  # Increase left boundary
xmax += buffer_size  # Increase right boundary
ymin -= buffer_size / 2  # Increase bottom boundary
ymax += buffer_size / 2  # Increase top boundary

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

gdf5.plot(ax=ax, color='#1f77b4', markersize=40, legend=True)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Title and labels
ax.set_title(f"Remaining Monitoring Wells - {modelname}")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.tight_layout()

plt.show()




#%% Basic map of Heijplaat 

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pathlib
import os
import contextily as ctx

gdf2 = gdf2.to_crs(epsg=3857)

modelname = 'Heijplaat'
output_dir = pathlib.Path(f"data/5-visualization/{modelname}/well_locations")
os.makedirs(output_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 10))

gdf2.plot(ax=ax, color='#1f77b4', markersize=40, legend=True)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title(f"Monitoring Wells - {modelname}")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.tight_layout()

well_map_path = os.path.join(output_dir, 'Monitoring_Wells_Map.png')
plt.savefig(well_map_path, dpi=300, bbox_inches='tight')

plt.show()








   
























# #### END OFFICIAL SCRIPT 


# # Find the monitoring wells that are staying in the study area. 

# err = recon_opt-obs_opt
# err_rel = err/((np.max(obs_opt)-np.min(obs_opt)))
# err_nash = obs_opt - np.mean(obs_opt)
# MAE = np.mean(abs(err))
# NSE = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))  
# r=recon_opt.corrwith(obs_opt, axis = 0)
# R2=r ** 2
# RMSE =  np.sqrt(np.mean(err ** 2))
# rRMSE = np.sqrt(np.mean(err_rel ** 2)) * 100
# Bias = np.mean(err)
# rBias = np.mean(err_rel) * 100
# alpha = np.std(recon_opt)/np.std(obs_opt)
# beta = np.mean(recon_opt)/np.mean(obs_opt)
# KGE = 1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)

# scores = pd.DataFrame({'MAE': MAE, 'NSE' : NSE, 'R2': R2, 'RMSE' : RMSE, 'rRMSE' :  rRMSE, 'Bias' : Bias, 'rBias' : rBias, 'KGE' : KGE, 'alpha': alpha, 'beta': beta, 'r_score' :r})


# plt.rcParams['figure.figsize'] = [10, 3]
# # Plot reconstruction

# for i in sorted(list(recon_opt.columns.values)):
#     plt.plot(obs_opt[i], 'o', color='tab:blue', linewidth=0.20)
#     plt.plot(recon_opt[i], '-', color='tab:red', linewidth=0.5)
#     plt.legend(labels=['Measured', 'Reconstructed'])
#     plt.title('GWM ID: '+str(i)+': maintained with ' +
#               str(n_sensors) + ' from ' + str(n_features) + ' GMWs', fontsize=12)
#     plt.ylabel('GWL [ m a.s.l]')
#     plt.autoscale(enable=True, axis='x', tight=True)
#     textstr1 = 'MAE [m]: ', scores.MAE[i].round(2), ',   KGE []: ', scores.KGE[i].round(2), ',   NSE []: ', scores.NSE[i].round(
#         2), ',   R2 []: ', scores.R2[i].round(2), ',   RMSE [m] :', scores.RMSE[i].round(2), ',   rBias [%] :', scores.rBias[i].round(3)
#     plt.figtext(0.12, -0.035, textstr1, color='white',
#                 fontsize=10, bbox={'fc': 'black', 'ec': 'black'})
#     plt.show()

