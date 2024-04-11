#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:56:44 2024

@author: willemijnelzinga


"""


#%% Task: Prepare the new dataframe for the script. 

# Dataframe: 'filter_data'. All duplicates are removed. Time period is from 2010-09-01 to 2023-12-31

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

filterroz=filter_data.pivot(index='meetpunt', columns='date', values='combination').reset_index()
filterroz.sample(frac=1, random_state=1)
filterroz.reset_index(inplace=True)

key = filterroz.iloc[:, 1:2]
filterroz.index.name = 'sensors'

filterroz = filterroz.drop(columns=['index', 'meetpunt'])
filterroz = filterroz.T
filterroz.describe().round(2)

filterroz.index = pd.to_datetime(filterroz.index)
filterroz = filterroz.loc['2020-01-01': '2024-01-01']


#%% Task: Geodataframes 

import geopandas as gpd

gdf = gpd.read_file('rozenburg.shp')
print("Original CRS:", gdf.crs)
gdf.crs = 'EPSG:28992'
gdf = gdf.to_crs(epsg=28992)

gdf2 = gpd.read_file('xyzrozenburg.shp')
print("Original CRS:", gdf.crs)
gdf.crs = 'EPSG:28992'
gdf = gdf.to_crs(epsg=28992)

gdf2['meetpunt'] = gdf2['BUISCODE'].astype(str) + '-' + gdf2['VOLGNUMMER'].astype(str)
print(gdf2.columns)
print(gdf2['meetpunt'].dtype)

gdf2 = pd.merge(gdf2, key, left_on='meetpunt', right_on='meetpunt', how='left').to_crs(epsg=28992)


gdf2 = gdf2.drop(columns=['HOOGTE_MAA', 'LAST_UPDAT', 'LAST_UPD_1', 'CREATED_BY', 'CREATION_D', 'MAT_CODE'])
gdf2 = gdf2.drop(columns=['fid', 'ID', 'BUISCODE', 'VOLGNUMMER', 'BUISCODE_P', 'INW_DIAMET', 'HOOGTE_MEE', 'NUL_METING', 'BOVENKANT_', 'LENGTE_BUI', 'HOOGTE_BOV', 'TOEL_AFWIJ', 'BTP_CODE', 'MEETMERK', 'PLAATSBEPA', 'DATUM_STAR', 'DATUM_EIND', 'DATUM_VERV', 'IND_PLAATS'])

gdf2.rename(columns={'Zmean (m N': 'Z'}, inplace=True)
gdf2.rename(columns={'X_COORDINA': 'X'}, inplace=True)
gdf2.rename(columns={'Y_COORDINA': 'Y'}, inplace=True)


# Figure
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry as sg
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import glob
    

df = pd.read_csv('gdf2.csv')
df = df.drop('geometry', axis=1)
# Spatially plotting the absolute difference in head
geometry = [sg.Point(float(x), float(y)) for x, y in zip(df["X"], df["Y"])]

gdf = gpd.GeoDataFrame({"geometry": geometry})
for column in df.columns:
    gdf[column] = df[column].values
gdf = gdf.drop('Unnamed: 0', axis=1)

   
gdf_rp = gdf.copy()

# converting coordinates epsg:28992 to epsg:3857
gdf.crs = "EPSG:28992"
gdfrp = gdf.to_crs("EPSG:3857")

modelname = 'Rozenburg'

# fig.show()
pathlib.Path(f"data/5-visualization/{modelname}/validate_heads").mkdir(
    exist_ok=True, parents=True
    )

list = ["Z" ]

#gdfrp = df.to_crs("EPSG:3857")
levels = [-3.0,-2.0,-1.0,-0.5,-0.25,-0.05, 0, 0.05, 0.25, 0.5, 1.0, 2.0, 3.0]
colors = ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']
in_layer = gdfrp


for i in list:
    # color bar settings
    if i == "Z":
        levels = [-3.0,-2.0,-1.0,-0.5,-0.25,-0.05, 0.05, 0.25, 0.5, 1.0, 2.0, 3.0]
        colors = ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']
        # levels = [-2,-1,-0.5,-0.25, 0.25, 0.5, 1, 2]
        # colors = ['#b2182b','#ef8a62','#fddbc7','#f7f7f7','#d1e5f0','#67a9cf','#2166ac']
    else:
        levels = [0, 0.05, 0.25, 0.50, 1.0, 2.0, 3.0]
        colors = ['#d73027','#fc8d59','#fee08b','#d9ef8b','#91cf60','#1a9850']
        # levels = [0, 0.25, 0.5, 1, 2]
        # colors = ['#d7191c','#fdae61','#a6d96a','#1a9641']
        colors.reverse()
        
        
    # te gebruiken bij eigen kleurencombinatie
    cmap = matplotlib.colors.ListedColormap(colors)     
        
    # Make triangles white if data is not larger/smaller than legend_levels-range
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    if gdf_rp[i].max() < levels[-1]:
        cmap.set_over("#FFFFFF")
    if gdf_rp[i].min() > levels[0]:
        cmap.set_under("#FFFFFF")
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

       
# Plot data points
    fig, ax = plt.subplots(figsize=(10, 10))
    in_layer.sort_values(by=i).plot(
        column=i, ax=ax, legend=False, cmap=cmap, norm=norm,
        markersize = 50, edgecolor='black', linewidth=1.0,
        )

    # Plot AOI
    #aoi.plot(edgecolor="black", color="none", ax=ax)

    # Plot basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Add colorbar
    settings_cbar = {"ticks": levels, "extend": "both", "label": "[m]"}
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(cbar, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)

    # Plot settings
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    if i == "Z":
        ax.set_title(r"Groundwater Level [m MSL] Monitoring Wells Rozenburg")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        fig.tight_layout()









#%% Task: Dataframe in array - global centering - train and test data 

import numpy as np

#X-Array
X = filterroz.to_numpy()
n_samples, n_features = X.shape
xmin, xmax= X.min(), X.max()

unit = "[m]"

print('Sampling period: ', filterroz.T.columns[1], filterroz.T.columns[-1])
print('Number of samples:', n_samples)
print('Number of features (sensors):', n_features)
print('Shape of X (samples, sensors):', X.shape)
print('Min. and max. value:', xmin, unit, ';', xmax, unit)

# Min and max values of variable X, with the asociated unit that is rounded to 3 decimals.
print('Min. and max. centered value:', X.min().round(3), unit, X.max().round(3), unit)
#zonder afronding
print(f"Min. and max. centered value: {X.min()} {unit}, {X.max()} {unit}")
print('Mean centered data:', X.mean().round(5), unit)

# Ratio 80/20
tdata = int(len(X)*0.80)
X_train, X_test = X[:tdata], X[tdata:]
print('Train data:', len(X_train), ', Test data:', len(X_test))
print('Train data:', round(((tdata/len(X))*100)), '%', ', Test data:', 100-round(((tdata/len(X))*100)), '%')



#%% Task: QR-based ranking 

seed = 42 

model = ps.SSPOR(
     basis=ps.basis.Identity(),
     n_sensors= n_features).fit(X_train,seed=seed)
model

# Get the rank of the wells
sensors_all = model.get_selected_sensors()

# Creates translation
s = sensors_all.tolist()
s = pd.DataFrame(s,columns=['sensors'])

# Merge sensor list and key
sensors_ID= pd.merge(s, key, left_on='sensors', right_on='sensors', how='left')
sensors_ID.insert(0,'rank', range(1, 1+len(sensors_ID)))
sensors_ID.set_index('rank', inplace=True, drop=False)

gdf = gdf.to_crs(epsg=28992)
gdf2 = gdf2.to_crs(epsg=28992)

#Cmap definition
bounds= [int(n_features*0), int(n_features*0.1), int(n_features*0.25), int(n_features*0.5),int(n_features*0.75),int(n_features*0.9), int(n_features*1)]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#103F6E','#5AA2CC','#DEEEF7','#FBD9CA','#EF8A62','#941F2D'])
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#Voeg df 'sensors_ID' toe aan het gdf3: 
gdf3 = pd.merge(gdf2, sensors_ID, left_on='meetpunt', right_on='meetpunt', sort=False, how='right').to_crs(epsg=28992)


## Random gegenereerde kleuren in de color bar 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

# Assuming gdf3 is already a copy with the correct format
df = gdf3.copy()

# Adjusting the script based on the provided geometry format
gdf = gpd.GeoDataFrame(df, geometry='geometry')

gdf = gdf.drop(['Unnamed: 0'], axis=1, errors='ignore')  # Drop Unnamed if exists

# Set CRS and convert if necessary
gdf.crs = "EPSG:28992"
gdf = gdf.to_crs("EPSG:3857")

modelname = 'Rozenburg'
pathlib.Path(f"data/5-visualization/{modelname}/validate_ranks").mkdir(parents=True, exist_ok=True)

# Visualization for 'rank'
list = ["rank"]
levels = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29]  # Adjust levels based on your 'rank' data specifics

# Generating a sufficient number of colors
# Here, we use matplotlib's colormap utilities to generate a gradient of colors
n_colors = len(levels) - 1  # Number of intervals defined by levels
cmap = plt.get_cmap('viridis', n_colors)  # Using 'viridis' colormap, but you can choose another
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


## add a label to the wells 

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib import patheffects

# Assuming 'gdf' is your GeoDataFrame with the 'rank' and 'geometry' columns correctly set

modelname = 'Rozenburg'
pathlib.Path(f"data/5-visualization/{modelname}/validate_ranks").mkdir(parents=True, exist_ok=True)

# Visualization for 'rank'
levels = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29]  # Adjust levels based on your 'rank' data specifics
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

# Text outline effect
text_path_effects = [patheffects.withStroke(linewidth=3, foreground="white")]

for idx, row in gdf.iterrows():
    # Check if the geometry is 'MultiPoint' and use its centroid
    point = row.geometry.centroid if row.geometry.geom_type == 'MultiPoint' else row.geometry
    
    ax.text(
        x=point.x, y=point.y,  # Use centroid for 'MultiPoint'
        s=row['rank'],  # Rank label
        fontsize=14,
        ha='right', va='bottom',
        path_effects=text_path_effects  # Add a comma here
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

#Define desired monitoring network reduction in percentage.
# Reduction of 82.75 % > in script staat een percentage van 25%

reduction = 0.10
n_sensors = int(n_features * (1-reduction))
print('wells removed', int(n_features-n_sensors))
print('wells remaining', n_sensors)


sensor_range = np.arange(1, n_features + 1)
errors = model.reconstruction_error(X_test, sensor_range = sensor_range)

plt.figure(figsize=(10, 6))

#Blue line 
# plt.plot(sensor_range, errors, color='tab:blue', linestyle='-', marker='')
#Orange point and check if n_sensors is within sensor_range
if n_sensors < len(errors):
    plt.plot(sensor_range, errors, color='tab:blue', linestyle='-', marker='')
    plt.scatter(n_sensors, errors[n_sensors - 1], color='tab:orange', zorder=5)
    plt.vlines(n_sensors, ymin=-0.5, ymax=errors[n_sensors - 1], colors='tab:orange', linestyle='--', linewidth=2.0, label='Selected number of wells')
    plt.hlines(errors[n_sensors -1], xmin=-5, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 30])
else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')
    
#Set labels and title 
plt.xlabel('Number of Monitoring Wells')
plt.ylabel(f'RMSE ({unit})')
plt.title('Reduction Rate 10% Rozenburg')

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

#Blue line 
# plt.plot(sensor_range, errors, color='tab:blue', linestyle='-', marker='')
#Orange point and check if n_sensors is within sensor_range
if n_sensors < len(errors):
    plt.plot(sensor_range, errors, color='tab:blue', linestyle='-', marker='')
    plt.scatter(n_sensors, errors[n_sensors - 1], color='tab:orange', zorder=5)
    plt.vlines(n_sensors, ymin=-0.5, ymax=errors[n_sensors - 1], colors='tab:orange', linestyle='--', linewidth=2.0, label='Selected number of wells')
    plt.hlines(errors[n_sensors -1], xmin=-5, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 30])
    # plt.ylim([0, 0.23])
else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')
    
#Set labels and title 
plt.xlabel('Number of Monitoring Wells')
plt.ylabel(f'RMSE ({unit})')
plt.title('Reduction Rate 25% Rozenburg')

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
    plt.hlines(errors[n_sensors -1], xmin=-5, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 30])

else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')

plt.xlabel('Number of Monitoring Wells ')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error Rozenburg 50%')

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
    plt.hlines(errors[n_sensors -1], xmin=-5, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 30])
else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')

plt.xlabel('Number of Monitoring Wells')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error Rozenburg 75%')

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
    plt.hlines(errors[n_sensors -1], xmin=-5, xmax=n_sensors, colors='tab:orange', linestyle='--', linewidth=2.0)
    plt.xlim([0, 30])
else: 
    print(f'n_sensors value of {n_sensors} is out of the valid range')

plt.xlabel('Number of Monitoring Wells')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error Rozenburg 90%')

plt.grid(True)

offset = 0.1 * (max(errors) - min(errors))
plt.ylim(bottom=0, top=max(errors))
plt.show()


# nieuw 28 maart

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Assuming 'n_features' is defined and 'model' is trained with 'X_test' available
# Let's also assume that 'unit' is defined, for instance: unit = 'units'

reductions = [0.10, 0.25, 0.50, 0.75, 0.90]
sensor_ranges = np.arange(1, n_features + 1)

# Create a figure and a single set of axes
plt.figure(figsize=(10, 6))
modelname = 'Rozenburg'
pathlib.Path(f"data/5-visualization/{modelname}/validate_ranks").mkdir(parents=True, exist_ok=True)

# Define colors or line styles for each reduction scenario
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for reduction, color in zip(reductions, colors):
    n_sensors = int(n_features * (1 - reduction))
    errors = model.reconstruction_error(X_test, sensor_range=sensor_ranges)

    # plt.plot(sensor_ranges, errors, color=color, linestyle='-', marker='', label=f'{int(reduction*100)}% Reduction')
    
    # Mark the selected number of wells with a scatter point
    if n_sensors < len(errors):
        plt.scatter(n_sensors, errors[n_sensors - 1], color=color, zorder=5)
        plt.vlines(n_sensors, ymin=0, ymax=errors[n_sensors - 1], colors=color, linestyle='--', linewidth=2.0, label=f'{int(reduction*100)}% Reduction')
        # plt.hlines(errors[n_sensors - 1], xmin=1, xmax=n_sensors, colors=color, linestyle='--', linewidth=2.0)
        plt.text(n_sensors + 0.5, errors[n_sensors -1], f'{errors[n_sensors -1]:.3f}', color=color, fontsize=12, va='bottom', ha='center')

# Set labels and title
plt.xlabel('Number of Monitoring Wells')
plt.ylabel(f'RMSE {unit}')
plt.title('Reconstruction Error with a 10-90% Reduction')

# Add grid, legend, and set limits
plt.grid(True)
plt.legend()
offset = 0.1 * (max(errors) - min(errors))  # Assuming 'errors' is not empty and has more than one value
plt.ylim(bottom=0.0)

# Display the combined graph
plt.show()



#%% Task: Reconstruction

# Set number of sensors after fitting
model.set_number_of_sensors(n_sensors)
print(model)
sensors = model.get_selected_sensors()

# Subsample data so we only have measurements at chosen sensor locations
X_test_subsampled = X_test[:, sensors]; X_train_subsampled = X_train[:, sensors]
X_test_reconstructed = model.predict(X_test_subsampled); X_train_reconstructed = model.predict(X_train_subsampled)

# Rescaling
X = np.append(X_train, X_test, axis=0)
# X += x_lc # local rescale
# X += x_gc  # global rescale
X_train, X_test = X[:tdata], X[tdata:]

X_reconstructed= np.append(X_train_reconstructed, X_test_reconstructed, axis=0) # appending in one array
# X_reconstructed+= x_lc # local rescale
# X_reconstructed+= x_gc # global rescale
X_train_reconstructed, X_test_reconstructed = X_reconstructed[:tdata], X_reconstructed[tdata:] # splitting in train and test set

#Rename Index and columns 
# recon = reconstructed data; obs = observed data
recon= pd.DataFrame(X_test_reconstructed)
recon.columns=filterroz.columns # df is filterroz 
obs=filterroz[len(X_train):len(filterroz)]
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

#Plotting hydrographs of elimination 
plt.rcParams['figure.figsize'] = [10, 3]
output_dir = "rozfigures"
os.makedirs(output_dir, exist_ok=True)

columns_list = sorted(recon_red.columns.values)
for i in columns_list:
    plt.plot(obs_red[i], '-', color='tab:blue', linewidth=2)
    plt.plot(recon_red[i], '-', color='tab:orange', linewidth=1.5)
    plt.legend(labels=['Measured', 'Reconstructed'])
    plt.title('GWM ID: '+str(i)+': reconstructed with ' +
              str(n_sensors) + ' from ' + str(n_features) + ' GMWs', fontsize=12)
    plt.ylabel('GWL [ m MSL]')
    plt.autoscale(enable=True, axis='x', tight=True)
    textstr1 = 'MAE [m]: ', scores.MAE[i].round(2), ',   R² []: ', scores.R2[i].round(2), ',   RMSE [m] :', scores.RMSE[i].round(2), ',   rBias [%] :', scores.rBias[i].round(3)
    plt.figtext(0.12, -0.035, textstr1, color='white',
                fontsize=10, bbox={'fc': 'black', 'ec': 'black'})
    
    reconstructed = os.path.join(output_dir, f'GWM_reconstructed{i}.png')
    plt.savefig(reconstructed, dpi=300, bbox_inches='tight')
    plt.show()
    
    

#Test if there is a significant difference between the reconstructed and measured data.
import os
import pandas as pd
import scipy.stats as stats

output_dir = "rozfigures"
os.makedirs(output_dir, exist_ok=True)

columns_list = sorted(recon_red.columns.values)

for i in columns_list:
    t_stat, p_value = stats.ttest_rel(obs_red[i], recon_red[i])
    
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
    if i ==2: 
        continue
    
    ax.text(0.5, q, f'q{i+1}: {q:.3f}', ha='left', va='center', color='#1f77b4', fontsize=12)

y_min = metricsplot.min().min()
y_max = metricsplot.max().max()

padding = 0.1 * (y_max - y_min)

ax.set_ylim(bottom=y_min - padding, top=y_max + padding)

ax.set_title('Reconstruction Metric: $R^2$ - Rozenburg', size=12)
# plt.xlabel('Metric: $R^2$')
plt.ylabel('[ ]')

plt.show()



# Error-based ranking

s1= abs(obs-recon).mean().sort_values(ascending=False)
s1=s1[:(n_features-n_sensors)]
rank=pd.DataFrame(s1, columns=['error']).sort_values(by=['error'])

rank.insert(0,'error_rank', range(1, 1+len(rank)))
rank.reset_index(level=0, inplace=True)
rank.set_index('error_rank', inplace=True, drop=False)

#create gdf of best sensors based on error
gdf4= pd.merge(gdf2, rank, left_on='meetpunt', right_on='meetpunt', sort=False, how='right').to_crs(epsg=28992)



# Plot Mean Absolute Error 

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pathlib
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib.ticker import FuncFormatter

def format_tick(value, tick_number):
    return f'{value:.2f}'

# Assuming gdf4 is properly defined and in the correct format
output_dir = "MAEroz"
os.makedirs(output_dir, exist_ok=True)

# Ensure the GeoDataFrame is in the desired CRS
gdf4 = gdf4.to_crs(epsg=3857)

# Setup the output directory
modelname = 'Rozenburg'
output_dir = pathlib.Path(f"data/5-visualization/{modelname}/validate_ranks")
output_dir.mkdir(parents=True, exist_ok=True)

# Define visualization parameters
column = "error"  # Assuming only one column for simplicity

# Dynamically generate error ranks based on the error values
min_error = gdf4[column].min()
max_error = gdf4[column].max()
error_ranks = np.linspace(min_error, max_error, 6)  # Adjust the number of ranks as needed

# error_ranks = [-0.2, 0, 0.02, 0.04, 0.06, 0.08, 0.10]
# n_colors = len(error_ranks) - 1
cmap = plt.get_cmap('viridis', n_colors)



# Use ListedColormap and BoundaryNorm for precise color mapping
cmap = matplotlib.colors.ListedColormap([cmap(i) for i in range(cmap.N)][:n_colors])
norm = matplotlib.colors.BoundaryNorm(error_ranks, cmap.N)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

gdf4.plot(ax=ax, column=column, cmap=cmap, norm=norm, legend=False, markersize=60)

text_path_effects = [patheffects.withStroke(linewidth=3, foreground='white')]

for idx, row in gdf4.iterrows(): 
    label_point = row.geometry.centroid
    label_text = f"{row[column]:.2f}"
    
    ax.text(
        x=label_point.x, y=label_point.y,
        s=label_text, 
        fontsize=10,
        ha='right',
        va='bottom'
    )

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Colorbar setup
settings_cbar = {"ticks": error_ranks, "extend": "both", "spacing": 'proportional', "label": "MAE"}
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="2%", pad=0.2)
scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(scalar_mappable, cax=cbar_ax, **settings_cbar)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_tick))

# Title and labels
ax.set_title(f"MAE Visualization - {modelname}")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.tight_layout()

errorsheijplaat = os.path.join(output_dir, 'GWM_errors.png')
plt.savefig(errorsheijplaat, dpi=400, bbox_inches='tight')

plt.show()


# Plot the MAE for the eliminated monitoring wells 

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

# Your geodataframe setup here
gdf4 = gdf4.to_crs(epsg=3857)

# Setup output directory
output_dir = "MAEroz"
os.makedirs(output_dir, exist_ok=True)

# Assuming the gdf4 is properly set
modelname = 'Rozenburg'
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
cbar_ax = divider.append_axes("right", size="2%", pad=0.1)
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
plt.savefig(errorsrozenburg, dpi=300, bbox_inches='tight')

plt.show()


# Plot Remaining Monitoring Wells 

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
gdf5.plot(ax=ax, color='#1f77b4', markersize=40, legend=True)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Title and labels
ax.set_title(f"Remaining Monitoring Wells - {modelname}")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.tight_layout()

# Save the figure
# well_map_path = os.path.join(output_dir, 'Monitoring_Wells_Map.png')
# plt.savefig(well_map_path, dpi=500, bbox_inches='tight')

plt.show()


#%% Basic map of Rozenburg

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pathlib
import os
import contextily as ctx

# Assuming gdf4 is properly defined and in the correct format

# Ensure the GeoDataFrame is in the desired CRS
gdf2 = gdf2.to_crs(epsg=3857)

# Setup the output directory
modelname = 'Rozenburg'
output_dir = pathlib.Path(f"data/5-visualization/{modelname}/well_locations")
os.makedirs(output_dir, exist_ok=True)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the monitoring wells with a uniform color and size
gdf2.plot(ax=ax, color='#1f77b4', markersize=40, legend=True)


# Add a basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Title and labels
ax.set_title(f"Monitoring Wells - {modelname}")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.tight_layout()

# Save the figure
well_map_path = os.path.join(output_dir, 'Monitoring_Wells_Map.png')
plt.savefig(well_map_path, dpi=500, bbox_inches='tight')

plt.show()









 
    
    




