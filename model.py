import math
import json
import time
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib.ticker import ScalarFormatter
import simpy
import random
import numpy as np
from sklearn.neighbors import BallTree

# Set up variables to download census data
base_url = "https://api.census.gov/data/2010/dec/sf1?get={}&for=block:*&in=state:36 county:{}&key={}"
api_key = "aefbef34be2ab2725a3ae890ca8c372574a2b7e6"
variables = "GEO_ID,ZCTA5,P001001,P005003,P005004,P005006,P005010,P012002,P012026"

# The code below is the metadata for the census variables
# Age is left out since the census age groups need to be aggregated to fit the age groups from the NYC COVID-19 reports
var_url = "https://api.census.gov/data/2010/dec/sf1/variables.json"
var_dict = requests.get(var_url).json()['variables']
var_list = variables.split(',')
var_df = pd.DataFrame(var_dict)[var_list]
var_df = var_df.T[['label', 'concept', 'predicateType']]
var_df


# Start timing the process
start_time = time.time()

# Dictionary of NYC boroughs and their county FIPS
nyc_counties = {'Bronx': '005', 'Brooklyn': '047', 'Manhattan': '061', 
                'Queens': '081', 'Staten Island': '085'}

# Dictionary of the number of census age variables that fall into each age group
age_groups = {'0to17': 4, 
              '18to44': 8, 
              '45to64': 5,
              '65to74': 3, 
              '75over': 3}

# Create a dataframe of demographic data by census tracts
df_list = []
# The census API URL does produces the data for each county
# The first loop will generate the census tract data for the 5 NYC counties
for county in nyc_counties:
    # Create a data frame with the population, race, and gender demographics
    county_fips = nyc_counties[county]
    url = base_url.format(variables, county_fips, api_key)
    response = requests.get(url)
    data = response.json()
    county_df = pd.DataFrame(data[1:], columns=data[0])
    county_df = county_df.set_index('GEO_ID')
    
    # Starting census age variable numbers for [Male, Female]
    num_list = [12003, 12027]
    
    # Combine the census age groups into those from the COVID-19 reports
    for group in age_groups:
        # Create the age variables string for the census API URL
        v_age_list = ['GEO_ID']
        n_var = age_groups[group]
        for num in num_list:
            for i in range(n_var):
                v_age = 'P0' +str(num)
                v_age_list.append(v_age)
                num += 1
        num_list = [num + n_var for num in num_list]
        age_vars = ','.join(v_age_list)
        
        # Create a data frame with the census age groups demographics
        age_url = base_url.format(age_vars, county_fips, api_key)
        age_data = requests.get(age_url).json()
        age_df = pd.DataFrame(age_data[1:], columns=age_data[0])
        # Group the census age groups into the NYC COVID-19 age groups
        age_df[group] = 0
        for col in v_age_list[1:]:
            age_df[group] += age_df[col].astype('int32')
        age_df = age_df[['GEO_ID', group]].set_index('GEO_ID')
        
        # Join age group data frame to the other demographics dataframe
        county_df = county_df.join(age_df)
    
    # Add county data frame to the list of other NYC county dataframes
    df_list.append(county_df)

# Combine all the county data frames in the list into one
df = pd.concat(df_list)

# Print the execution time of the above code
end_time = time.time()
print("Code Execution Time: ", end_time - start_time)

# Clean up data frame
df = df.drop(columns=['tract', 'block', 'state', 'county']).astype('int32')
col_names = {'ZCTA5': 'ZipCode', 'P001001': 'TotalPop', 'P005003': 'White', 
             'P005004': 'Black', 'P005006': 'Asian', 'P005010': 'Hispanic', 
             'P012002': 'Male', 'P012026': 'Female'}
nyc_df = df.rename(columns=col_names)

nyc_df.head()

# Use the reference table from the repository to convert ZCTA to MODZCTA
zcta_modzcta_url = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/Geography-resources/ZCTA-to-MODZCTA.csv"
zcta_modzcta_df = pd.read_csv(zcta_modzcta_url, index_col='ZCTA')
zcta_modzcta_df.index.name='ZipCode'
mod_df = nyc_df.join(zcta_modzcta_df, on='ZipCode').groupby('MODZCTA').sum().drop(columns='ZipCode')

# Convert the demographic variables from count to percentages of total population
mod_df.loc[:, mod_df.columns != 'TotalPop'] = round(
    mod_df.loc[:, mod_df.columns != 'TotalPop'].div(mod_df['TotalPop'], axis=0) * 100, 2)
mod_df.index.name='ZipCode'

mod_df.head()

### NYC Deparment of Health ZCTA Data
# Create a GeoDataFrame of ZCTAs
zcta_url = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/Geography-resources/MODZCTA_2010_WGS1984.geo.json"
zcta_geom = gpd.read_file(zcta_url).to_crs({'init': 'epsg:2263'})
# Keep only the zip codes and geometries
zcta_geom = zcta_geom.filter(['MODZCTA', 'geometry'])
zcta_geom = zcta_geom.rename(columns={'MODZCTA': 'ZipCode'}).dissolve(by='ZipCode')
# Add an area column
zcta_geom.insert(0, 'Area', zcta_geom.area)
zcta_geom.index = zcta_geom.index.astype(int)

zcta_geom.head()

### Merging All 3 Data Sets

# Combine all the ZCTA datasets into one GeoDataFrame
# Read COVID-19 data
covid_zcta_df = pd.read_csv('covid_zctas.csv', index_col=0)
# Join COVID-19 to demographics data
covid_zcta_df = gpd.GeoDataFrame(mod_df.join(covid_zcta_df, how='outer'))
# Join geometries
zcta_gdf = gpd.GeoDataFrame(covid_zcta_df.join(zcta_geom, how='outer'))
# Keep only correct zip codes
zcta_gdf.index.name = "ZipCode"
zcta_gdf = zcta_gdf.query('ZipCode > 10000 & ZipCode < 20000')
zcta_gdf = zcta_gdf.reset_index().rename(columns={'index':'ZipCode'})

# Specify the GeoDataFrame's coordinate system
zcta_gdf.crs = 'EPSG:2263'

zcta_gdf.head()

### NYC Hospitals Data

# Get hospital point data
hospitals_url = "https://geo.nyu.edu/download/file/nyu-2451-34494-geojson.json"
hospitals_geom = gpd.read_file(hospitals_url).to_crs({'init': 'epsg:2263'})
# Drop hospitals that don't have beds
hospitals_geom = hospitals_geom.dropna(subset=['capacity'])

hospitals_geom.head()

## Exploratory Data Analysis



### COVID-19 Choropleth Map

# Set the date to the most recently reported date
day = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
# zcta_pos = zcta_gdf.copy()

# Calculate the mapped criteria as the total number of positive cases per 1000 persons in a ZCTA
column = zcta_gdf[day] / zcta_gdf['TotalPop'] * 1000

# Set up the map
fig, ax = plt.subplots(figsize=(13, 13))

# Add the title
date_str = datetime.strptime(day, '%Y-%m-%d').strftime('%b %d, %Y')
total = int(zcta_gdf[day].sum())
title = 'Positive Cases per 1,000 People By Zip Code\nDate: {}; Total Cases: {}'.format(date_str, total)
plt.title(title, size='16')

# Create the legend
divider = make_axes_locatable(ax)
cax = divider.append_axes('bottom', size='5%', pad=0.1)
cax.tick_params(labelsize=16)
ax.tick_params(left=0, labelleft=0, bottom=0, labelbottom=0)
legend_kwds = {'label': 'Number of Confirmed COVID-19 Cases per 1000 People',
               'orientation': 'horizontal'}

# Plot the ZCTA data
zcta_gdf.plot(ax=ax, column=column, cmap='Blues', edgecolor='black', legend=True, cax=cax, 
              legend_kwds=legend_kwds)

# Add the hospitals layer
hospitals_geom.plot(ax=ax, marker='P', markersize=100, color='red',
                    edgecolor='black', linewidth=.3)
# Add the hospitals legend
ax.legend(labels=['Hospitals'], loc='lower right', prop={'size': 15}, 
          markerscale=1.5)

plt.show()

## Discrete Event Simulation

# ![DES](http://bestirtech.com/blog/wp-content/uploads/2019/07/DES_2.png)



def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    #closest_dist = distances[0]

    # Return indices and distances
    return closest


def nearest_neighbor(left_gdf, right_gdf, left_id, right_id):
    """For each point in left_gdf, find the closest point in right_gdf"""
    
    # The function requires the GeoDataFrames to be in WGS 84
    # Reproject both GeoDataFrames, dropping any rows with no geometries
    left_gdf_wgs84 = left_gdf.dropna(subset=['geometry']).to_crs({'init': 'epsg:4326'}).reset_index()
    right_gdf_wgs84 = right_gdf.dropna(subset=['geometry']).to_crs({'init': 'epsg:4326'}).reset_index()
    
    # Use the centroid of the left GeoDataFrame to in the algorithm
    left_gdf_wgs84['centroid'] = left_gdf_wgs84.centroid
    left_geom_col='centroid'
    right_geom_col='geometry'
    
    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf_wgs84.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_gdf_wgs84[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # get the index in right_gdf_wgs84 that corresponds to the closest point
    closest = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf_wgs84
    closest_points = closest_points.reset_index(drop=True)
    
    # Join and return a data frame with the id's of the closest points
    df = left_gdf_wgs84.join(closest_points[right_id])[[left_id, right_id]]
    return left_gdf.join(df.set_index(left_id), on=left_id)


# Aggregate all the zcta_gdf data to the ZCTA's nearest hospital
nearest_hosp = nearest_neighbor(
    zcta_gdf, hospitals_geom, 
    'ZipCode', 'id').groupby('id').sum()

# Add a column for the number of new daily COVID-19 cases for that hospitals service area
# Calculate the average number of new cases each day from 4/3 to 5/3
nearest_hosp['avgnewcases'] = nearest_hosp.iloc[:, 14:44].diff(axis=1).mean(axis=1).astype(int)
# Add a column for the number of cases when DOH started reporting by zip code, 4/1
day_start = '2020-04-01'
hospitals_gdf = hospitals_geom.join(
    nearest_hosp[[day_start, 'avgnewcases']].rename(
        columns={day_start: 'cases'}), 
    on='id')

hospitals_gdf.head()

### Simulation Set Up


class Hospital():
    '''
    A hospital has a limited number of beds for patients. 
    
    A patient will request for a bed. If one is available they will stay for a length of time and be released.
    
    '''
    def __init__(self, env, num_beds):
        self.env = env
        self.bed = simpy.Resource(env, num_beds)
    
    def fill_bed(self, patient, length_of_stay):
        '''The patient admission process. It takes a "patient" process and admits them to the hospital.'''
        yield self.env.timeout(length_of_stay)


def patient(env, name, status, hosp, df):
    '''
    The "patient" process arrives at the hospital and requests for admission.
    
    Parameters:
        name: patient's name
        status: COVID-19 status (Positive/Negative)
        hosp: the hospital (as a class object)
        df: a dataframe to track the number of patients, released and refused from the hospital
    '''
    
    with hosp.bed.request() as admit:
        # Be admitted to the hopsital or refused if no beds are available
        results = yield admit | env.timeout(0)
        
        # Check if the patient is admitted or refused
        if admit in results:
            # Add 1 to the admitted column in the data frame
            counter(env, df, 'admitted')
            
            # If the patient was not admitted due to COVID-19, stay between 0 to 14 days
            if status == 'negative':
                num = int(round(np.random.normal(4, 3, 1)[0]))
                if num <= 0:
                    num = 0
                length_of_stay = num
            
            # If the patient was admitted because of COVID-19, stay between ~7 to ~34 days
            else:
                length_of_stay = int(round(np.random.normal(19, 3, 1)[0]))
            yield env.process(hosp.fill_bed(name, length_of_stay))
            
            # Add 1 to the released column when the patient is released
            counter(env, df, 'released')
        
        # If the patient was refused, add 1 to the refused column in the data frame
        else:
            counter(env, df, 'refused')


def setup(env, start_patients, num_beds, new_patients, df):
    '''Create a hospital, the number of initial patients and keep creating new patients'''
    # Create the hospital
    hospital = Hospital(env, num_beds)
    
    # Create the inital number of patients already in the hospital
    for i in range(start_patients):
        env.process(patient(env, 'Patient{}'.format(i), 'negative', hospital, df))
    
    # Create more patients while the simulation is running
    while True:
        yield env.timeout(1)
        
        # Randomly generate the number of new patients each day
        num = int(round(np.random.normal(new_patients, 5, 1)[0]))
        if num <= 0:
            num = 0
        
        for j in range(num):
            i += 1
            env.process(patient(env, 'Patient{}'.format(i), 'positive', hospital, df))

            
def counter(env, df, col_name):
    '''Helper function to track the number of patients admitted, released, and refused at a hospital'''
    loc = df.columns.get_loc(col_name)
    l = [0] * len(df.columns)
    l[loc] = 1
    
    if env.now not in df.index:
        df.loc[env.now] = l
    else:
        df.loc[env.now][col_name] += 1

### Run Simulation


# Separate the hospitals that do not have a service area as overflow hospitals
hospitals_overflow = hospitals_gdf[hospitals_gdf['cases'].isna()]
hospitals_sim = hospitals_gdf.dropna(subset=['cases'])

# Start timing the process
start_time = time.time()

# Run the simulation for each hospital in the GeoDataFrame and add them to a dictionary
sim_dict = dict()
for index, row in hospitals_sim.iterrows():
    # Set the simulation environment
    env = simpy.Environment()
    
    # Definte the environment parameters
    start_patients = int(row['capacity'] * 0.7419)
    num_beds = row['capacity']
    new_patients = int(row['avgnewcases'] * 0.19)
    
    # Create an empty admitted, released, refused (ARR) dataframe
    hospital_df = pd.DataFrame(columns=['admitted', 'released', 'refused'])
    
    # Set up the simulation with parameters from the above variables
    env.process(setup(env, start_patients, num_beds, new_patients, hospital_df))
    
    # Run the simulation for 60 days
    # 61 is used since the simulation starts at day 0 and ends at sim_days - 1
    sim_days = 61
    env.run(until=sim_days)
    
    # Add the hospital's ARR dataframe to the dictionary
    hosp_name = row['name']
    sim_dict[hosp_name] = hospital_df

# Print the execution time of the above code
end_time = time.time()
print("Code Execution Time: ", end_time - start_time)

## Results

### Comparison Graphs


# Create a list of hospitals
hospitals = ['ELMHURST HOSPITAL CENTER', 'NEW YORK METHODIST HOSPITAL']

# Set up two side-by-side plots
fig = plt.figure(figsize=(15, 11))
for i, hospital in enumerate(hospitals, start=1):
    # Plot the number of patients admitted, released and refused
    ax = fig.add_subplot(2, 2, i)
    df = sim_dict[hospital].copy()
    df[['admitted', 'released', 'refused']].iloc[1:].plot(ax=ax, legend=False)

    # Add titles
    hosp_capacity = hospitals_geom.loc[hospitals_geom['name'] == hospital]['capacity'].item()
    title = '{}, Capacity: {}'.format(hospital.title(), int(hosp_capacity))
    plt.title(title, size=15)
    plt.xlabel('Day')
    plt.ylabel('Number of Patients')
    
    # Add a legend for the first two plots
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='center right', bbox_to_anchor=(0.81, 0.75))
    
    # Plot the number of beds filled and the capacity
    ax = fig.add_subplot(2, 2, i+2)
    df['filled'] = (df['admitted'] - df['released'].shift(1, fill_value=0)).cumsum()
    df['capacity'] = int(hosp_capacity)
    df[['capacity', 'filled']].plot(ax=ax, legend=False)
    
    # Add titles
    plt.title(title, size=15)
    plt.xlabel('Day')
    plt.ylabel('Number of Beds Filled')
    
    # Add a legend for the bottom two plots
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='center right', bbox_to_anchor=(0.81, 0.33))

plt.show()


# For each hospital in the dictionary, append the column total to a list
hosp_list = list()
for hospital in sim_dict:
    # Get the sum of each column as a 3 row series
    df = sim_dict[hospital].copy()
    hosp_summary = df.sum(axis=0)
    # Get the hospital ID as the series name
    hosp_id = hospitals_sim.loc[hospitals_sim['name'] == hospital]['id'].item()
    hosp_summary.name = hosp_id
    # Append to list of DataFrames
    hosp_list.append(hosp_summary)
# Combine list of DataFrames into 1
hosp_summary_df = pd.concat(hosp_list, axis=1).transpose()

# Join the new columns to the hospital GeoDataFrame
hosp_sim_gdf = hospitals_sim.join(hosp_summary_df, on='id')

hosp_sim_gdf.head()


# Set up figure
fig, ax = plt.subplots(figsize=(14, 14))

# Add title
title = 'Positive Cases per 1,000 People By Zip Code, Total Cases: ' + str(total)
plt.title(title, size='16')

# Create the colorbar legend
divider = make_axes_locatable(ax)
cax = divider.append_axes('bottom', size='5%', pad=0.1)
cax.tick_params(labelsize=16)
ax.tick_params(left=0, labelleft=0, bottom=0, labelbottom=0)
legend_kwds = {'label': 'Number of Confirmed COVID-19 Cases',
               'orientation': 'horizontal'}

# Plot the ZCTA polygons by COVID-19 cases
column = zcta_gdf[day] / zcta_gdf['TotalPop'] * 1000
zcta_gdf.plot(ax=ax, column=column, cmap='viridis_r', edgecolor='black', 
              legend=True, cax=cax, legend_kwds=legend_kwds)

# Add the two hospital layers
hosp_sim_gdf[hosp_sim_gdf['refused'] == 0].plot(
    ax=ax, marker='o', markersize=30,
    color='pink', edgecolor='black', linewidth=.3)

hosp_sim_gdf.plot(
    ax=ax, marker='o', markersize=hosp_sim_gdf['refused']*1.5, 
    color='red', alpha=0.7, edgecolor='black', linewidth=.3)

# Add the hospitals legend
ax.legend(labels=['Hospitals - None Refused', 'Hospitals - Refused'], 
          loc='lower right', prop={'size': 15}, markerscale=0.6)

plt.show()
