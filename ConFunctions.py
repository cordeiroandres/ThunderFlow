# -*- coding: utf-8 -*-
"""
@author: deiro
"""

import numpy as np
import pandas as pd
from numba import njit
import requests
import json
import math

import os
from pathlib import Path
import zipfile
import glob
import rasterio
from rasterio.merge import merge

import polyline as pc


from ConstantsConsumption import (
    COP,
    TARGET_TEMP,
    GRAVITY,

    ZONE_LAYERS,
    ZONE_SURFACE,
    LAYER_CONDUCTIVITY,
    LAYER_THICKNESS,

    PATH,
    FOLDER_DIR,
    URL,
    TILES_DIR,
    MOSAIC,
    TILE_SRTM,      
    PASSENGER_SENSIBLE_HEAT,
    AIR_CABIN_HEAT_TRANSFER_COEF,
    AIR_FLOW,
    WIND_SPEED,
    CABIN_VOLUME,
    
    BATTERY_CHARGE_EFF,
    BATTERY_DISCHARGE_EFF,
    TRANSMISSION_EFF,    
    PASSENGER_NR,
    
    EFFMOT,
    PAUX,
    R,
    RHO,
    A,
    CD,
    EBATCAP,
    REGEN_RATIO,
    AUXILIARY_POWER,
    DRAG_COEFFICIENT,
    AIR_DENSITY,
    LOAD_FRACTION,
    MOTOR,
    GENERATOR,    
    M_I,
    VEHICLE_MASS,
    FRONTAL_AREA,
    P_MAX,
    T_RNG,
    CP_RNG,   
    S_RC, 
    COLUMNS_REQ         
)

global SRC,DEM_DATA

def check_required_columns(dataframe,MapMatching=None):
    required_columns = True
    for column in COLUMNS_REQ:
        if column not in dataframe.columns:
            print('You need this column to do the calculation',column)
            required_columns = False
            return required_columns,dataframe
    try: 
        dataframe['lat'] = dataframe['lat'].astype(float)
        dataframe['lon'] = dataframe['lon'].astype(float)
    except ValueError:
        print("The column 'lon' on 'lat' does not contain numerical values.")
        required_columns = False
        return required_columns,dataframe
    
    if 'uid' not in dataframe.columns:
        dataframe['uid'] = 0
    if MapMatching:
        srtm_assign(dataframe)
    else:        
        if 'elevation' not in dataframe.columns:
            srtm_assign(dataframe)        
    
    try:
        dataframe['ts'] = pd.to_datetime(dataframe['ts'], format='%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        print("The column 'ts' does not contain datetime values.")
        required_columns = False
        return required_columns,dataframe
    
        
    return required_columns,dataframe
    
def check_columns(dataframe, column_req):
    if column_req in dataframe.columns:
        return True
    else:
        return False
    
@njit(cache=True,fastmath=True)
def spherical_distance(a_lat, a_lng, b_lat, b_lng):
    # returns distance between a and b in meters
    #Radius = 6371  # earth radius in km
    Radius = 6371000 #meters

    a_lat = np.radians(a_lat)
    a_lng = np.radians(a_lng)
    b_lat = np.radians(b_lat)
    b_lng = np.radians(b_lng)

    d_lat = b_lat - a_lat
    d_lng = b_lng - a_lng

    d_lat_sq = np.sin(d_lat / 2) ** 2
    d_lng_sq = np.sin(d_lng / 2) ** 2

    a = d_lat_sq + np.cos(a_lat) * np.cos(b_lat) * d_lng_sq
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return Radius * c

def add_last_row(df,temporal_thr=1200):
    last_row = df[-1:].copy()
    last_row['ts'] = last_row['ts'] + pd.Timedelta(temporal_thr*2, unit='sec')
    df = pd.concat([df,last_row])
    return df

def time_difference(df,time_thr=1200,SegmentationExists=False):
    prev_id = df['uid'].shift(1).fillna(0)
    prev_date = df['ts'].shift(1).fillna(pd.Timestamp('1900'))
    if SegmentationExists:
        prev_user_progressive = df['user_progressive'].shift(1).fillna(-1)        
        conditions = [((df['uid'].values == prev_id) & (df['user_progressive'] == prev_user_progressive))]
        choices = [(df['ts'] - prev_date).dt.total_seconds()]           
    else:
                
        conditions = [((df['uid'].values == prev_id) & ((df['ts']-prev_date).dt.total_seconds() <= time_thr))]
        choices = [(df['ts'] - prev_date).dt.total_seconds()]    

    df['ts_dif']= np.select(conditions,choices,default=0)
    
    return df

def create_trajectory(df,temporal_thr,spatial_thr,minpoints):          
    traj_new = list()    
    traj = list()
    first_iteration = True  
    i = 1      
    #for row in df.itertuples(index=False):
    for row in df.itertuples(index=False):
        next_p = row        
        if first_iteration:
            uid = row.uid           
            p = row                        
            p = p._replace(user_progressive=i)
            #break
            traj = [p]              
            first_iteration = False 
        else: 
            temporal_dist = next_p.ts_dif            
           #calculates distance between two points        
            spatial_dist = spherical_distance(p.lat,p.lon,next_p.lat,next_p.lon)
           
            if uid!=next_p.uid:
                i = 1
                uid = next_p.uid
               
            if temporal_dist == 0.0:                                    
                if len(traj) >= minpoints:                                    
                    traj_new.extend(traj)                                            
                    traj=[]                                         
                    uid = traj_new[-1].uid                     
                    if uid==next_p.uid:
                        i += 1                                                          
                    next_p = next_p._replace(user_progressive=i)
                    p = next_p 
                    traj.append(p) 
                    continue
                else:                    
                    #insufficient number of points
                    traj=[]                        
                    next_p = next_p._replace(user_progressive=i)
                    traj.append(next_p)
                    p = next_p                 
                    continue
            
            if spatial_dist > spatial_thr:                                                                                         
                next_p = next_p._replace(user_progressive=i)
                p = next_p                
                traj.append(p)                                       
               
    return traj_new

def create_trajectories(df,temporal_thr=1200,spatial_thr=10,minpoints=4):     
    df = add_last_row(df)    
    df = time_difference(df,temporal_thr)
    df['user_progressive'] = 0
    tj_new = create_trajectory(df,temporal_thr,spatial_thr,minpoints)          
    df_traj = pd.DataFrame(tj_new)
    return df_traj


#Functions for downloading the elevation tiles
def srtm3_tile(lon, lat):
    ilon, ilat = int(math.floor(lon)), int(math.floor(lat))
    x , y = (ilon + 180) // 5 + 1, (64 - ilat) // 5 
    tile_name = str(x).zfill(2)+"_"+str(y).zfill(2)
    return tile_name

def download_url(args):        
    url, fn, tif = args[0], args[1], args[2]
    print("Dowloading file: ",tif)
    try:
        r = requests.get(url)
        with open(fn, 'wb') as f:
            f.write(r.content)
        return(url,tif)
    except Exception as e:
        print('Exception in download_file():', e)

def download_parallel(args):    
    #cpus = cpu_count()    
    #results = ThreadPool(cpus - 1).imap_unordered(download_url, args)
    #for result in results:
        #print('Downloaded file:', result[2], 'time(s):', result[1])
    for arg in args:
        download_url(arg)

def unzip_files(dir_name):
    extension = ".zip"
    os.chdir(dir_name) # change directory from working dir to dir with files
    for item in os.listdir(dir_name): # loop through items in dir
        if item.endswith(extension): # check for ".zip" extension
            file_name = os.path.abspath(item) # get full path of files
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall(dir_name) # extract file to dir
            zip_ref.close() # close file
            os.remove(file_name) # delete zipped file
            
def merge_tile():   
    out_ph = os.path.join(TILES_DIR, MOSAIC)    
    # Make a search criteria to select the DEM files
    search_criteria = "s*.tif"
    q = os.path.join(TILES_DIR, search_criteria)
    dem_fps = glob.glob(q)
    src_files_to_mosaic = []
    for fp in dem_fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
        
    mosaic, out_trans = merge(src_files_to_mosaic)
    # Copy the metadata
    out_meta = src.meta.copy()
    
    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                     }
                    )
    # Write the mosaic raster to disk
    with rasterio.open(out_ph, "w", **out_meta) as dest:
        dest.write(mosaic)

def create_srtm_title(list_dem): 
    zips = []
    urls = []
    tifs = []
    tiles_to_dowload = []
    
    #Checks if the directory is created
    if not os.path.exists(TILES_DIR):
        os.makedirs(TILES_DIR)
    
    #check if there is a zip files
    try:
        unzip_files(TILES_DIR)
    except Exception as e:
        print(e)       
    
    for i in range(0, len(list_dem)):
        val=str(list_dem[i])                
        srtm_tiff= TILE_SRTM+val+'.tif'                
        dir_tif = os.path.join(PATH, FOLDER_DIR, srtm_tiff)                
        #Get the tile names
        path = Path(dir_tif)    
        #checks if the files were downloaded
        if not path.is_file():
            srtm_zip = TILE_SRTM+val+'.zip'
            dir_zip = os.path.join(PATH, FOLDER_DIR, srtm_zip)
            zips.append(dir_zip)              
            url_tif = URL.format(val)            
            urls.append(url_tif)
            name_tif = val+'.tif' 
            tifs.append(name_tif) 
    
    tiles_to_dowload = list(zip(urls,zips,tifs))     
    if len(tiles_to_dowload) > 0:
        download_parallel(tiles_to_dowload)
        unzip_files(TILES_DIR)
        merge_tile()
       
def srtm_assign(df):
    vec_tile = np.vectorize(srtm3_tile, cache=False )
    srtm1 = vec_tile(df['lon'].values, df['lat'].values)    
    list_dem = np.unique(srtm1)    
    tifs = create_srtm_title(list_dem)
    return tifs

#General 
def speed_calculation(df):
    column_req = 'distance'
    distance = check_columns(df, column_req)
    if not distance:
        df = distance_difference(df)
    
    conditions = [(df['ts_dif'] > 0)] 
    choices = [(df['distance']/df['ts_dif'])]
    df['speed']= np.select(conditions,choices,default=0) 
    return df

def time_calculation(df):
    conditions = [(df['speed'] > 0)] 
    choices = [(df['distance']/df['speed'])]
    df['ts_dif']= np.select(conditions,choices,default=0) 
    return df

def distance_difference(df,IsRecontruct=False):    
    prev_lat = df['lat'].shift(1).fillna(-4)
    prev_lon = df['lon'].shift(1).fillna(-4)
    vec_spherical_dist = np.vectorize(spherical_distance, otypes=[float] )
            
    if IsRecontruct:        
        conditions = [(prev_lat != -4) & (prev_lon != -4)]                  
    else:
        conditions = [(df['ts_dif'] > 0)]
            
    choices = [vec_spherical_dist(prev_lat,prev_lon,df['lat'],df['lon'])]
    df['distance']= np.select(conditions,choices,default=0)    
    return df    

def acceleration(VN,V0,t):  # V0, VN m/s
    """
    Calculate and returns acceleration.
    Args:
        V0 (float): Old speed.
        VN (float): New speed.
        t (float): time difference.
    Returns:
        float: Acceleration.
    """    
    acc = 0
    if t > 0.0:
        acc = (VN - V0) / t  # acc m/s**2        
    
    return acc

def calculate_acceleration(df):    
    #Calculates the mean velocity from one point to the next            
    prev_speed = df['speed'].shift(1).fillna(0)              
    #Calculates the acceleration from one point to the next    
    vec_acc = np.vectorize(acceleration, otypes=[float] )    
    conditions = [ (df['ts_dif'] > 0) ]   
        
    choices = [vec_acc(df['speed'],prev_speed,df['ts_dif'])]
    df['acceleration']= np.select(conditions,choices,default=0)    
    
    return df

def get_elev(latitude, longitude, SRC):
    elevation = 0        
    row, col = SRC.index(longitude, latitude)
    elevation = DEM_DATA[row,col]  
    
    return elevation

def assign_elevation(df):
    vec_elevation = np.vectorize(get_elev, cache=False)
    
    tile = os.path.join(TILES_DIR,MOSAIC)
    dem_file = Path(tile)
    global SRC,DEM_DATA
    
    #checks if the file exist
    if dem_file.is_file():
        SRC = rasterio.open(tile)
        DEM_DATA = SRC.read(1)
    
    df['elevation'] = vec_elevation(df['lat'], df['lon'], SRC)
        
    return df

def angle(elev,nxt_elev,dist):    
    agl = 0
    if dist != 0:
        Slope = (nxt_elev - elev)/dist
        if Slope < -1 or Slope > 1:
            agl = 0            
        else:            
            agl = np.arcsin(Slope)                
            
    return agl

def calculate_slope(df):    
    prev_elev = df['elevation'].shift(1).fillna(0)    
    vec_angle = np.vectorize(angle, otypes=[float] )
    conditions = [(df['distance'] > 0)]    
    choices = [vec_angle(prev_elev,df['elevation'],df['distance'])]
    df['angle']= np.select(conditions,choices,default=0)    
    return df

@njit(cache=True,fastmath=True)
def BatteryConsumption (timeGap, speed, accel, alpha):
    # Resistenza al rotolamento
    Frr = R * (VEHICLE_MASS) * GRAVITY * math.cos(alpha)
    # Resistenza areodinamica
    Fa = 0.5 * A *CD * RHO * math.pow(speed, 2)
    # Gravità
    Fgo = (VEHICLE_MASS) * GRAVITY * math.sin(alpha)
    # Forza d'inerzia
    Fi = 1.05 * (VEHICLE_MASS) * accel
    # Forza totale
    Ftot = Frr + Fa + Fgo + Fi
    # Forza trazione meccanica
    Ptot = Ftot * speed
    
    PmotOut = 0
    if (Ptot >= 0):
        PmotOut = Ptot/TRANSMISSION_EFF
    else:
        PmotOut = REGEN_RATIO * Ptot * TRANSMISSION_EFF
    
    PmotIn = 0
    if (PmotOut >= 0):
        PmotIn = PmotOut/EFFMOT
    else:
        PmotIn = PmotOut*EFFMOT
    
    Pbat = PmotIn + PAUX
    
    # Modellazione batteria
    eBat = 0
    if(Pbat >= 0):
        eBat = Pbat * timeGap/BATTERY_DISCHARGE_EFF
    else:
        eBat = Pbat * timeGap * BATTERY_CHARGE_EFF
    
    # Calcolo DeltaSoC
    kWh2Ws = 36000
    deltaSoC = eBat/(EBATCAP*kWh2Ws)  
    #deltaSoC = eBat/(kWh2Ws)  
    
    return deltaSoC

def calculate_consumption(df):         
    vec_javacon = np.vectorize(BatteryConsumption, otypes=[float] )
    conditions = [(df['distance'] > 0) ]    
    
    choices = [vec_javacon(df['ts_dif'],df['speed'],df['acceleration'],df['angle'])]
    df['con']= np.select(conditions,choices,default=0)    
    return df

#Funcionts for map matching 
def decode(encoded):
  inv = 1.0 / 1e6
  decoded = []
  previous = [0,0]
  i = 0
  #for each byte
  while i < len(encoded):
    #for each coord (lat, lon)
    ll = [0,0]
    for j in [0, 1]:
      shift = 0
      byte = 0x20
      #keep decoding bytes until you have this coord
      while byte >= 0x20:
        byte = ord(encoded[i]) - 63
        i += 1
        ll[j] |= (byte & 0x1f) << shift
        shift += 5
      #get the final value adding the previous offset and remember it for the next
      ll[j] = previous[j] + (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
      previous[j] = ll[j]
    #scale by the precision and chop off long coords also flip the positions so
    #its the far more standard lon,lat instead of lat,lon
    decoded.append([float('%.6f' % (ll[1] * inv)), float('%.6f' % (ll[0] * inv))])
  #hand back the list of coordinates
  return decoded

def MapMatching_Valhalla(df):
    df_trip_for_meili = df[['lon', 'lat', 'ts']].copy()
    df_trip_for_meili.columns = ['lon', 'lat', 'time']
    # Preparing the request to Valhalla's Meili
    meili_coordinates = df_trip_for_meili.to_json(orient='records')
    meili_head = '{"shape":'
    meili_tail = """, "shape_match":"map_snap", "costing":"auto","verbose":true,
    "filters":{"attributes":["shape","matched.point","matched.type","matched.lat","matched.lon", 
    "matched.edge_index","node.elapsed_time","edge.way_id","edge.speed","edge.names","edge.length",
    "edge.begin_shape_index","edge.end_shape_index"],"action":"include"},"format":"osrm"}"""
    meili_request_body = meili_head + meili_coordinates + meili_tail
    # Sending a request to Valhalla's Meili
    url = "http://localhost:8002/trace_attributes"
    headers = {'Content-type': 'application/json'}
    data = str(meili_request_body)      
    r = requests.post(url, data=data, headers=headers)    
    return r

def MapMatching_Valhalla_traj(df):
    SuccessMapMatching = False
    try:
        r  = MapMatching_Valhalla(df)   
        if r.status_code == 200: 
            SuccessMapMatching = True
            response_text = json.loads(r.text)
            #Converts to dataframe the matched points from the map matching
            search_1 = response_text.get('matched_points')  
            edges = response_text.get('edges') 
            df_mat_pts = pd.json_normalize(data=search_1) 
            
            df_edges = pd.json_normalize(data=edges) 
            df_edges['length'] = df_edges['length']*1000
            
            polyline6 = response_text.get('shape')
            MapMatchingRoute = decode(polyline6)
            index = ['lon','lat']
            df_mpr = pd.DataFrame(MapMatchingRoute, columns=index) 
                
            df_edges['diff'] = df_edges['end_shape_index'] - df_edges['begin_shape_index']
            grouped_way_id = df_edges.groupby('way_id',sort=False).agg({'diff': 'sum','speed':'sum'}).reset_index()
            # Count the occurrences of each 'way_id'
            way_id_counts = df_edges.groupby('way_id',sort=False).size().reset_index()
            way_id_counts = way_id_counts.rename(columns={0: 'Cnt'})
            grouped_way_id['count'] = way_id_counts['Cnt']
            grouped_way_id['spd_avg'] = grouped_way_id['speed'] /grouped_way_id['count'] 
            
            list_way_id=[]
            list_speed=[]
            
            list_way_id.append(grouped_way_id.way_id[0])
            list_speed.append(grouped_way_id.spd_avg[0])
            
            for row in grouped_way_id.itertuples():
                l=row.diff        
                for i in range(l):
                    list_way_id.append(row.way_id) 
                    list_speed.append(row.spd_avg) 
            
            df.rename(columns = {'lon':'lon_old', 'lat':'lat_old'}, inplace = True)        
            df_pts = pd.merge(df, df_mat_pts, left_index=True, right_index=True)
            uid = df.uid[0]
            df_mpr['way_id']=list_way_id
            df_mpr['speed']=list_speed            
            df_mpr['uid'] =  uid
            df_mpr['user_progressive'] = df_pts.user_progressive[0]
            df_mpr['ts'] = df_pts.ts[0]
            df_mpr['type'] = 'interpolated'
            df_mpr.at[0,'speed'] = df_pts.speed[0]
            df_mpr.at[0,'type'] = 'matched'
            df_mpr.at[len(df_mpr)-1,'type'] = 'matched'
            df_mpr.at[len(df_mpr)-1,'ts'] = df_pts.ts[len(df_pts)-1]
            df_pts = df_pts.dropna().reset_index()
            lst_edge = df_pts.edge_index
            
            for i in range(1, len(lst_edge)-1):
                f = lst_edge[i]    
                c = df_edges.end_shape_index[f]-1    
                df_mpr.loc[c, 'type'] = 'matched' 
                df_mpr.loc[c, 'ts'] = df_pts.ts[i]
                df_mpr.loc[c, 'speed'] = df_pts.speed[i]            
            
            return SuccessMapMatching,df_mpr            
        else:
            #print(r)
            return SuccessMapMatching,df            

    except Exception as e:
        print('Error at map matching ',e)
        return SuccessMapMatching,df 

def MapMatching_OSRM (lat,lon,lat1,lon1):
    osrm_request = ""                
    osrm_request = "{},{};".format(lon, lat)+"{},{}".format(lon1, lat1)
    osmrm_base = "http://localhost:5000/route/v1/driving/"        
    osrm_call = requests.get (osmrm_base + osrm_request, params =  {"annotations":"true","overview": "full"} )    
    
    return osrm_call

def add_time(traj_new):   
    is_first = True
    for i,row in traj_new.iterrows():                
        if is_first:
            ts = traj_new.at[i, 'ts'] 
            is_first = False
        else:           
            ts = ts + pd.to_timedelta(traj_new.at[i, 'ts_dif'], unit='s') 
            traj_new.at[i, 'ts'] = ts
    
    return traj_new

def MapMatching_seq_osrm(df):
    df_mpr = []
    uid = df.uid[0]
    ts = df.ts[0]
    vel = df.speed[0]
    osrm_call = MapMatching_OSRM(df.lat[0],df.lon[0],df.lat[1],df.lon[1])
    if osrm_call.status_code == 200: 
        geometry =  osrm_call.json()['routes'][0]['geometry']
        #waypoints = pc.PolylineCodec().decode(geometry) 
        waypoints = pc.decode(geometry)
        #nodes = osrm_call.json()['routes'][0]['legs'][0]['annotation']['nodes']
        speed = osrm_call.json()['routes'][0]['legs'][0]['annotation']['speed']
        time = osrm_call.json()['routes'][0]['legs'][0]['annotation']['duration']
        dist = osrm_call.json()['routes'][0]['legs'][0]['annotation']['distance']        
        index = ['lat','lon']
        df_mpr = pd.DataFrame(waypoints, columns=index)
        time = [t+0.1 for t in time]
        speed.insert(0, vel)
        time.insert(0, 0)
        dist.insert(0, 0)
        df_mpr['speed'] = speed
        df_mpr['speed'] = df_mpr["speed"]/3.6
        #df_mpr.at[0,'speed'] = vel        
        df_mpr['ts_dif'] = time     
        df_mpr['distance'] = dist
        #df_mpr['nodes'] = nodes
        df_mpr['type'] = 'interpolated'    
        df_mpr['type'][0] ='matched'
        df_mpr['type'][len(df_mpr)-1] ='matched'
        df_mpr['PtOrigin'] = np.where(df_mpr['type'].values == 'interpolated',False,True)
        df_mpr['uid'] = uid
        df_mpr['ts'] = ts
        df_mpr['ts'] = pd.to_datetime(df_mpr['ts'])
        df_mpr = add_time(df_mpr)
        
    
    return df_mpr

def MapMatching_OSRM_traj(dfa):
    dfinal = []
    MapMatchingOsrm=False
    for i in range(len(dfa)-1):
        df = []    
        dfo = []     
        dfo = dfa.iloc[i:i+2]     
        dfo = dfo.reset_index(drop=True)    
        df = MapMatching_seq_osrm(dfo) 
        dfinal.append(df)    
    
    if len(dfinal)>0:
        MapMatchingOsrm=True
    df_osrm = pd.concat(dfinal, ignore_index=True)     
    df_osrm = df_osrm.drop_duplicates(subset=['lat', 'lon'], keep='first',ignore_index=True)
    return MapMatchingOsrm,df_osrm
    

def consumption_traj(df,MapMatching=None):
    column_req = 'ts_dif'
    tsdif = check_columns(df, column_req)
    column_req = 'speed'
    speed = check_columns(df, column_req)
    SuccessMapMatching = False
    MapMatchingValhalla = False
    MapMatchingOSRM = False
    
    if not tsdif:
        df = time_difference(df)        
    if not speed:
        df = speed_calculation(df)
    else:    
        df['speed'] = df['speed']/3.6
    
    if MapMatching=='valhalla':       
        MapMatchingValhalla,df = MapMatching_Valhalla_traj(df)
        SuccessMapMatching = MapMatchingValhalla        
    
    df = distance_difference(df,MapMatchingValhalla)   
    if MapMatchingValhalla:            
        df = time_calculation(df) 
        
    if MapMatching=='osrm':       
        MapMatchingOSRM,df = MapMatching_OSRM_traj(df)
        SuccessMapMatching = MapMatchingOSRM
    
    df_mpr = calculate_acceleration(df)        
    df_mpr = assign_elevation(df_mpr)
    df_mpr = calculate_slope(df_mpr)
    df_mpr = calculate_consumption(df_mpr)           

    return df_mpr,SuccessMapMatching
    
def calculation_consumption_traj(dj,
                            lst,
                            MapMatching,
                            ResultsByTrajectory,
                            ResultsByWayId,
                            ResultsByDetailPoints):                                
    lst_wayid = []
    lst_traj = []
    lst_points = []
    
    for i in range(len(lst)-1):    
           dfa=dj.iloc[lst[i]:lst[i+1]].reset_index(drop=True)    
           try:               
               djr,SuccessMapMatching = consumption_traj(dfa,MapMatching)
               if len(djr) > 0:                                          
                   if ResultsByTrajectory:
                       dist = round(djr['distance'].sum()/1000, 3) 
                       con = round(djr['con'].sum(),3)
                       dfj=[djr.uid[0],djr.user_progressive[0],djr.ts[0],djr.ts[len(djr)-1],dist,con ]                            
                       lst_traj.append(dfj)   
                
                   if MapMatching=='valhalla':
                       if SuccessMapMatching:
                           group_wayid = djr.groupby('way_id',sort=False).agg({'con': 'sum','distance':'sum'}).reset_index()                        
                           group_wayid['distance'] = group_wayid['distance']/1000
                           group_wayid['distance'] = group_wayid['distance'].round(3)
                           group_wayid['con'] = group_wayid['con'].round(3)        
                           
                           way_id_list = group_wayid.values.tolist()
                           do = [djr.uid[0],djr.user_progressive[0],djr.ts[0],djr.ts[len(djr)-1], way_id_list]        
                           lst_wayid.append(do) 
                       
                   if ResultsByDetailPoints:                       
                       compressed_row = djr.to_json()
                       lst_points.append(compressed_row)
                   
           except Exception as e: 
               print("Error ",e)
    if ResultsByTrajectory:
        if len(lst_traj)>0:
            col = ['uid','user_progressive','start_date','end_date','distance','consumption']        
            lst_traj = pd.DataFrame(lst_traj, columns=col)
    if MapMatching=='valhalla':
        if len(lst_wayid)>0:
            col = ['uid','user_progressive','start_date','end_date','way_id_list']        
            lst_wayid = pd.DataFrame(lst_wayid,columns=col)            
    
        
    return lst_traj,lst_wayid,lst_points
               
  
def consumption(df,
                CreateTrajectories=None,                               
                temporal_thr=1200,
                spatial_thr=50,
                minpoints=4,
                MapMatching=None,
                ResultsByTrajectory=True,
                ResultsByWayId=False,
                ResultsByDetailPoints=False
                ): 
    required_columns,df=check_required_columns(df,MapMatching)
          
    if required_columns:
        column_req = 'user_progressive'
        user_progressive = check_columns(df, column_req)
        
        if CreateTrajectories:                
            df = create_trajectories(df,temporal_thr,spatial_thr,minpoints)            
        else:
            if user_progressive:
                df = time_difference(df,SegmentationExists=True)
            else:
                df = create_trajectories(df,temporal_thr,spatial_thr,minpoints)
                
        df_traj  = distance_difference(df)
        lst_inter = df_traj[df_traj['distance'] == 0.0].index.tolist()          
        lst_inter.append(len(df_traj))
        
        lst_traj,lst_wayid,lst_pts=calculation_consumption_traj(df_traj,
                                                                lst_inter,
                                                                MapMatching,
                                                                ResultsByTrajectory,
                                                                ResultsByWayId,
                                                                ResultsByDetailPoints)
               
        if MapMatching=='valhalla':
            if ResultsByTrajectory:
                if ResultsByWayId:
                    if ResultsByDetailPoints:
                        return lst_traj,lst_wayid,lst_pts
                    else:
                        return lst_traj,lst_wayid
                else:
                    if ResultsByDetailPoints:
                        return lst_traj,lst_pts
                    else:
                        return lst_traj
            else:
                if ResultsByWayId:
                    if ResultsByDetailPoints:
                        return lst_wayid,lst_pts
                    else:
                        return lst_wayid
                else:
                    if ResultsByDetailPoints:
                        return lst_pts 
        else:                                 
            if ResultsByTrajectory:
                if ResultsByDetailPoints:
                    return lst_traj,lst_pts
                else:
                    return lst_traj
            else:
                if ResultsByDetailPoints:
                    return lst_pts
                
       
    
       
"""    
def calculate_consumption_traj(df,
                               temporal_thr=1200,
                               spatial_thr=50,
                               minpoints=4,
                               GotElevation=False,
                               DoMapMatching=False):        
    if check_required_columns(df):
        dj = create_trajectories(df,temporal_thr,spatial_thr,minpoints)
        df_traj  = distance_difference(dj)
        lst_inter = df_traj[df_traj['distance'] == 0.0].index.tolist()          
        lst_inter.append(len(df_traj))
        lst_traj,lst_wayid = calculation_consumption(df_traj, lst_inter,GotElevation,DoMapMatching)
        
        col = ['uid','user_progressive','start_date','end_date','distance','consumption']        
        data_traj = pd.DataFrame(lst_traj, columns=col)
        if DoMapMatching:
            col = ['uid','user_progressive','start_date','end_date','way_id_list']        
            data_wayid = pd.DataFrame(lst_wayid,columns=col)
            return data_traj,data_wayid
        
        return data_traj,lst_wayid
    
def calculation_consumption(dj,lst,GotElevation=False,DoMapMatching=False):    
    lst_wayid = []
    lst_traj = []
    for i in range(len(lst)-1):    
           dfa=dj.iloc[lst[i]:lst[i+1]].reset_index(drop=True)    
           try:               
               con,djr,SuccessMapMatching = consumption_traj(dfa,GotElevation,DoMapMatching)
               if len(djr) > 0:
                   dist = round(djr['distance'].sum()/1000, 3) 
                   #con = round(djr['con'].sum(),3)
                   dfj=[djr.uid[0],djr.user_progressive[0],djr.ts[0],djr.ts[len(djr)-1],dist,con ]                            
                   lst_traj.append(dfj)    
                                          
                   if DoMapMatching & SuccessMapMatching:
                       group_wayid = djr.groupby('way_id',sort=False).agg({'con': 'sum','distance':'sum'}).reset_index()                        
                       group_wayid['distance'] = group_wayid['distance']/1000
                       group_wayid['distance'] = group_wayid['distance'].round(3)
                       group_wayid['con'] = group_wayid['con'].round(3)        
                       
                       way_id_list = group_wayid.values.tolist()
                       do = [djr.uid[0],djr.user_progressive[0],djr.ts[0],djr.ts[len(djr)-1], way_id_list]        
                       lst_wayid.append(do)                    
                   
           except Exception as e: 
               print("Error ",e)
    return lst_traj,lst_wayid
               
def consumption_traj(df,GotElevation=False,DoMapMatching=False):
    column_req = 'ts_dif'
    tsdif = check_columns(df, column_req)
    column_req = 'speed'
    speed = check_columns(df, column_req)
    con = 0
    SuccessMapMatching = False
    
    if not tsdif:
        df = time_difference(df)
        
    if not speed:
        df = speed_calculation(df)
    else:    
        df['speed'] = df['speed']/3.6
    
    if DoMapMatching:
        srtm_assign(df)
        SuccessMapMatching,df=MapMatching_traj(df)
    else:                
        if not GotElevation:
            srtm_assign(df)
    
    df_mpr = distance_difference(df,DoMapMatching)   
    
    
    
    if DoMapMatching & SuccessMapMatching:            
        df_mpr = time_calculation(df_mpr) 
    
    df_mpr = calculate_acceleration(df_mpr)        
    df_mpr = assign_elevation(df_mpr)
    df_mpr = calculate_slope(df_mpr)
    df_mpr = calculate_consumption(df_mpr)        
    con = round(df_mpr['con'].sum(),3)

    return con,df_mpr,SuccessMapMatching
    
def calculation_consumption(dj,
                            lst,
                            MapMatching,
                            ResultsByTrajectory,
                            ResultsByWayId,
                            ResultsByDetailPoints):                                
    lst_wayid = []
    lst_traj = []
    lst_points = []
    
    for i in range(len(lst)-1):    
           dfa=dj.iloc[lst[i]:lst[i+1]].reset_index(drop=True)    
           try:               
               con,djr,SuccessMapMatching = consumption_traj(dfa,GotElevation,DoMapMatching)
               if len(djr) > 0:
                   dist = round(djr['distance'].sum()/1000, 3) 
                   #con = round(djr['con'].sum(),3)
                   dfj=[djr.uid[0],djr.user_progressive[0],djr.ts[0],djr.ts[len(djr)-1],dist,con ]                            
                   lst_traj.append(dfj)    
                                          
                   if DoMapMatching & SuccessMapMatching:
                       group_wayid = djr.groupby('way_id',sort=False).agg({'con': 'sum','distance':'sum'}).reset_index()                        
                       group_wayid['distance'] = group_wayid['distance']/1000
                       group_wayid['distance'] = group_wayid['distance'].round(3)
                       group_wayid['con'] = group_wayid['con'].round(3)        
                       
                       way_id_list = group_wayid.values.tolist()
                       do = [djr.uid[0],djr.user_progressive[0],djr.ts[0],djr.ts[len(djr)-1], way_id_list]        
                       lst_wayid.append(do)                    
                   
           except Exception as e: 
               print("Error ",e)
    return lst_traj,lst_wayid
"""
