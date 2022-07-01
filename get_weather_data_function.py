#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 11:42:37 2021

@author: Robert_Hennings
"""

# Import Meteostat library und dependencies sowie weiterer libraries
from datetime import datetime

#Building a function to fetch the desired meteostat weather data
#the parameter "data" takes the following possible inputs:
#'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco'
#if "all" is passed then all parameters are fetched
    

def get_weather_data_hourly(lat,lon,start,end,data):
    #importing libraries
    global weather_data
    from datetime import datetime
    from meteostat import Daily
    from meteostat import Hourly
    from meteostat import Stations
        
    stations = Stations()
    stations = stations.nearby(lat, lon)
    station = stations.fetch(1)
        
    weather_data = Hourly(station, start, end)
    weather_data = weather_data.fetch()
    
    if data == "all":
        weather_data = weather_data
    else: 
        angegeben = data
        cols_ausgabe = list(weather_data.columns)
        fertig = [i for i in cols_ausgabe if i not in angegeben]
        weather_data = weather_data.drop(fertig,axis=1)
        
    print(weather_data.head())
    print("Station data:","\n",station.iloc[0,0:14],"\n\n Distance from user given lat lon:", station.iloc[0,13])
    print("Data is included in the global env DataFrame weather_data")
    
#Example
get_weather_data_hourly(lat=54.40949,lon=10.22698,start=datetime(1991,2,2),end=datetime(2020,12,25),data = "all") 


