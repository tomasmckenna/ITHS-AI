import os

import csv
import yaml
import json

import pandas as pd
import pickle
from datetime import datetime, timedelta
import geopy.distance

CAR_DISTANCE_THRESHOLD = 150 # if car is closer than  this, it's relevant
CAR_TIME_THRESHOLD_BEFORE = 2 # if car is parked within these limits, it's relevant
CAR_TIME_THRESHOLD_AFTER = 1
MIN_DISTANCE_DEFAULT = 1000000  # 
BUCKETS_FOR_DISTANCE = 6 # num of buckets for distance
BUCKET_FOR_CAR_START_TIME = 12 # num of  buckets for 'time of day'

# Display all columns
pd.set_option('display.max_columns', None)

# ConfigLoader Class
class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get_file_path(self):
        if os.name == 'nt':  # Windows
            return self.config['file_paths']['windows']
        elif os.uname().sysname == 'Darwin':  # macOS
            return self.config['file_paths']['mac']
        else:  # Assume Linux for other systems
            return self.config['file_paths']['linux']

    def get_output_file_path(self):
        if os.name == 'nt':  # Windows
            return self.config['output_file_path']['windows']
        elif os.uname().sysname == 'Darwin':  # macOS
            return self.config['output_file_path']['mac']
        else:  # Assume Linux for other systems
            return self.config['output_file_path']['linux']

# OS
def get_file_path(config):
    if os.name == 'nt':  # Windows
        return config['file_paths']['windows']
    elif os.uname().sysname == 'Darwin':  # macOS
        return config['file_paths']['mac']
    else:  # Assume Linux for other systems
        return config['file_paths']['linux']

def add_new_row(distance_df, visit_id, patient_id, car_start_time, car_end_time, duration_min, distance, span_distance, span_car_start_time, info):
    new_row = pd.DataFrame([{
        'VisitID': visit_id,
        'CareEpisodeID': patient_id,
        'CarStartTime': car_start_time,
        'CarEndTime': car_end_time,
        'DurationMin': duration_min,
        'DistanceM': distance,
        'SpanDistanceM': span_distance,
        'SpanCarStartTime': span_car_start_time,
        'Information': info
    }])

    if not new_row.empty:
        distance_df = pd.concat([distance_df, new_row], ignore_index=True)
    
    return distance_df

def load_data(file_path):
    xls = pd.ExcelFile(file_path)
    finished_occurrences = xls.parse('finishedOccurrences')
    finished_visits = xls.parse('finishedVisits')
    car_trips = xls.parse('carTrips')
    car_trips['location.1.timestamp'] = pd.to_datetime(car_trips['location.1.timestamp'])  # Convert to datetime
    car_trips['location.timestamp'] = pd.to_datetime(car_trips['location.timestamp']) 
    patient_location = xls.parse('pLocation')
    patients_demographics = xls.parse('patientsDemographics')
    demographics_df = pd.json_normalize(patients_demographics['demographics'].apply(json.loads))
    patients_demographics = patients_demographics.drop(columns=['demographics'])
    patient_demographics = pd.concat([patients_demographics, demographics_df], axis=1)
    return finished_occurrences, finished_visits, car_trips, patient_location, patient_demographics

def preprocess_data(finished_visits, patient_location, patient_demographics):
    df = pd.merge(finished_visits, patient_location, how='inner', left_on='CareEpisodeID', right_on='id')
    df = pd.merge(df, patient_demographics, how='left', left_on='CareEpisodeID', right_on='careEpisodeID')
    df['date'] = df['TravelToVisitStarted.StartTime'].dt.date
    df['VisitFinished.event_data.finishedAt'] = pd.to_datetime(df['VisitFinished.event_data.finishedAt']).dt.tz_localize(None)
    return df

def process_daily_data(df, car_trips): #df == preprocessed data # car_trips == car trip data with timestamps and Gps coordinates
    distance_df = pd.DataFrame(columns=['VisitID', 'CareEpisodeID', 'CarStartTime', 'CarEndTime', 'DurationMin', 'DistanceM', 'SpanDistanceM', 'SpanCarStartTime', 'Information']) #matched car trips and visits and stats
    start_date = df['TravelToVisitStarted.StartTime'].min().date()
    end_date = df['TravelToVisitStarted.StartTime'].max().date()
    no_match_count = 0
    min_distance = MIN_DISTANCE_DEFAULT

    for single_date in pd.date_range(start=start_date, end=end_date):#Iterate over all dates
        daily_df = df[df['TravelToVisitStarted.StartTime'].dt.date == single_date.date()]#Filter daily_df for current date
        daily_car_trips = car_trips[car_trips['location.1.timestamp'].dt.date == single_date.date()]#filter car_trips for current date

        if daily_car_trips.empty and not daily_df.empty:#if no car trips on that date, mark day with 'no car trips'
            distance_df = add_new_row(distance_df, 0, 0, single_date, single_date, 0, 0, 0, 0, 'no car trips this date')
            continue

        care_episode_counts = daily_df['CareEpisodeID'].value_counts()#Group visits by CareEpisodeID

        for _, daily_row in daily_df.iterrows():#Loop through each vist in daily_df
            found_trip = False
            patient_id = daily_row['CareEpisodeID']
            patient_location = (daily_row['latitude'], daily_row['longitude'])
            visit_id = daily_row['visitId']
            visit_finished_at = daily_row['VisitFinished.event_data.finishedAt']

            for daily_car_trips_counter in range(len(daily_car_trips) - 1):#Loop through daily_car_trips to find potential matches
                car_trip_row = daily_car_trips.iloc[daily_car_trips_counter]
                car_trip_location = (car_trip_row['location.1.latitude'], car_trip_row['location.1.longitude'])#lat long of car start point
                car_start_time = car_trip_row['location.1.timestamp'].tz_localize(None)  # timezone-naive

                time_window_before = CAR_TIME_THRESHOLD_BEFORE if care_episode_counts[patient_id] > 1 else 24#if multuple visits, use a smaller time window
                time_window_after = CAR_TIME_THRESHOLD_AFTER if care_episode_counts[patient_id] > 1 else 24
                if not (visit_finished_at - timedelta(hours=time_window_before) <= car_start_time <= visit_finished_at + timedelta(hours=time_window_after)):#Skips car trip if start time not within time window
                    continue    

                distance = round(1000 * geopy.distance.geodesic(patient_location, car_trip_location).km)#calcs  distance twixt patient_locatoin and car_trip_location
                
                if distance < CAR_DISTANCE_THRESHOLD:                #if less than CAR_DISTANCE_TRESHOLD constant, it's a match
                    next_car_trip_row = daily_car_trips_counter + 1
                    while next_car_trip_row < len(daily_car_trips) and daily_car_trips.iloc[next_car_trip_row]['id.1'] != car_trip_row['id.1']:
                        next_car_trip_row += 1
                    
                    if next_car_trip_row < len(daily_car_trips):
                        car_end_time = daily_car_trips.iloc[next_car_trip_row]['location.timestamp'].tz_localize(None)  # timezone-naive when trip ended
                        duration_min = (car_end_time - car_start_time).total_seconds() / 60 #trip duration in mins
                        span_distance = min((distance * BUCKETS_FOR_DISTANCE) // CAR_DISTANCE_THRESHOLD + 1, BUCKETS_FOR_DISTANCE - 1) #Group distances into buckets
                        span_car_start_time = min((car_start_time.hour * BUCKET_FOR_CAR_START_TIME) // 25, BUCKET_FOR_CAR_START_TIME - 1) #Group start time into buckets

                        distance_df = add_new_row(distance_df, visit_id, patient_id, car_start_time, car_end_time, duration_min, distance, span_distance, span_car_start_time, "match") #adds matches to distance_df
                        found_trip = True

                        daily_car_trips = daily_car_trips.drop(daily_car_trips.index[daily_car_trips_counter]).reset_index(drop=True)#remove matched trup from daily_car_trips
                        break
                else:
                    min_distance = min(distance, min_distance) #If not less than CAR_DISTANCE_TRESHOLD, update mindistance 
                  
            if not found_trip:
                no_match_count += 1
                distance_df = add_new_row(distance_df, visit_id, patient_id, single_date, single_date, 0, min_distance, 0, 0, 'no match')
                min_distance = MIN_DISTANCE_DEFAULT

    return distance_df

def save_results(df, distance_df, finished_occurrences, output_file_path):
    distance_df.to_excel(output_file_path, index=False, engine='openpyxl')
    df = pd.merge(df, distance_df, how='left', left_on='visitId', right_on='VisitID')
    df = pd.merge(df, finished_occurrences, how='inner', left_on='visitId', right_on='ActivityOccurenceEvents.event_data.visitId')
    df.to_excel(output_file_path, index=False, engine='openpyxl')

# Main
config = ConfigLoader('/home/tomas/GitHub/ITHS-AI/config/config.yaml')
file_path = '/home/tomas/GitHub/ITHS-AI/data/allData2.xlsx'
output_file_path = '/home/tomas/GitHub/ITHS-AI/data/df.xlsx'
finished_occurrences, finished_visits, car_trips, patient_location, patient_demographics = load_data(file_path)
df = preprocess_data(finished_visits, patient_location, patient_demographics)
distance_df = process_daily_data(df, car_trips)
save_results(df, distance_df, finished_occurrences, output_file_path)

#config = load_config('/home/tomas/GitHub/AImed/config/config.yaml')
#  windows: "C:\\Users\\jenny\\Documents\\GitHub\\AImed\\config\\config.yaml"
#  mac: "/Users/tomas/Documents/GitHub/AImed/config/config.yaml"
#  linux: "/home/tomas/GitHub/AImed/config/config.yaml"
