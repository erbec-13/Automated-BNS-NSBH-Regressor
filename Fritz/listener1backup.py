# Author: Ethan Erb
# Listener 1
# Purpose: This script will run a listener that collects events from new ZTF alerts. The events' ids and predictions will be saved to the disk
# for future reference. The prediction being used is Natalya Pletskova's machine learning forecast model.

import os
import numpy as np
import pandas as pd
import traceback
from gcn_kafka import Consumer
from ligo.gracedb.rest import GraceDb
from utils import (
    parse_gcn,
    get_params,
    predict_with_uncertainty,
    plot_all_light_curves_with_uncertainty,
)
from joblib import load
from Model import build_lstm_model
import json
from tensorflow.keras.models import load_model
#import traceback
#import faulthandler
#faulthandler.enable()
import socket

def force_ipv4():
    orig_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4_only(*args, **kwargs):
        return [info for info in orig_getaddrinfo(*args, **kwargs) if info[0] == socket.AF_INET]

    socket.getaddrinfo = getaddrinfo_ipv4_only

force_ipv4()

os.environ['GCN_CLIENT_ID'] = '16ijqredn34sh4gn539vn5bav4'
os.environ['GCN_CLIENT_SECRET'] = 'g1e6rilm1lo0v38rnj86r93qr5gbvqoa1jqnkgp57m61jc7snkb'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#
GCN_CLIENT_ID = os.getenv("GCN_CLIENT_ID")
GCN_CLIENT_SECRET = os.getenv("GCN_CLIENT_SECRET")
GCN_GROUP_ID = str(os.getenv("GCN_GROUP_ID", "oraclefritzbot"))
if not GCN_CLIENT_ID or not GCN_CLIENT_SECRET:
    raise ValueError(
        "GCN_CLIENT_ID and GCN_CLIENT_SECRET must be set as environment variables"
    )

config = {'group.id': 'bnsAndnsbhLCforslack', 'auto.offset.reset': 'earliest', 'enable.auto.commit': False}
consumer = Consumer(
    config=config,
    client_id=GCN_CLIENT_ID,
    client_secret=GCN_CLIENT_SECRET,
    domain="gcn.nasa.gov",
)
consumer.subscribe(
    [
        "gcn.classic.voevent.LVC_PRELIMINARY",
        "gcn.classic.voevent.LVC_INITIAL",
        "gcn.classic.voevent.LVC_UPDATE",
    ]
)

client = GraceDb()

superevent_id = "S250818k"

# Choose the VOEvent file you want
filename = "S250818k-4-Update.xml"

# Download the file content
response = client.files(superevent_id, filename)

# Save it locally if needed
with open(filename, "wb") as f:
    f.write(response.read())

print(f"Downloaded {filename}")

voevent = client.files(superevent_id, filename)
content = voevent.read()


# get the base directory, as the directory where this file is located
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)
print("Loading model and scalers...")
target_scaler = load(f"{base_dir}/target_scaler_O4.joblib")
feature_scaler = load(f"{base_dir}/feature_scaler_O4.joblib")
print("Building LSTM model once...")
num_time_points = 30
input_shape = (num_time_points, feature_scaler.transform(np.zeros((1, 6))).shape[1])
#model = build_lstm_model(input_shape=input_shape)
#model = load(f"{base_dir}/LSTMpredLC__PAstro.joblib")
model = load_model(
    f"{base_dir}/LSTM_model_production.h5",
    compile=False  # Ignore optimizer, loss, and metrics
)
print(feature_scaler.feature_names_in_)


counter = 0
while True:
    print(counter)
    try:
        for message in consumer.consume():
            value = message.value()
            print('a')
            parsed = parse_gcn(content)
            print('b')
            params = get_params(parsed)
            print('c')
            superevent_id, event_page, alert_type, group, prob_bbh, prob_bns, prob_nsbh, far_format, distmean, area_90, longitude, latitude, has_ns, has_remnant, has_mass_gap, significant, prob_ter, skymap, PAstro, time = params

            print(superevent_id + " " + alert_type)
            counter+=1
            if alert_type != "RETRACTION" and distmean != "error": #superevent_id == "S250818k" and 
                print(f"Processing {superevent_id} ({alert_type})")
                
                X = np.vstack((area_90, distmean, has_ns, has_remnant, has_mass_gap, PAstro)).T

                print(X)

                # Time array
                t_min = 0.1
                t_max = 6.0
                dt = 0.2
                time_single = np.linspace(t_min, t_max, num_time_points)
                filter_order = 3

                # Standardize the target data
                X_new = feature_scaler.transform(X)

                # Reshape X data for LSTM input based on the model's input shape
                X_new_reshaped = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
                #X_new_reshaped = X_new.reshape((X_new.shape[0], 90, X_new.shape[1]))

                #X_tiled = np.tile(X_new, (1, num_time_points)).reshape((X_new.shape[0], num_time_points, X_new.shape[1]))


                n_mc_samples = 1000
                mean_preds_new, uncertainty_new = predict_with_uncertainty(model, X_new_reshaped, n_iter=n_mc_samples)

                # Reshape the mean predictions to match the shape used during scaling (num_samples, num_time_points * num_filters)
                mean_preds_flat = mean_preds_new.reshape(mean_preds_new.shape[0], num_time_points * 3)

                # Invert the standardization for the mean predictions
                mean_preds_inverted = target_scaler.inverse_transform(mean_preds_flat).reshape(mean_preds_new.shape[0], num_time_points, 3)

                # Reshape uncertainty to match mean_preds_inverted shape
                uncertainty_reshaped = uncertainty_new.reshape(uncertainty_new.shape[0], num_time_points, 3)

                event_data = {"time_single": time_single.tolist(), "mean_preds_inverted": mean_preds_inverted.tolist(), "uncertainty_reshaped": uncertainty_reshaped.tolist(), "time": time, "alert_type": alert_type}
                
                file_path = "events.json"

                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        events = json.load(f)
                else:
                    events = {}
                
                events[superevent_id] = event_data

                with open(file_path, "w") as f:
                    json.dump(events, f, indent=2)
                
                plot_all_light_curves_with_uncertainty(
                    time_single,
                    mean_preds_inverted,
                    uncertainty_reshaped,
                    superevent_id,
                    time
                )
            else:
                counter-=1


    except Exception as e:
        if counter != 0:
            print(e)
            traceback.print_exc()
        continue

consumer.close()