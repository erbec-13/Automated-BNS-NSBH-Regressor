# Author: Ethan Erb (referencing and adapting code from Natalya Pletskova)
# Listener 1
# Purpose: This script will run a listener that collects events from new ZTF alerts. The events' ids and predictions will be saved to the disk
# for future reference. The prediction being used is Natalya Pletskova's machine learning forecast model found in
# LSTM_model_production.h5. This code assumes that the script exists in the same directory as the model,
# the target scaler, the feature scaler, utils.py, and events.json (if it already exists). If there is no
# events.json file, it will be created when the first event is processed.

import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
import socket
import warnings

# Suppress specific warnings
#warnings.filterwarnings("ignore", category=UserWarning)

def force_ipv4():
    orig_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4_only(*args, **kwargs):
        return [info for info in orig_getaddrinfo(*args, **kwargs) if info[0] == socket.AF_INET]

    socket.getaddrinfo = getaddrinfo_ipv4_only

force_ipv4()

# Check to make sure your environment variables are set
GCN_CLIENT_ID = os.getenv("GCN_CLIENT_ID")
GCN_CLIENT_SECRET = os.getenv("GCN_CLIENT_SECRET")
GCN_GROUP_ID = str(os.getenv("GCN_GROUP_ID", "oraclefritzbot"))
if not GCN_CLIENT_ID or not GCN_CLIENT_SECRET:
    raise ValueError(
        "GCN_CLIENT_ID and GCN_CLIENT_SECRET must be set as environment variables"
    )

# Set up the Kafka consumer where we retrieve GCN notices
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

# ~~~ DEBUGGING CODE ~~~
# Uncomment this section to download a specific event from GraceDB for testing purposes
# In the while loop below, pass 'content' to parse_gcn() instead of 'value'

#client = GraceDb()
#
#superevent_id = "S250818k"
#
## Choose the VOEvent file you want
#filename = "S250818k-4-Update.xml"
#
## Download the file content
#response = client.files(superevent_id, filename)

## Save it locally if needed
#with open(filename, "wb") as f:
#    f.write(response.read())

#print(f"Downloaded {filename}")

#voevent = client.files(superevent_id, filename)
#content = voevent.read()

# ~~~ END DEBUGGING CODE ~~~


# get the base directory, as the directory where this file is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the prediction model and the scalers
print("Loading model and scalers...")
target_scaler = load(f"{base_dir}/target_scaler_O4.joblib")
feature_scaler = load(f"{base_dir}/feature_scaler_O4.joblib")
print("Building LSTM model once...")
num_time_points = 30
input_shape = (num_time_points, feature_scaler.transform(np.zeros((1, 6))).shape[1])
model = load_model(
    f"{base_dir}/LSTM_model_production.h5",
    compile=False  # Ignore optimizer, loss, and metrics
)
# Display the feature names used in the model
print(feature_scaler.feature_names_in_)

# Iterate indefinitely over new GCN notices
while True:
    try:
        # Retrieve the next message packet from the Kafka consumer
        for message in consumer.consume():
            # Get the parameters from the GCN notice
            value = message.value()
            parsed = parse_gcn(value)
            if (graceid := next((p['@value'] for p in parsed['voe:VOEvent']['What']['Param'] if p.get('@name') == 'GraceID'), None)) and graceid.startswith("M"):
                continue
            params = get_params(parsed)
            superevent_id, event_page, alert_type, group, prob_bbh, prob_bns, prob_nsbh, far_format, distmean, area_90, longitude, latitude, has_ns, has_remnant, has_mass_gap, significant, prob_ter, skymap, PAstro, time = params

            # Display the event id and alert type
            print(superevent_id + " " + alert_type)

            # Only process non-retraction alerts with valid distance
            if alert_type != "RETRACTION" and distmean != "error":
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

                n_mc_samples = 1000
                mean_preds_new, uncertainty_new = predict_with_uncertainty(model, X_new_reshaped, n_iter=n_mc_samples)

                # Reshape the mean predictions to match the shape used during scaling (num_samples, num_time_points * num_filters)
                mean_preds_flat = mean_preds_new.reshape(mean_preds_new.shape[0], num_time_points * 3)

                # Invert the standardization for the mean predictions
                mean_preds_inverted = target_scaler.inverse_transform(mean_preds_flat).reshape(mean_preds_new.shape[0], num_time_points, 3)

                # Reshape uncertainty to match mean_preds_inverted shape
                uncertainty_reshaped = uncertainty_new.reshape(uncertainty_new.shape[0], num_time_points, 3)

                # Store the event data needed to plot the prediction
                event_data = {"time_single": time_single.tolist(), "mean_preds_inverted": mean_preds_inverted.tolist(), "uncertainty_reshaped": uncertainty_reshaped.tolist(), "time": time, "alert_type": alert_type}
                
                # Identify the events.json file to store the event data to
                file_path = "events.json"

                # Load existing events from events.json if it exists, otherwise create a new dictionary
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        events = json.load(f)
                else:
                    events = {}
                
                # Add or update the event data for the current superevent_id
                events[superevent_id] = event_data

                # Save the updated events dictionary back to events.json (creating the file if it doesn't exist)
                with open(file_path, "w") as f:
                    json.dump(events, f, indent=2)

    # Handle exceptions gracefully and continue listening for new events
    except Exception as e:
        print(e)
        continue

# Close the consumer when done iterating

consumer.close()
