# Author: Ethan Erb (referencing and adapting code from Natalya Pletskova)
# Listener 1
# Purpose: This script will query the SkyPortal API for new sources in the EM+GW group and check if they
# match any recent GW events in space (within the 90% skymap region) and time (within -3 to +7 days of the event).
# If a source matches with an event, the script will plot the source's photometry onto the event's predicted
# light curve. The prediction uses Natalya Pletskova's machine learning forecast model found in
# LSTM_model_production.h5. The code assumes there is an events.json file in the same directory that contains
# GW events.
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import requests
import time
import json
from datetime import datetime, timedelta, UTC, timezone
import os
from ligo.gracedb.rest import GraceDb
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Set the SkyPortal token as an environment variable for security
SKYPORTAL_TOKEN = os.getenv("SKYPORTAL_TOKEN")
if SKYPORTAL_TOKEN is None:
    raise ValueError("Please set the SKYPORTAL_TOKEN environment variable")

# This function uses Natalya's plotting code from utils.py to plot the source's photometry onto the event's
# predicted light curve
def plot_source_on_event(source_id, event_id):
    # Retrieve the event information from events.json
    with open("events.json") as f:
        event = json.load(f).get(event_id)
    time_array = np.array(event.get("time_single"))
    mean_preds = np.array(event.get("mean_preds_inverted"))
    uncertainty = np.array(event.get("uncertainty_reshaped"))
    superevent_id = event_id
    time = event.get("time")

    # Convert time into MJD
    event_t = Time(time, format='isot', scale='utc')
    event_mjd = event_t.mjd

    # Fetch photometry data for the source from SkyPortal
    photometry_url = f"https://fritz.science/api/sources/{source_id}/photometry"
    token = SKYPORTAL_TOKEN
    headers = {"Authorization": f"token {token}"}

    # Get the photometry for the source
    photometry_r = requests.get(photometry_url, headers=headers)

    if photometry_r.status_code != 200:
        print("Request failed")
        return

    photometry_data = photometry_r.json()
    photometry_list = photometry_data.get("data", [])

    # Define colors for plotting
    colors = {'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'blue'}

    # Time array for plotting
    time_single = time_array

    # Filter names for ZTF filters
    filter_names = ['ztfg', 'ztfr', 'ztfi']

    # Determine the number of examples from mean_preds
    num_examples = len(mean_preds)

    # Loop through all available examples
    for example_idx in range(num_examples):
        # Select one example light curve to plot
        mean_curve_new = mean_preds[example_idx]
        uncertainty_curve_new = uncertainty[example_idx]

        # Create a plot for the predicted light curve and uncertainty
        plt.figure(figsize=(10, 6))

        for i in range(3):  # 3 filters
            # Plot the mean predicted light curve
            plt.plot(time_single, mean_curve_new[:, i], label=f'Predicted {filter_names[i]}', color=colors[filter_names[i]])
            plt.fill_between(time_single, 
                             mean_curve_new[:, i] - 5 * uncertainty_curve_new[:, i], 
                             mean_curve_new[:, i] + 5 * uncertainty_curve_new[:, i], 
                             color=colors[filter_names[i]], alpha=0.2)

        plt.legend()
        
        for obj in photometry_list:
            if obj.get("mag") is not None:
                # Plot the observed magnitude
                plt.errorbar(obj['mjd'] - event_mjd, obj['mag'], yerr=obj['magerr'], fmt='o', label=f'Observed {obj["filter"]}', color=colors[obj["filter"]])
            else:
                # Plot the upper limit
                plt.plot(obj['mjd'] - event_mjd, obj['limiting_mag'], marker='v', color=colors[obj["filter"]])

        # Plot settings
        plt.xlabel('Time (days)')
        plt.ylabel('Magnitude AB')
        plt.gca().invert_yaxis()  # Invert the y-axis for magnitude
        plt.xlim(0, 6)
        plt.savefig(source_id+'.png')

        # Save the plot
        buffer = BytesIO()
        # Uncomment the next line to save the plot to the directory
        #plt.savefig(buffer, format="png", bbox_inches="tight")  # Save to buffer
        buffer.seek(0)
        plt.close()  # Close the figure to free memory

        #post_comment_to_skyportal(time, buffer, superevent_id)
        body = base64.b64encode(buffer.getvalue()).decode("utf-8")
        files = {
            "text": f"{source_id}_{superevent_id}_LC_plot.png",

            "attachment": {
                "body": body,
                "name": f"{source_id}_{superevent_id}_LC_plot.png"
            }
        }
        url = f"https://fritz.science/api/sources/{source_id}/comments"
        headers = {'Authorization': f'token {token}'}
        response = requests.post(url, json=files, headers=headers)
        if response.status_code == 200:
            print(f"Comment posted successfully: {source_id}")
    return

# This function retrieves the bayestar skymap for a GraceDB event and saves it in a directory named skymaps
# Created with the assistance of ChatGPT
def get_skymap_path(graceid, download_dir="skymaps"):
    os.makedirs(download_dir, exist_ok=True)
    client = GraceDb()

    files = client.files(graceid)

    url=f"https://gracedb.ligo.org/api/superevents/{graceid}/files/bayestar.fits.gz"
    local_path = os.path.join(download_dir, f"{graceid}_bayestar.fits.gz")
    if not os.path.exists(local_path):
        print(f"Downloading skymap from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
    return local_path

    raise FileNotFoundError(f"No skymap file found for event {graceid}")

# This function checks if a coordinate in the sky (ra, dec) is within the 90% confidence region of a
# given skymap
# Created with the assistance of ChatGPT
def is_in_90_percent(ra, dec, skymap_path):
    with fits.open(skymap_path) as hdul:
        prob = hdul[1].data['PROB']
        nside = hp.npix2nside(len(prob))

        # Find the probability threshold that encloses 90% of the total probability
        sorted_prob = np.sort(prob)[::-1]
        cumsum = np.cumsum(sorted_prob)
        level_90 = sorted_prob[np.searchsorted(cumsum, 0.9)]

        # Convert RA, Dec to HEALPix pixel
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        theta = 0.5 * np.pi - coord.dec.radian
        phi = coord.ra.radian
        ipix = hp.ang2pix(nside, theta, phi)

        return prob[ipix] >= level_90

# Check if the source id is in the file where we store previously seen source ids
def id_in_file(string_id, filepath):
    with open(filepath, "r") as f:
        for line in f:
            if line.strip() == string_id:
                return True
    return False

# This function checks if a source matches with any events younger than 10 days old in events.json in time
# (3 days before alert time to 7 days after alert time) and space (within the 90% skymap region)
# If a match is found, the source's photometry is plotted onto the event's predicted light curve)
def check_source_with_events(source):
    # Load events
    with open("events.json") as f:
        events = json.load(f)

    now = datetime.now(UTC)

    for event_id, event_info in events.items():
        # Parse event time
        event_time = datetime.fromisoformat(event_info["time"]).replace(tzinfo=timezone.utc)
        age = now - event_time
        if age > timedelta(days=10):
            continue  # Skip old events

        source_time = datetime.fromisoformat(source.get("created_at")).replace(tzinfo=timezone.utc)
        source_id = source.get("id")

        # Check time window
        if not (event_time - timedelta(days=3) <= source_time <= event_time + timedelta(days=7)):
            continue

        # Spatial crossmatch check
        skymap_path = get_skymap_path(event_id)
        print(skymap_path)
        if skymap_path:
            spatial_ok = is_in_90_percent(source.get("ra"), source.get("dec"), skymap_path)

        # If the source matches the event in both time and space, display the event id and plot the source's
        # photometry onto the event's predicted light curve
        if spatial_ok:
            print(event_id)
            plot_source_on_event(source_id, event_id)
            return

    # If no match found:
    print("No match")
    return


# Move the code into this function to run at a specified time
def scheduled_run():
    print("Running task at", datetime.now())

scheduler = BlockingScheduler()

# Use a DST-aware timezone
pacific = pytz.timezone("US/Pacific")

# ~~~ IMPORTANT SCHEDULING CODE ~~~
# Uncomment the following lines to run the scheduled_run function at a specific time
# Multiple triggers can be added as needed, as long as there is a corresponding scheduler.add_job line
#trigger = CronTrigger(hour=11, minute=0, timezone=pacific)  # 11:00 AM PT every day
#scheduler.add_job(scheduled_run, trigger)
#scheduler.start()

# Create a file to store previously seen source ids
base_dir = os.path.dirname(os.path.abspath(__file__))
filepath = f"{base_dir}/source_ids.txt"

# Only create the file if it doesn't exist
if not os.path.exists(filepath):
    with open(filepath, "w") as f:
        pass  # Creates an empty file

# Establish the information needed to connect to the SkyPortal API
base_url = "https://fritz.science/api"
url = base_url + "/sources"
token = SKYPORTAL_TOKEN
headers = {"Authorization": f"token {token}"}
group_ids = [1544]  # If applicable
max_retries = 3

# Introduce the starting pagination parameters
params = {
    'pageNumber': 1,
    'numPerPage': 100,
    'group_ids': group_ids,
    'totalMatches': None,
    'useCache': True, # Enable caching
    'queryID': None # Server will return this in the first response
}

# Create a dict to store all sources by their id
all_sources = {}

# Keep track of whether we've found a source that has already been processed
first_source_found = False

now = datetime.now(UTC)

# Run the code to fetch sources with 3 attempts in case of a request error
retries_remaining = max_retries
while retries_remaining > 0:
    # Query the SkyPortal API for sources
    r = requests.get(
        url,
        params=params,
        headers=headers,
    )

    if r.status_code == 429:
        print("Request rate limit exceeded; waiting 1s before trying again...")
        time.sleep(1)
        continue

    data = r.json()

    # Create a list of sources from the current page of the query
    source_list = data["data"].get("sources", [])

    if not source_list:
        break  # No more sources

    if data["status"] == "success":
        retries_remaining = max_retries
    else:
        print(f"Error: {data["message"]}; waiting 5s before trying again...")  # log as appropriate
        retry_attempts -= 1
        time.sleep(5)
        continue
    
    # For every source in the source list
    for src in source_list:
        # Retrieve the source id and its creation time
        src_id = src.get("id")
        src_time = src.get("created_at")
        # If the source is older than 10 days or has already been processed, stop looking at sources
        if ((now - datetime.fromisoformat(src_time).replace(tzinfo=timezone.utc)) > timedelta(days=10)) or id_in_file(src_id, filepath):
            first_source_found = True # Indicate that we have found the newest source we don't care about
            break
        if src_id:
            # Add the source to the all_sources dict
            all_sources[src_id] = src

    # Figure out how many total sources there are and how many we have fetched so far
    total_matches = data["data"]["totalMatches"]
    params["queryID"] = data["data"]["queryID"] # Pass the queryID to the next request

    # Display how many sources have been fetched so far
    print(f"Fetched {len(all_sources)} of {total_matches} sources.")

    # If we have found a source that is either too old or has already been processed, stop querying sources
    # Stop querying sources if we have fetched all available sources
    if first_source_found or len(all_sources) >= total_matches:
        break

    # Move to the next page of sources
    params['pageNumber'] += 1

# For every source in all_sources, check if it matches with any recent GW events
for src_id in all_sources:
    print(f"Checking source {src_id}")
    # Save the source id to the source_ids.txt file to avoid reprocessing it in the future
    with open(filepath, "a") as f:
        f.write(f"{src_id}\n")
    check_source_with_events(all_sources[src_id])