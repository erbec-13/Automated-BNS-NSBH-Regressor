# Author: Ethan Erb (referencing and adapting code from Natalya Pletskova)
# Listener 2
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
from ligo.skymap.io import read_sky_map
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from urllib.parse import urlparse

global_counter_5 = 0
global_counter_8 = 0
global_counter_10 = 0
global_counter_20 = 0
global_counter_30 = 0
global_counter_50 = 0
global_counter_75 = 0
global_counter_100 = 0
histogram_dict = dict()
matches_list = []

# Set the SkyPortal token as an environment variable for security
SKYPORTAL_TOKEN = os.getenv("SKYPORTAL_TOKEN")
if SKYPORTAL_TOKEN is None:
    raise ValueError("Please set the SKYPORTAL_TOKEN environment variable")

# This function retrieves the bayestar skymap for a GraceDB event and saves it in a directory named skymaps
# Created with the assistance of ChatGPT
def get_skymap_path(graceid, events, download_dir="skymaps"):
    """
    Download a skymap from a given URL and return the local file path.

    Parameters
    ----------
    skymap_url : str
        Full URL to the skymap file (e.g., from GraceDB or elsewhere).
    graceid : str, optional
        GraceDB event ID, used to name the local file (optional).
    download_dir : str
        Directory to save skymaps (default: 'skymaps').

    Returns
    -------
    str
        Local file path to the downloaded skymap.
    """
                
    # Add or update the event data for the current superevent_id
    skymap_url = events.get(graceid).get("skymap_url")

    if skymap_url == None:
        return None

    os.makedirs(download_dir, exist_ok=True)

    # Extract the filename from the URL
    filename = os.path.basename(urlparse(skymap_url).path)
    if graceid:
        local_filename = f"{graceid}_{filename}"
    else:
        local_filename = filename

    local_path = os.path.join(download_dir, local_filename)

    # Download if not already present
    if not os.path.exists(local_path):
        print(f"Downloading skymap from {skymap_url}")
        response = requests.get(skymap_url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)

    return local_path

# This function checks if a coordinate in the sky (ra, dec) is within the 90% confidence region of a
# given skymap
# Created with the assistance of ChatGPT
def spec_90_percent(ra, dec, prob, header, level_90):

    nside = hp.npix2nside(len(prob))

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    theta = 0.5 * np.pi - coord.dec.radian
    phi = coord.ra.radian
    ipix = hp.ang2pix(nside, theta, phi)

    return prob[ipix] >= level_90

# This function checks if a source matches with any events younger than 10 days old in events.json in time
# (3 days before alert time to 7 days after alert time) and space (within the 90% skymap region)
# If a match is found, the source's photometry is plotted onto the event's predicted light curve)
def check_source_with_events(source, events, skymap_dict, level_dict):
    source_id = source.get("id")

    now = datetime.now(UTC)

    for event_id, event_info in events.items():
        # Parse event time
        event_time = datetime.fromisoformat(event_info["time"]).replace(tzinfo=timezone.utc)
        age = now - event_time
        if age > timedelta(days=10):
            continue  # Skip old events

        source_time = datetime.fromisoformat(source.get("created_at")).replace(tzinfo=timezone.utc)

        # Check time window
        if not (event_time - timedelta(days=3) <= source_time <= event_time + timedelta(days=7)):
            continue
        
        spatial_ok = spec_90_percent(source.get("ra"), source.get("dec"), skymap_dict.get(event_id)[0], skymap_dict.get(event_id)[1], level_dict.get(event_id))

        # If the source matches the event in both time and space, display the event id and plot the source's
        # photometry onto the event's predicted light curve
        if spatial_ok:
            plot_source_on_event(source_id, event_id)
            check_if_source_photometry_matches_with_event_prediction(source_id, event_id)

    return

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
                             mean_curve_new[:, i] - 3 * uncertainty_curve_new[:, i], 
                             mean_curve_new[:, i] + 3 * uncertainty_curve_new[:, i], 
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
        url = f"https://fritz.science/api/candidates/{source_id}/comments"
        headers = {'Authorization': f'token {token}'}
        response = requests.post(url, json=files, headers=headers)
        if response.status_code == 200:
            print(f"Comment posted successfully: {source_id}")
    return

def linear_interpolate_event(x, displacement, time_array, mean_preds, uncertainty, filter):
    time_actual = x - displacement
    filter_index = 0
    index_l = 0
    index_r = len(time_array) - 1
    for i in range(len(time_array)):
        if time_actual <= time_array[i]:
            index_r = i
            index_l = i - 1
            break
    time_l = time_array[index_l]
    means_l = mean_preds[index_l]
    uncertainties_l = uncertainty[index_l]
    time_r = time_array[index_r]
    means_r = mean_preds[index_r]
    uncertainties_r = uncertainty[index_r]
    if filter == "ztfg":
        filter_index = 0
    elif filter == "ztfr":
        filter_index = 1
    elif filter == "ztfi":
        filter_index = 2
    mag_l = means_l[filter_index]
    magerr_l = uncertainties_l[filter_index]
    mag_r = means_r[filter_index]
    magerr_r = uncertainties_r[filter_index]
    mag = ((mag_r - mag_l) / (time_r - time_l)) * (time_actual - time_l) + mag_l
    magerr = ((magerr_r - magerr_l) / (time_r - time_l)) * (time_actual - time_l) + magerr_l
    flux = 10**((23.9 - mag) / 2.5)          # µJy
    fluxerr = 0.4 * np.log(10) * flux * magerr
    return (mag, magerr)

def reduced_chi_squared(A_dict, B_dict):
    counter = 0
    total = 0
    for key in A_dict:
        if A_dict.get(key) is not None and B_dict.get(key) is not None:
            counter += 1
            A = A_dict.get(key)[0]
            sigma_A = A_dict.get(key)[1]
            B = B_dict.get(key)[0]
            sigma_B = B_dict.get(key)[1]
            chi_squared = ((A - B) * (A - B)) / (sigma_A * sigma_A + sigma_B * sigma_B)
            total += chi_squared
    if counter == 0:
        #print("Error calculating reduced chi squared: Defaulted to 1000")
        #print(A_dict)
        #print(B_dict)
        return 10000
    score = total / counter
    return score

def save_source_to_group(source_id):
    url = "https://fritz.science/api/source_groups"
    token = SKYPORTAL_TOKEN

    payload = {
        "objId": source_id,
        "inviteGroupIds": [1862]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        print("Save to group failed")

def check_if_source_photometry_matches_with_event_prediction(source_id, event_id):

    # 1. Get event data
    # 2. Get source data
    # 3. For each source data point, retrieve A_i and sigma(A_i)
    # 4. For each source data point, run linear interpolation on event to get B_i and sigma(B_i)
    # 5. Compute true standard deviations for sigma(A_i) and sigma(B_i)
    # 6. Compile all the data into two dicts of tuples,
    #    where the ith time is a key with value (A_i, sigma(A_i)) in one and (B_i, sigma(B_i)) in two
    # 7. Pass the lists to a reduced chi squared function and return the result
    # 8. Compare the result to see if it's less than a desired maximum
    # 9. If it is, send the Slack message

    #################### 1. Get event data ####################

    # Retrieve the event information from events.json
    with open("events.json") as f:
        event = json.load(f).get(event_id)
    time_array = np.array(event.get("time_single"))
    mean_preds = np.array(event.get("mean_preds_inverted"))
    uncertainty = np.array(event.get("uncertainty_reshaped"))
    superevent_id = event_id
    time = event.get("time")
    far = event.get("far")
    area_90 = event.get("area_90")

    # Convert time into MJD
    event_t = Time(time, format='isot', scale='utc')
    event_mjd = event_t.mjd
    #print("Event MJD: " + str(event_mjd))

    #################### 2. Get source data ####################

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

    #################### 6. Compile all the data into two dicts of tuples ####################

    A_data = dict()
    B_data = dict()

    #################### 3. For each source data point, retrieve A_i and sigma(A_i) ####################

    for obj in photometry_list:
        #if obj.get("origin") != "alert_fp":
        #    continue
        if obj.get("mag") is not None and obj.get('mjd') >= event_mjd + 0.1 and obj.get('mjd') <= event_mjd + 6:
            # Data Point: ob'] = x_i, obj['mag'] = A_i, obj['magerr'] = sigma(A_i)
            flux = 10**((23.9 - obj['mag']) / 2.5)          # µJy
            fluxerr = 0.4 * np.log(10) * flux * obj['magerr']
            A_data[obj['mjd']] = (obj['mag'], obj['magerr'])
            ############ 4. For each source data point, run linear interpolation on event to get B_i and sigma(B_i) ############
            B_data[obj['mjd']] = linear_interpolate_event(obj['mjd'], event_mjd, time_array, mean_preds[0], uncertainty[0], obj['filter'])
        else:
            # Upper Limit: Check at x = obj['mjd'] if A = obj['limiting_mag'] is greater than or equal to B = eventY
            continue
        
    #################### 7. Pass the lists to a reduced chi squared function and return the result ####################
    passing_score = 10
    red_chi_squared = reduced_chi_squared(A_data, B_data)
    if far > 1.19e-8:
        if area_90 >= 25010:
            passing_score = 9999
        elif 25010 > area_90 > 1681:
            passing_score = 13.46
        else:
            passing_score = 19.78
    elif 1.19e-8 >= far > 4.254e-17:
        if area_90 >= 25010:
            passing_score = 9999
        elif 25010 > area_90 > 1681:
            passing_score = 27.62
        else:
            passing_score = 1.26
    else:
        if area_90 >= 25010:
            passing_score = 8.67
        elif 25010 > area_90 > 1681:
            passing_score = 9999
        else:
            passing_score = 108.6
    #global global_counter_5
    #global global_counter_8
    #global global_counter_10
    #global global_counter_20
    #global global_counter_30
    #global global_counter_50
    #global global_counter_75
    #global global_counter_100
    global matches_list
    #if red_chi_squared <= 5:
    #    global_counter_5 += 1
    #if red_chi_squared <= 8:
    #    global_counter_8 += 1
    #if red_chi_squared <= 10:
    #    global_counter_10 += 1
    #if red_chi_squared <= 20:
    #    global_counter_20 += 1
    #if red_chi_squared <= 30:
    #    global_counter_30 += 1
    #if red_chi_squared <= 50:
    #    global_counter_50 += 1
    #if red_chi_squared <= 75:
    #    global_counter_75 += 1
    #if red_chi_squared <= 100:
    #    global_counter_100 += 1
    if red_chi_squared <= passing_score:
        save_source_to_group(source_id)
        matches_list.append([source_id, event_id, red_chi_squared, True])
    else:
        matches_list.append([source_id, event_id, red_chi_squared, False])
    #send_slack(source_id, event_id, user_id="U044QV5LVFE")
    #global histogram_dict
    #if histogram_dict.get(event_id):
    #    histogram_dict[event_id]['scores'].append(red_chi_squared)
    #else:
    #    dict_to_insert = dict()
    #    dict_to_insert['scores'] = [red_chi_squared]
    #    dict_to_insert['area'] = 0
    #    dict_to_insert['hasNS'] = 0
    #    dict_to_insert['far'] = 0
    #    histogram_dict[event_id] = dict_to_insert
    return

def send_slack(text_load, user_id="U0AKLPPK8RM"):
    token = "REDACTED_SLACK_TOKEN"
    #user_id = "U044QV5LVFE"
    url = "https://slack.com/api/chat.postMessage"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "channel": user_id,   # user ID works for DMs
        "text": text_load
    }

    response = requests.post(url, headers=headers, json=payload)

    data = response.json()

    if not data.get("ok"):
        raise Exception(f"Slack API error: {data}")

    return data

# Move the code into this function to run at a specified time
def scheduled_run():
    global matches_list
    matches_list = []
    print("Running task at", datetime.now())
    now = datetime.now(UTC)
    ten_days_ago = now - timedelta(days=10)
    # Create a file to store previously seen source ids
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Establish the information needed to connect to the SkyPortal API
    base_url = "https://fritz.science/api"
    url = base_url + "/candidates"
    token = SKYPORTAL_TOKEN
    headers = {"Authorization": f"token {token}"}
    group_ids = [1544]  # If applicable
    max_retries = 3

    # Introduce the starting pagination parameters
    params = {
        'pageNumber': 1,
        'numPerPage': 100,
        'totalMatches': None,
        'startDate': str(ten_days_ago.isoformat())
    }

    # Create a dict to store all sources by their id
    all_sources = {}

    # Keep track of whether we've found a source that has already been processed
    first_source_found = False

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
        source_list = data["data"].get("candidates", [])

        if not source_list:
            # Fritz sometimes returns an empty list on the first call with a new queryID
            if "queryID" in data["data"]:
                print("1a")
                params["queryID"] = data["data"]["queryID"]
                params["useCache"] = True
                print("Got queryID, retrying to fetch first page...")
                continue  # Retry now that query is cached
            else:
                print("1b")
                #print(json.dumps(data["data"], indent=2))
                break  # Truly no more sources

        if data["status"] == "success":
            retries_remaining = max_retries
        else:
            print(f"Error: {data["message"]}; waiting 5s before trying again...")  # log as appropriate
            retries_remaining -= 1
            time.sleep(5)
            continue
        
        # For every source in the source list
        for src in source_list:
            # Retrieve the source id and its creation time
            src_id = src.get("id")
            src_time = src.get("created_at")
            print(src_id)
            if src_id:
                # Add the source to the all_sources dict
                all_sources[src_id] = src

        # Figure out how many total sources there are and how many we have fetched so far
        total_matches = data["data"]["totalMatches"]

        # Display how many sources have been fetched so far
        print(f"Fetched {len(all_sources)} of {total_matches} sources.")

        # If we have found a source that is either too old or has already been processed, stop querying sources
        # Stop querying sources if we have fetched all available sources
        if len(all_sources) >= total_matches:
            break

        # Move to the next page of sources
        params['pageNumber'] += 1

    counter = 0
    with open("events.json") as f:
        new_dict = json.load(f)
    skymap_dict = dict()
    for ident, event in new_dict.items():
        skymap_dict[ident] = read_sky_map(get_skymap_path(ident, new_dict))
    level_dict = dict()
    for ident, skymap in skymap_dict.items():
        sorted_prob = np.sort(skymap_dict.get(ident)[0])[::-1]
        cumsum = np.cumsum(sorted_prob)
        level_dict[ident] = sorted_prob[np.searchsorted(cumsum, 0.9)]

    # For every source in all_sources, check if it matches with any recent GW events
    for src_id in all_sources:
        check_source_with_events(all_sources[src_id], new_dict, skymap_dict, level_dict)

    # PREPARING THE SLACK SUMMARY
    group_by_event_dict = dict()
    for some_match in matches_list:
        source_id = some_match[0]
        event_id = some_match[1]
        score = some_match[2]
        top_5_percent = some_match[3]

        if group_by_event_dict.get(event_id):
            for i, saved_thing in enumerate(group_by_event_dict[event_id]):
                if score < saved_thing[1]:
                    group_by_event_dict[event_id].insert(i, (source_id, score, top_5_percent))
                    break
        else:
            group_by_event_dict[event_id] = [(source_id, score)]

    text_load = "Grouping All Matches By Event:\n"
    for event_id in group_by_event_dict:
        text_line = "\n" + event_id + "\n"
        text_load += text_line
        for saved_thing in group_by_event_dict[event_id]:
            significant_text = ""
            if saved_thing[2]:
                significant_text = " - Significant"
            text_line = saved_thing[0] + " - " + str(saved_thing[1]) + significant_text + "\n"
            text_load += text_line

    mailing_list = ["U0AKLPPK8RM", "U044QV5LVFE"]
    for user_id in mailing_list:
        send_slack(text_load, user_id=user_id)

scheduler = BlockingScheduler()

# Use a DST-aware timezone
pacific = pytz.timezone("US/Pacific")

# ~~~ IMPORTANT SCHEDULING CODE ~~~
# Uncomment the following lines to run the scheduled_run function at a specific time
# Multiple triggers can be added as needed, as long as there is a corresponding scheduler.add_job line
trigger = CronTrigger(hour=11, minute=0, timezone=pacific)  # 11:00 AM PT every day
scheduler.add_job(scheduled_run, trigger)
scheduler.start()
#########################################
# Comment this line to run the scheduler
#scheduled_run()
# ~~~ IMPORTANT SCHEDULING CODE END ~~~

# ~~~ ANY OTHER FUNCTIONS USED FOR TESTING STORED BELOW THIS LINE ~~~

def sample_testing():
    print("Running task at", datetime.now())
    # Create a file to store previously seen source ids
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Establish the information needed to connect to the SkyPortal API
    base_url = "https://fritz.science/api"
    url = base_url + "/candidates"
    token = SKYPORTAL_TOKEN
    headers = {"Authorization": f"token {token}"}
    group_ids = [1544]  # If applicable
    max_retries = 3

    # Introduce the starting pagination parameters
    params = {
        'pageNumber': 1,
        'numPerPage': 100,
        'totalMatches': None,
        'startDate': "2025-06-28T11:00:00",
        'endDate': "2025-11-27T11:00:00"
    }

    # Create a dict to store all sources by their id
    all_sources = {}

    # Keep track of whether we've found a source that has already been processed
    first_source_found = False

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

        if r.status_code != 200:
            print("Request failed")
            print("Response text:", r.text)
            retries_remaining -= 1
            time.sleep(2)
            continue

        try:
            data = r.json()
        except ValueError:
            print("Response was not JSON:")
            print(r.text)
            retries_remaining -= 1
            time.sleep(2)
            continue

        # Create a list of sources from the current page of the query
        source_list = data["data"].get("candidates", [])

        if not source_list:
            # Fritz sometimes returns an empty list on the first call with a new queryID
            if "queryID" in data["data"]:
                print("1a")
                params["queryID"] = data["data"]["queryID"]
                params["useCache"] = True
                print("Got queryID, retrying to fetch first page...")
                continue  # Retry now that query is cached
            else:
                print("1b")
                #print(json.dumps(data["data"], indent=2))
                break  # Truly no more sources

        if data["status"] == "success":
            retries_remaining = max_retries
        else:
            print(f"Error: {data["message"]}; waiting 5s before trying again...")  # log as appropriate
            retries_remaining -= 1
            time.sleep(5)
            continue
        
        # For every source in the source list
        for src in source_list:
            # Retrieve the source id and its creation time
            src_id = src.get("id")
            src_time = src.get("created_at")
            #print(src_id)
            if src_id:
                # Add the source to the all_sources dict
                all_sources[src_id] = src

        # Figure out how many total sources there are and how many we have fetched so far
        total_matches = data["data"]["totalMatches"]

        # Display how many sources have been fetched so far
        print(f"Fetched {len(all_sources)} of {total_matches} sources.")

        # If we have found a source that is either too old or has already been processed, stop querying sources
        # Stop querying sources if we have fetched all available sources
        if len(all_sources) >= total_matches:
            break

        # Move to the next page of sources
        params['pageNumber'] += 1

    counter = 0
    with open("events.json") as f:
        events = json.load(f)
    with open("histogram.json") as f:
        histo = json.load(f)
    skymap_dict = dict()
    for ident in histo:
        skymap_dict[ident] = read_sky_map(get_skymap_path(ident, events))
    level_dict = dict()
    for ident, skymap in skymap_dict.items():
        sorted_prob = np.sort(skymap_dict.get(ident)[0])[::-1]
        cumsum = np.cumsum(sorted_prob)
        level_dict[ident] = sorted_prob[np.searchsorted(cumsum, 0.9)]
    # For every source in all_sources, check if it matches with any recent GW events
    for num, src_id in enumerate(all_sources):
        print(num, "/", len(all_sources))
        counter += 1
        checkSampleSources(all_sources[src_id], events, skymap_dict, level_dict)

    global matches_list
    top_100_matches = []
    for some_match in matches_list:
        if len(top_100_matches) < 100:
            top_100_matches.append(some_match)
        else:
            for i, saved_thing in enumerate(top_100_matches):
                if some_match[2] < saved_thing[2]:
                    top_100_matches.insert(i, some_match)
                    break
            if len(top_100_matches) > 100:
                top_100_matches = top_100_matches[0:100]

    text_load = "Top 100 Matches:\n"
    for i, saved_thing in enumerate(top_100_matches):
        text_line = str(i) + ") " + saved_thing[0] + " - " + saved_thing[1] + " - " + str(saved_thing[2]) + "\n"
        text_load += text_line
    send_slack(text_load)

def checkSampleSources(source, events, skymap_dict, level_dict):
    now = datetime.now(UTC)

    source_id = source.get("id")

    for event_id in skymap_dict:

        event_info = events[event_id]

        # Parse event time
        event_time = datetime.fromisoformat(event_info["time"]).replace(tzinfo=timezone.utc)

        source_time = datetime.fromisoformat(source.get("created_at")).replace(tzinfo=timezone.utc)

        # Check time window
        if not (event_time - timedelta(days=3) <= source_time <= event_time + timedelta(days=7)):
            continue

        # Spatial crossmatch check
        spatial_ok = spec_90_percent(source.get("ra"), source.get("dec"), skymap_dict.get(event_id)[0], skymap_dict.get(event_id)[1], level_dict.get(event_id))

        # If the source matches the event in both time and space, display the event id and plot the source's
        # photometry onto the event's predicted light curve
        if spatial_ok:
            check_if_source_photometry_matches_with_event_prediction(source_id, event_id)

    # If no match found:
    return

# ~~~ TESTING FUNCTION CALLS STORED HERE ~~~

#send_slack("X", "Y", user_id="U044QV5LVFE")
#send_slack("X", user_id="U0AKLPPK8RM")

#sample_testing()
#print("f(5) = " + str(global_counter_5))
#print("f(8) = " + str(global_counter_8))
#print("f(10) = " + str(global_counter_10))
#print("f(20) = " + str(global_counter_20))
#print("f(30) = " + str(global_counter_30))
#print("f(50) = " + str(global_counter_50))
#print("f(75) = " + str(global_counter_75))
#print("f(100) = " + str(global_counter_100))

#plot_source_on_event("ZTF25abrwedh", "S250908y")
#ZTF25abthqiv S250917aq
#ZTF25abrwedh S250908y 1.902475822409297e-08
#ZTF25abrxakd S250908y 0.01050113435335753
#ZTF25abquxgk S250908y 0.6025329936484785

#save_source_to_group("ZTF25abquxgk")

# Load existing events from events.json if it exists, otherwise create a new dictionary
#file_path = "histogram.json"
# Save the updated events dictionary back to events.json (creating the file if it doesn't exist)
#with open(file_path, "w") as f:
#    json.dump(histogram_dict, f, indent=2)