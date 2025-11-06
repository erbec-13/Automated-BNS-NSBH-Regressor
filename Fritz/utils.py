import numpy as np
from astropy.table import Table
from astropy.io import fits
import requests
from io import BytesIO
import base64
import astropy_healpix as ah
import healpy as hp
import xmltodict
from ligo.gracedb.rest import GraceDb
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import find_greedy_credible_levels
import matplotlib.pyplot as plt
from astropy.time import Time
import json
import os

SKYPORTAL_HOST = os.getenv("SKYPORTAL_HOST", "https://fritz.science")
SKYPORTAL_TOKEN = os.getenv("SKYPORTAL_TOKEN")
if SKYPORTAL_TOKEN is None:
    raise ValueError("Please set the SKYPORTAL_TOKEN environment variable")
HEADERS = {"Authorization": f"token {SKYPORTAL_TOKEN}"}


def parse_gcn(response):
    return xmltodict.parse(response)


def get_params(event_dict):
    params = {item['@name']: item['@value'] for item in event_dict['voe:VOEvent']['What']['Param']}
    
    superevent_id = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'GraceID'][0]
    event_page = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'EventPage'][0]
    alert_type = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'AlertType'][0]
    group = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'Group'][0]
    
    significant = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'Significant'][0]
    
    classification = [item for item in event_dict['voe:VOEvent']['What']['Group'] if item.get('@name') == 'Classification']
    properties = [item for item in event_dict['voe:VOEvent']['What']['Group'] if item.get('@name') == 'Properties']
    try:
        prob_bbh = float([item['@value'] for item in classification[0]['Param'] if item.get('@name') == 'BBH'][0])  
    except:
        prob_bbh = 0
    try:
        prob_ter = float([item['@value'] for item in classification[0]['Param'] if item.get('@name') == 'Terrestrial'][0])
    except:
        prob_ter = 1
    try:
        prob_bns = float([item['@value'] for item in classification[0]['Param'] if item.get('@name') == 'BNS'][0])  
    except:
        prob_bns = 0
    try:
        prob_nsbh = float([item['@value'] for item in classification[0]['Param'] if item.get('@name') == 'NSBH'][0])
    except:
        prob_nsbh = 1
    try:
        has_ns = float([item['@value'] for item in properties[0]['Param'] if item.get('@name') == 'HasNS'][0])
    except:
        has_ns = 0
    try:
        has_remnant = float([item['@value'] for item in properties[0]['Param'] if item.get('@name') == 'HasRemnant'][0])
    except:
        has_remnant = 0
    try:
        has_mass_gap = float([item['@value'] for item in properties[0]['Param'] if item.get('@name') == 'HasMassGap'][0])
    except:
        has_mass_gap = 0
    try:
        PAstro = 1 - prob_ter
    except:
        PAstro = 1
    far = float(params.get('FAR', 0))
    skymap_url = next(item['Param']['@value'] for item in event_dict['voe:VOEvent']['What']['Group'] if item.get('@name') == 'GW_SKYMAP')
    print(skymap_url)
    skymap_response = requests.get(skymap_url)
    skymap_bytes = skymap_response.content
    skymap = Table.read(BytesIO(skymap_bytes))
    if skymap is None:
        print("Skipping bad or unsupported skymap.")
        return
    if isinstance(skymap, Table):
        # Bayestar-like
        level, ipix = ah.uniq_to_level_ipix(skymap[np.argmax(skymap['PROBDENSITY'])]['UNIQ'])
        ra, dec = ah.healpix_to_lonlat(ipix, ah.level_to_nside(level), order='nested')
        m, meta = read_sky_map(BytesIO(skymap_bytes))
        distmean = skymap.meta.get('DISTMEAN', 'error')
        nside = ah.level_to_nside(level)
    else:
        # CWB or fallback
        try:
            m, meta = read_sky_map(BytesIO(skymap_bytes))  # read_sky_map can handle HDUList too
            distmean = meta.get('DISTMEAN', 'error')
            nside = hp.npix2nside(len(m))
            ipix = np.argmax(m)
            ra, dec = hp.pix2ang(nside, ipix, nest=True, lonlat=True)
        except Exception as e:
            print(f"Could not parse CWB skymap structure: {e}")
            return
    #skymap = Table.read(BytesIO(skymap_bytes))  # or 'votable'
    #level, ipix = ah.uniq_to_level_ipix(skymap[np.argmax(skymap['PROBDENSITY'])]['UNIQ'])
    #ra, dec = ah.healpix_to_lonlat(ipix, ah.level_to_nside(level), order='nested')
    #c = SkyCoord(ra, dec, frame='icrs', unit='deg')
    #m, meta = read_sky_map(BytesIO(skymap_bytes))
    credible_levels = find_greedy_credible_levels(m)
    pixel_area_deg2 = np.sum(credible_levels <= 0.9) * hp.nside2pixarea(nside, degrees=True)
    longitude = ra.value
    latitude = dec.value
    #distmean = skymap.meta.get('DISTMEAN', 'error')
    area_90 = pixel_area_deg2
    
    far_format = 1. / (far * 3.15576e7)
    t0 = event_dict['voe:VOEvent']['WhereWhen']['ObsDataLocation']['ObservationLocation']['AstroCoords']['Time']['TimeInstant']['ISOTime']
    dateobs = Time(t0, precision=0).datetime
    time = dateobs.strftime('%Y-%m-%dT%H:%M:%S')
    
    chirp_mass = get_bin_edges_for_event(superevent_id)
    diststd = skymap.meta.get('DISTSTD', 'error')
    
    return superevent_id, event_page, alert_type, group, prob_bbh, prob_bns, prob_nsbh, far_format, distmean, area_90, longitude, latitude, has_ns, has_remnant, has_mass_gap, significant, prob_ter, skymap, PAstro, time, diststd, chirp_mass, skymap_url

# Extract ChirpMass if present
def get_bin_edges_for_event(superevent_id, filename="mchirp_source_PE.json", service_url='https://gracedb.ligo.org/api/'):
    client = GraceDb(service_url=service_url)
    # Download file
    try:
        r = client.files(superevent_id, filename)
    except:
        r = client.files(superevent_id, "mchirp_source.json")
    #for key in r.json():
        #print(key, r.json()[key])
    if r.status_code != 200:
        raise FileNotFoundError(f"{filename} not found for superevent {superevent_id} (HTTP {r.status_code})")
    raw = r.read()
    data = json.loads(raw)
    # Extract bin_edges
    if "bin_edges" in data:
        return data["bin_edges"]
    else:
        raise KeyError(f"'bin_edges' not found in {filename} for {superevent_id}")

def predict_with_uncertainty(model, X, n_iter=1000):
    predictions = []
    for _ in range(n_iter):
        predictions.append(model(X, training=True))

    predictions = np.array(predictions)
    mean_preds = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)

    return mean_preds, uncertainty


def fetch_event_id(time):
    token = SKYPORTAL_TOKEN
    headers = {'Authorization': f'token {token}'}
    response = requests.get(f'https://fritz.science/api/gcn_event/{time}', headers=headers)
    if response.status_code != 200:
        print(f"API returned {response.status_code}: {response.text}")
        return None  # Fail gracefully
    #print(response.status_code)
    #print(response.text)

    return response.json()['data']['id']

def try_load_skymap(skymap_bytes):
    buffer = BytesIO(skymap_bytes)

    # Try loading as table
    try:
        return Table.read(buffer, format='fits')  # If it is a FITS table
    except Exception as e1:
        buffer.seek(0)  # Reset pointer for next attempt

        # Try loading with astropy.io.fits to inspect content
        try:
            with fits.open(buffer) as hdul:
                # Print or inspect to guide logic
                print("FITS HDU types:", [type(hdu) for hdu in hdul])
                print("FITS content summary:", hdul.info())
                # Optionally return the raw HDUList for custom handling
                return hdul
        except Exception as e2:
            buffer.seek(0)  # Reset again

            # Try as skymap
            try:
                prob, header = read_sky_map(buffer, moc=True)
                return {'prob': prob, 'header': header}
            except Exception as e3:
                print(f"Skymap file could not be read: {e3}")
                return None


def post_comment_to_skyportal(time, buffer, superevent_id):
    event_id = fetch_event_id(time)
    if event_id is None:
        print(f"Could not fetch event ID for {superevent_id}. Skipping comment post.")
        return
    body = base64.b64encode(buffer.getvalue()).decode("utf-8")
    files = {
        "text": f"GraceDB file: Oracle_{superevent_id}_LC_plot.png",

        "attachment": {
            "body": body,
            "name": f"Oracle_{event_id}_LC_plot.png"
        }
    }
    url = f"https://fritz.science/api/gcn_event/{event_id}/comments"
    token = SKYPORTAL_TOKEN
    headers = {'Authorization': f'token {token}'}
    response = requests.post(url, json=files, headers=headers)
    if response.status_code == 200:
        print(f"Comment posted successfully: {superevent_id}")


def plot_all_light_curves_with_uncertainty(time_array, mean_preds, uncertainty, superevent_id, time):
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

        # Plot settings
        plt.xlabel('Time (days)')
        plt.ylabel('Magnitude AB')
        plt.gca().invert_yaxis()  # Invert the y-axis for magnitude
        plt.legend()
        plt.xlim(0, 6)
        plt.savefig(superevent_id+'.png')

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")  # Save to buffer
        buffer.seek(0)
        #plt.show()
        plt.close()  # Close the figure to free memory

        post_comment_to_skyportal(time, buffer, superevent_id)
