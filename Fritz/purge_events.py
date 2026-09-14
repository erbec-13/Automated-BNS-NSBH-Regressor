import os
import json
from datetime import datetime, timezone

def older_than_ten_days(cur_date):
    input_date = datetime.strptime(cur_date, "%Y-%m-%dT%H:%M:%S")
    input_date = input_date.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    diff = now - input_date

    return diff.days > 150

# Identify the events.json file to store the event data to
file_path = "eventsRem.json"

# Load existing events from events.json if it exists, otherwise create a new dictionary
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        events = json.load(f)
else:
    events = {}

idList = []
            
for event_id in events:
    if not events.get(event_id).get("skymap_url"):
        idList.append(event_id)

for listedId in idList:
    del events[listedId]

# Save the updated events dictionary back to events.json (creating the file if it doesn't exist)
with open(file_path, "w") as f:
    json.dump(events, f, indent=2)