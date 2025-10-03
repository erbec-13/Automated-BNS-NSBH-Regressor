import requests

url = "https://fritz.science/api/gcn_event/2025-07-25T04:09:44"
headers = {"Authorization": "token bb8a0369-068c-4aca-94d7-ee9f97a6a412"}

response = requests.get(url, params={"download": True, "associated_resource_type": "spectra", "resource_id": 14461, "comment_id": 35545, "attachment_name": "Oracle_S250725j_Update_lc.png"}, headers=headers)

print(response.json())

#/35545/Oracle_S250725j_Update_lc.png