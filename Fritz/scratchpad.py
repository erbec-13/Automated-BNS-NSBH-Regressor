from matplotlib import pyplot as plt
import numpy as np
import json
import os

# Load existing events from events.json if it exists, otherwise create a new dictionary
file_path = "histogram.json"
# Save the updated events dictionary back to events.json (creating the file if it doesn't exist)
with open(file_path, "r") as f:
    data = json.load(f)

for event_id in data:
    scores = data[event_id]['scores']
    plt.hist(scores, bins='auto', color='skyblue', edgecolor='black', label="n=" + str(len(scores)))
    plt.title(event_id)
    plt.xlabel("reduced chi squared")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
'''
if True:
    for i, saved_thing in enumerate(top_100_matches):
        if score < saved_thing[2]:
            top_100_matches.insert(i, some_match)
            break
    if len(top_100_matches) > 100:
        top_100_matches = top_100_matches[0:100]

#text_load = "Top 100 Matches:\n"
for i, saved_thing in enumerate(top_100_matches):
    text_line = str(i) + ") " + saved_thing[0] + " - " + saved_thing[1] + " - " + str(saved_thing[2]) + "\n"
    text_load += text_line
'''
'''
far_list = []
area_list = []

for key, value in data.items():
    far_list = far_list + [value['far']] #* len(value['scores'])
    area_list = area_list + [value['area']] #* len(value['scores'])

far_list.sort()
area_list.sort()

far_T1 = far_list[len(far_list)//3]
far_T2 = far_list[2*len(far_list)//3]
area_T1 = area_list[len(area_list)//3]
area_T2 = area_list[2*len(area_list)//3]

print(far_T1, far_T2, area_T1, area_T2)

left_flag = False
right_flag = False
for i, item in enumerate(far_list):
    if not left_flag:
        if far_T1 == item:
            left_flag = True
            left_ind = i
        continue
    if not right_flag:
        #print(i)
        if far_T1 != item:
            right_flag = True
            right_ind = i-1
            break
print("FarT1", left_ind, len(far_list)//3, right_ind)

left_flag = False
right_flag = False
for i, item in enumerate(far_list):
    if not left_flag:
        if far_T2 == item:
            left_flag = True
            left_ind = i
        continue
    if not right_flag:
        if far_T2 != item:
            right_flag = True
            right_ind = i-1
            continue
print("FarT2", left_ind, 2*len(far_list)//3, right_ind)

left_flag = False
right_flag = False
for i, item in enumerate(area_list):
    if not left_flag:
        if area_T1 == item:
            left_flag = True
            left_ind = i
        continue
    if not right_flag:
        if area_T1 != item:
            right_flag = True
            right_ind = i-1
            continue
print("AreaT1", left_ind, len(area_list)//3, right_ind)

left_flag = False
right_flag = False
for i, item in enumerate(area_list):
    if not left_flag:
        if area_T2 == item:
            left_flag = True
            left_ind = i
        continue
    if not right_flag:
        if area_T2 != item:
            right_flag = True
            right_ind = i-1
            continue
print("AreaT2", left_ind, 2*len(area_list)//3, right_ind)

bin_00 = []
bin_01 = []
bin_02 = []
bin_10 = []
bin_11 = []
bin_12 = []
bin_20 = []
bin_21 = []
bin_22 = []

for key, value in data.items():
    if value['far'] > 1.19e-8:
        if value['area'] >= 25010:
            bin_00 = bin_00 + value['scores']
        elif 25010 > value['area'] > 1681:
            bin_01 = bin_01 + value['scores']
        else:
            bin_02 = bin_02 + value['scores']
    elif 1.19e-8 >= value['far'] > 4.254e-17:
        if value['area'] >= 25010:
            bin_10 = bin_10 + value['scores']
        elif 25010 > value['area'] > 1681:
            bin_11 = bin_11 + value['scores']
        else:
            bin_12 = bin_12 + value['scores']
    else:
        if value['area'] >= 25010:
            bin_20 = bin_20 + value['scores']
        elif 25010 > value['area'] > 1681:
            bin_21 = bin_21 + value['scores']
        else:
            bin_22 = bin_22 + value['scores']

bin_22.sort()
print(bin_22[len(bin_22)//20])

plt.hist(np.array(bin_11), bins='auto', color='skyblue', edgecolor='black', label="n="+str(len(bin_11)))
plt.title("Medium FAR | Medium Skymap Area")
plt.xlabel("reduced chi squared")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

plt.hist(np.array(bin_01), bins='auto', color='skyblue', edgecolor='black', label="n="+str(len(bin_01)))
plt.title("High FAR | Medium Skymap Area")
plt.xlabel("reduced chi squared")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

plt.hist(np.array(bin_02), bins='auto', color='skyblue', edgecolor='black', label="n="+str(len(bin_02)))
plt.title("High FAR | Low Skymap Area")
plt.xlabel("reduced chi squared")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

plt.hist(np.array(bin_12), bins='auto', color='skyblue', edgecolor='black', label="n="+str(len(bin_12)))
plt.title("Medium FAR | Low Skymap Area")
plt.xlabel("reduced chi squared")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

plt.hist(np.array(bin_22), bins='auto', color='skyblue', edgecolor='black', label="n="+str(len(bin_22)))
plt.title("Low FAR | Low Skymap Area")
plt.xlabel("reduced chi squared")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

plt.hist(np.array(bin_20), bins='auto', color='skyblue', edgecolor='black', label="n="+str(len(bin_20)))
plt.title("Low FAR | High Skymap Area")
plt.xlabel("reduced chi squared")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()
'''