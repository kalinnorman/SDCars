"""
Global Parameter File
If you feel an urge to use a magic number, place it in this file so we can keep track of them all.
"""

import numpy as np

#
# Route Planning
#

# stores decision values
# row indicates current location
# column indicates the region the car should turn toward to reach destination
decision_matrix = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 3, 3, 4],
    [0, 4, 2, 3, 4],
    [0, 1, 2, 3, 1],
    [0, 1, 2, 2, 4]
])

# Stores the gray values for the various regions
region_values = {
    0: 'Out of bounds',
    192: 'Region 1',
    150: 'Region 2',
    128: 'Region 3',
    51: 'Region 4',
    111: 'Intersection',
    250: 'North Stop',
    240: 'South Stop'
}

# Stores names and values of regions
region_dict = {
    'Out of bounds': 0,
    'Region 1': 1,
    'Region 2': 2,
    'Region 3': 3,
    'Region 4': 4,
    'Intersection': 5,
    'North Stop': 6,
    'South Stop': 7
}

region_dict_reverse = {
    0: 'Fail',
    1: 'Region 1',
    2: 'Region 2',
    3: 'Region 3',
    4: 'Region 4',
    5: 'Intersection',
    6: 'North Stop',
    7: 'South Stop'
}