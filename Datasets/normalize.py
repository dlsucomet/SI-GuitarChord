# Data Normalization and Collation Script
# Myo Guitar Chords Project Gen 2
# Written by Jan Christian Blaise Cruz

from os import listdir
from re import match
from csv import reader, writer
from numpy import mean, std
import numpy as np
import pandas as pd


# Fetches the filename of a certain file based on the pattern.
# Works only in the current directory script is in.
def fetch_name(pattern, folder=''):
    # To fetch anything inside a folder, give a folder variable
    if folder != '':
        folder = folder + '/'

    directory = './' + folder
    file_list = listdir(directory)
    for name in file_list:
        if match(pattern, name):
            return name
    return ''


# Turns the CSV file into a dictionary indexed by their headers by number
# Values is a list under that header, regardless of timestamp
def normalize(filename, folder=''):
    # To fetch anything inside a folder, give a folder variable
    if folder != '':
        folder = folder + '/'

    with open(folder + filename, 'rt') as fin:
        cin = reader(fin)
        raw = [row for row in cin]

    # Remove timestamps and remove header
    raw.remove(raw[0])
    for row in raw:
        del row[0]

    # Unstring the values
    data = [[float(n) for n in row] for row in raw]

    # Flip the matrix. Make a mock dictionary with 0 in the head of the list
    n_data = {n: [0] for n in range(0, len(data[0]))}
    for row in data:
        for i in range(0, len(row)):
            n_data[i].append(row[i])

    # Remove header bit 0
    for row in n_data:
        n_row = n_data[row][1:]
        n_data[row] = n_row

    return n_data


# Calculates a moving average with a base window of 3
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Gets the statisticals for each of the datasets, Mean-Std-Min-Max.
def get_valuation(data):
    m = [np.mean(moving_average(data[row], n=30)) for row in data]
    s = [std(data[row]) for row in data]
    min = [np.min(data[row]) for row in data]
    max = [np.max(data[row]) for row in data]
    res = m + s + min + max
    return res


# For each dataset that matches the category, get the valuation of the datasets, then append
# Returns a matrix where each of the row is the collated valuation of a row.
def collate_dataset(category):
    # Get category matches per folder
    base = listdir('./')
    matches = [folder for folder in base if match(category, folder)]

    # For each folder, get the valuations, append, then place in the return container
    collated = []
    for folder in matches:
        print('Parsing in folder', folder, '...')
        emg = get_valuation(normalize(fetch_name('emg', folder), folder))
        accel = get_valuation(normalize(fetch_name('accel', folder), folder))
        joint = emg + accel
        joint.append(category)
        collated.append(joint)

    return collated


# Merges multiple collated datasets of different ctegories
def merge_collated(lst):
    dataset = []
    for st in lst:
        for row in st:
            dataset.append(row)
    return dataset


# Writes the collated dataset into a CSV file
def write_collated(collated):
    with open('finalized.csv', 'wt') as fout:
        csvout = writer(fout)
        headers = ['MeanEMG1', 'MeanEMG2', 'MeanEMG3', 'MeanEMG4', 'MeanEMG5', 'MeanEMG6', 'MeanEMG7', 'MeanEMG8',
                   'StdEMG1', 'StdEMG2', 'StdEMG3', 'StdEMG4', 'StdEMG5', 'StdEMG6', 'StdEMG7', 'StdEMG8',
                   'MinEMG1', 'MinEMG2', 'MinEMG3', 'MinEMG4', 'MinEMG5', 'MinEMG6', 'MinEMG7', 'MinEMG8',
                   'MaxEMG1', 'MaxEMG2', 'MaxEMG3', 'MaxEMG4', 'MaxEMG5', 'MaxEMG6', 'MaxEMG7', 'MaxEMG8',
                   'MeanXAccel', 'MeanYAccel', 'MeanZAccel', 'StdXAccel', 'StdYAccel', 'StdZAccel',
                   'MinXAccel', 'MinYAccel', 'MinZAccel', 'MaxXAccel', 'MaxYAccel', 'MaxZAccel', 'Category'
                   ]

        # Make an output set
        dataset = []

        # Append the headers and all the datapoints
        dataset.append(headers)
        for row in collated:
            dataset.append(row)

        # Write the file
        csvout.writerows(dataset)


# Main Routine
print('Collating data from sources...')
down_data = collate_dataset('Down')
both_data = collate_dataset('Both')
up_data = collate_dataset('Up')

print('Merging...')
data = merge_collated([down_data, both_data, up_data])

print('Writing...')
write_collated(data)

print('Done!')

