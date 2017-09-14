#!/usr/bin/env python3
#
# Convert Jeremy's velocity fields to something that can be read by Pele.
#
#

# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import sys
import time
from datetime import timedelta
import numpy as np
import pandas as pd

# ========================================================================
#
# Parse arguments
#
# ========================================================================
parser = argparse.ArgumentParser(
    description='Convert velocity fields for Pele')
parser.add_argument(
    '-p', '--plot', help='Show plots', action='store_true')
args = parser.parse_args()


# ========================================================================
#
# Function definitions
#
# ========================================================================
def get_viscosity(fname):
    with open(fname) as f:
        for line in f:
            if "viscosity" in line:
                return float(line.split()[-1])
    return 0


def get_skiprows_num(fname):
    with open(fname) as f:
        for nskip, line in enumerate(f, 1):
            if "exporting field in real space" in line:
                return nskip
    return 0


# ========================================================================
#
# Main
#
# ========================================================================
# Timer
start = time.time()

# ========================================================================
# Read in UT data
resolution = 128
fname = "initialDNS{0:d}_114.109.txt".format(resolution)
nskip = get_skiprows_num(fname)
dat = pd.read_csv(fname,
                  delim_whitespace=True,
                  header=None,
                  names=['x', 'y', 'z', 'u', 'v', 'w'],
                  skiprows=nskip)

# viscosity
mu = get_viscosity(fname)
Re = 1. / mu

# ========================================================================
# Process the data

# Rename columns to conform to Pele ordering
dat = dat.rename(columns={'x': 'y', 'y': 'x'})

# Shift the x, y, z coordinates to Pele cell centers.
# Right now the data is at [0,dx,2dx,...,(N-1)dx]
dx = np.max(np.diff(dat['x']))

dat['x'] = dat['x'] + 0.5 * dx
dat['y'] = dat['y'] + 0.5 * dx
dat['z'] = dat['z'] + 0.5 * dx

# Sort coordinates to be read easily in Fortran
dat.sort_values(by=['z', 'y', 'x'], inplace=True)

# Calculate urms
urms = np.sqrt(np.mean(dat['u']**2 + dat['v']**2 + dat['w']**2) / 3)

# Calculate Taylor length scale (note Fortran ordering assumption for gradient)
u2 = np.mean(dat['u']**2)
dudx2 = np.mean(
    np.gradient(dat['u'].values.reshape((resolution, resolution, resolution),
                                        order='F'),
                dx,
                axis=0)**2)
lambda0 = np.sqrt(u2 / dudx2)
k0 = 2. / lambda0
Re_lambda = urms * lambda0 / mu

# Normalize the data by urms
dat['u'] /= urms
dat['v'] /= urms
dat['w'] /= urms

# Print some information
print("Simulation information:")
print('\t resolution =', resolution)
print('\t urms =', urms)
print('\t lambda0 =', lambda0)
print('\t k0 = 2/lambda0 =', k0)
print('\t mu =', mu)
print('\t Re = 1/mu =', Re)
print('\t Re_lambda = urms*lambda0/mu = ', Re_lambda)

# ========================================================================
# Write out the data so we can read it in Pele IC function
oname = "hit_ic_ut_{0:d}.dat".format(resolution)
dat.to_csv(oname,
           columns=['x', 'y', 'z', 'u', 'v', 'w'],
           float_format='%.18e',
           index=False)

# output timer
end = time.time() - start
print("Elapsed time " + str(timedelta(seconds=end)) +
      " (or {0:f} seconds)".format(end))
