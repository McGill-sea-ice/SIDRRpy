"""
Authors: Mathieu Plante, Amelie Bouchat, Damien Ringeisen

--------------------------------------------------------------------------------
Tools for analysing and processing netCDF files
--------------------------------------------------------------------------------

This file contains functions for analysing and processing netCDF files.

"""

# Loading from default packages
import os
import sys
parent = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,parent)
from netCDF4 import Dataset
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# Loads netCDF data
class Data_Distribution:
    """
    This object sets bins for a PDF distribution, according to
    the specifications indicated by the user.

    Data can then be added to the distribution.
    """

    def __init__(self,LeftBC= None, RightBC=None,nbins=None,label = None):
        """
        This Function creates a 1D histogram based on the
        given specifications.

        Input: LeftBC  :: minimum on x-axis of the PDF
               RightBC :: maximum on x-axis of the PDF
               nbins   :: number of bins in the distribution
               label   :: string label describing the data in the distribution
        """

        # Making the bins from the input information
        delta = (RightBC-LeftBC)/nbins
        self.BinEdges = np.arange(0,nbins+1)*delta+LeftBC # bins initialization
        self.bins = (self.BinEdges[1:]+self.BinEdges[:-1])/2.0
        self.BinsWidth = self.BinEdges[1:]-self.BinEdges[:-1]
        self.distribution = self.bins.copy()*0.0

        #Metadata for labelling
        self.label = label



    def add2hist_1D(self, Data = None):
        """
        This function distributes the inptu data into the bins,
        add adds the counts to the previous numbers

        Input: Data :: array which values will be distributed in bins.
        """
        (temp_hist,_) = np.histogram(Data,
                                     self.BinEdges,
                                     density=False)
        self.distribution = self.distribution[:] + temp_hist[:]




class Coverage_map:

    def __init__(self, resolution = None, label = None):
        """
        This class defines a 2D histogram of satellite covevage.
        """

        if resolution is None:
            resolution = 10000 #default
        self.resolution = resolution

        # Upper (u) and lower (l) extents of map_x, map_y (metres)
        lxextent = -4400000
        uxextent =  2600000
        uyextent =  4000000
        lyextent = -2600000

        # Make 2D map bins
        dxi = (float(resolution))
        dyj = (float(resolution))
        self.xbins = np.arange(lxextent,uxextent+dxi,dxi)
        self.ybins = np.arange(lyextent,uyextent+dyj,dyj)
        self.H = np.array([])
        self.ntime = 0
        self.dxi = dxi
        self.dyj = dyj
        self.label = label
        self.covered_area = 0.0
        self.covered_area_tseries = []
        self.timeaxis = []

    def add2hist_2D(self, Data = None, Satellite = None, Time = None):
        """
        This function adds 1 in the histrogram H in location with data.
        """
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        trans = ccrs.Geodetic()

        if Satellite is not None:
            try:  #This is when using the revised SIDRR dataset set, Plante et al., 2024
                idpairs = np.unique(Data.idpair[Data.satellite[:]==Satellite])
                start_lats = Data.start_lat[np.isin(Data.pts_idpair[:],idpairs)]
                start_lons = Data.start_lon[np.isin(Data.pts_idpair[:],idpairs)]
            except: #Otherwise, fall back using the former data format
                start_lats = np.concatenate((Data.start_lat1[Data.satellite[:]==Satellite],
                                         Data.start_lat2[Data.satellite[:]==Satellite],
                                         Data.start_lat3[Data.satellite[:]==Satellite]), axis=0)
                start_lons = np.concatenate((Data.start_lon1[Data.satellite[:]==Satellite],
                                         Data.start_lon2[Data.satellite[:]==Satellite],
                                         Data.start_lon3[Data.satellite[:]==Satellite]), axis=0)
        else:
            try: #This is when using the revised SIDRR dataset set, Plante et al., 2024
                start_lats = Data.start_lat[:]
                start_lons = Data.start_lon[:]
            except: #Otherwise, fall back using the former data format
                start_lats = np.concatenate((Data.start_lat1,Data.start_lat2,Data.start_lat3), axis=0)
                start_lons = np.concatenate((Data.start_lon1, Data.start_lon2, Data.start_lon3), axis=0)

        try:
            new_coords = proj.transform_points(trans, np.array(start_lons), np.array(start_lats))
        except KeyError:
            xi, yj = (0,0)
            sys.exit("Coordinate error when mapping the coverage")

        # Plotting histogram (H) and converting bin values to 0 or 1 range=[[lxextent,uxextent], [lyextent,uyextent]]
        H, _, _ = np.histogram2d(new_coords[:,0], new_coords[:,1], bins=(self.xbins, self.ybins))
        # Changing size of total histogram (only on first run)
        if self.H.shape == (0,):
            self.H.resize(H.shape)

        # Adding interval-specific histogram to total histogram
        H[H > 0.0] = 1.0

        self.covered_area = (np.nansum(H)*(int(self.resolution)**2))*1e-12
        self.H = self.H.copy() + H


        self.covered_area_tseries.append(self.covered_area)
        self.timeaxis.append(Time.t)
        self.ntime = self.ntime + 1




