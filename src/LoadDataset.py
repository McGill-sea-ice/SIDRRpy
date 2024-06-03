"""
Authors: Lekima Yakuden, Mathieu Plante, Amelie Bouchat, Damien Ringeisen
GitHub: LekiYak

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
import matplotlib.pyplot as plt

# Code from other files
#from SatelliteCoverage.utils import date_to_seconds, seconds_to_date, convert_to_grid, get_prefix



# Loads netCDF data
class SID_dataset:

    def __init__(self,FileName= None, config=None):
        """
        This function reads and loads data from the daily SIDRR netCDF files.
        This creates an object containing the data fields.

        INPUTS:
        FileName = path to the netcdf file with SIDRR data.
        config   = namelist object, from the python config parser object.

        OBJECT CARACTERISTICS:
        timestep:
        tolerance:
        resolution:
        interval:

        """

        print('--- Loading data (netcdf) ---')
        path = FileName
        print(path)

        # Load netCDF as Dataset from *path*
        ds = Dataset(path, mode='r')


        # Get meta data in object
        # 1. From config
        Date_options = config['Date_options']

        # 2. from netcdf metadata
        self.SatelliteSource  = config['Metadata']['Satellite_Source']
        start_year, start_month, start_day = Date_options['start_year'], Date_options['start_month'], Date_options['start_day']

        reftime = ds.getncattr('referenceTime')
        icetracker = ds.getncattr('SatelliteSource')
        tolerance = ds.getncattr('Max_Deltat')
        trackingerror = ds.getncattr('trackingError')
#       end_year, end_month, end_day = Date_options['end_year'], Date_options['end_month'], Date_options['end_day']

        # Load netCDF as Dataset from *path*
        ds = Dataset(path, mode='r')

        # Extracting acquisition source and time for each triangle
        self.start_time = ds.variables['start_time'][:]
        self.end_time = ds.variables['end_time'][:]
        self.satellite = ds.variables['satellite'][:]

        # Extracting motion data (Filtered by time only)
        self.start_lat1 = (ds.variables['start_lat1'][:])[:]
        self.start_lat2 = (ds.variables['start_lat2'][:])[:]
        self.start_lat3 = (ds.variables['start_lat3'][:])[:]
        self.start_lon1 = (ds.variables['start_lon1'][:])[:]
        self.start_lon2 = (ds.variables['start_lon2'][:])[:]
        self.start_lon3 = (ds.variables['start_lon3'][:])[:]
        self.end_lat1 = (ds.variables['end_lat1'][:])[:]
        self.end_lat2 = (ds.variables['end_lat2'][:])[:]
        self.end_lat3 = (ds.variables['end_lat3'][:])[:]
        self.end_lon1 = (ds.variables['end_lon1'][:])[:]
        self.end_lon2 = (ds.variables['end_lon2'][:])[:]
        self.end_lon3 = (ds.variables['end_lon3'][:])[:]

        #Extracting SAR image and triangle vertex IDs
        self.ids1 = (ds.variables['ids1'][:])[:]
        self.ids2 = (ds.variables['ids2'][:])[:]
        self.ids3 = (ds.variables['ids3'][:])[:]
        self.idpair   = (ds.variables['idpair'][:])[:]

        #Extracting deformation data
        self.A = (ds.variables['A'][:])[:]
        self.div = (ds.variables['div'][:])[:]
        self.shr = (ds.variables['shr'][:])[:]
        self.vrt = (ds.variables['vrt'][:])[:]
        self.dudx = (ds.variables['dudx'][:])[:]
        self.dudy = (ds.variables['dudy'][:])[:]
        self.dvdx = (ds.variables['dvdx'][:])[:]
        self.dvdy = (ds.variables['dvdy'][:])[:]

        #Extracting uncertainty data
        self.errI = (ds.variables['err_div'][:])[:]**0.5
        self.errII = (ds.variables['err_shr'][:])[:]**0.5
        self.errvrt = (ds.variables['err_vrt'][:])[:]**0.5
        self.s2n = (ds.variables['s2n'][:])[:]

        self.errtot = ((self.div*self.errI)**2.0 + (self.shr*self.errII)**2.0 / (self.div**2.0 + self.shr**2.0))**0.5

        #closing the dataset
        ds.close()

        #Create a Mask object
        self.Mask = self.A.copy()
        self.Mask[:] = 1

        #Create an object to track the start date
        self.day_flag = self.idpair.copy()
        self.day_flag[:]= 1

        #Filter out data with too large Area (> 20km^2)
        indices = [i for i in range(len(self.A)) if  self.A[i] < 20000.0**2.0 ]
        self.Mask[indices] = 0
        self.filter_data(indices = indices)


    def filter_data(self, indices = None):
        """
        This function filters out data that not satisfying
        the condition indices == 1.
        """
        print('--- Filtering ---')
        self.start_time = self.start_time[indices]
        self.satellite = self.satellite[indices]
        self.end_time = self.end_time[indices]
        self.start_lat1 = self.start_lat1[indices]
        self.start_lat2 = self.start_lat2[indices]
        self.start_lat3 = self.start_lat3[indices]
        self.start_lon1 = self.start_lon1[indices]
        self.start_lon2 = self.start_lon2[indices]
        self.start_lon3 = self.start_lon3[indices]
        self.end_lat1 = self.end_lat1[indices]
        self.end_lat2 = self.end_lat2[indices]
        self.end_lat3 = self.end_lat3[indices]
        self.end_lon1 = self.end_lon1[indices]
        self.end_lon2 = self.end_lon2[indices]
        self.end_lon3 = self.end_lon3[indices]
        self.A = self.A[indices]
        self.div = self.div[indices]
        self.shr = self.shr[indices]
        self.vrt = self.vrt[indices]
        self.ids1 = self.ids1[indices]
        self.ids2 = self.ids2[indices]
        self.ids3 = self.ids3[indices]
        self.idpair   = self.idpair[indices]
        self.dudx = self.dudx[indices]
        self.dudy = self.dudy[indices]
        self.dvdx = self.dvdx[indices]
        self.dvdy = self.dvdy[indices]
        self.errI = self.errI[indices]
        self.errII = self.errII[indices]
        self.errvrt = self.errvrt[indices]
        self.errtot = self.errtot[indices]
        self.s2n   = self.s2n[indices]
        self.Mask = self.Mask[indices]
        self.day_flag = self.day_flag[indices]

    def mask_data(self, indices = None):
        """
        This function masks values satisfying
        the condition indices == 1.

        Note: This is only for calculated parameters (i.e. after triangulation)
        """

        print('--- Masking ---')
        self.A[indices] = np.nan
        self.div[indices] = np.nan
        self.shr[indices] = np.nan
        self.vrt[indices] = np.nan
        self.dudx[indices] = np.nan
        self.dudy[indices] = np.nan
        self.dvdx[indices] = np.nan
        self.dvdy[indices] = np.nan
        self.Mask[indices] = np.nan

    def Concatenate_data(self, Data2 = None):
        """
        This function concatenate the new data to
        an existing data object.

        Note: this is to produce an object including
              data from different netcdf files
        """

        print('--- Concatenate new data to history ---')

        self.start_time = np.append(self.start_time,Data2.start_time)
        self.end_time = np.append(self.end_time,Data2.end_time)
        self.satellite = np.append(self.satellite, Data2.satellite)

        self.start_lat1 = np.append(self.start_lat1, Data2.start_lat1)
        self.start_lat2 = np.append(self.start_lat2,Data2.start_lat2)
        self.start_lat3 = np.append(self.start_lat3,Data2.start_lat3)
        self.start_lon1 = np.append(self.start_lon1,Data2.start_lon1)
        self.start_lon2 = np.append(self.start_lon2,Data2.start_lon2)
        self.start_lon3 = np.append(self.start_lon3,Data2.start_lon3)
        self.end_lat1 = np.append(self.end_lat1,Data2.end_lat1)
        self.end_lat2 = np.append(self.end_lat2,Data2.end_lat2)
        self.end_lat3 = np.append(self.end_lat3,Data2.end_lat3)
        self.end_lon1 = np.append(self.end_lon1,Data2.end_lon1)
        self.end_lon2 = np.append(self.end_lon2,Data2.end_lon2)
        self.end_lon3 = np.append(self.end_lon3,Data2.end_lon3)

        self.ids1 = np.append(self.ids1,Data2.ids1)
        self.ids2 = np.append(self.ids2,Data2.ids2)
        self.ids3 = np.append(self.ids3,Data2.ids3)
        self.idpair = np.append(self.idpair,Data2.idpair)
        self.day_flag = np.append(self.day_flag,Data2.day_flag)

        self.A = np.append(self.A,Data2.A)
        self.div = np.append(self.div,Data2.div)
        self.shr = np.append(self.shr,Data2.shr)
        self.vrt = np.append(self.vrt,Data2.vrt)
        self.dudx = np.append(self.dudx,Data2.dudx)
        self.dudy = np.append(self.dudy,Data2.dudy)
        self.dvdx = np.append(self.dvdx,Data2.dvdx)
        self.dvdy = np.append(self.dvdy,Data2.dvdy)

        self.errI = np.append(self.errI,Data2.errI)
        self.errII = np.append(self.errII,Data2.errII)
        self.errvrt = np.append(self.errvrt,Data2.errvrt)
        self.errtot = np.append(self.errtot,Data2.errtot)
        self.s2n = np.append(self.s2n,Data2.s2n)

        self.Mask = np.append(self.Mask,Data2.Mask)


    def reconstruct_position_lists(self, min_index= None,max_index=None, EndPoint = None):
        """
        This function reconstruct the list of Lat, Lon position prior to the triangulation
        and outputs corresponding, larger lists of start lats/lons (with 0s in some indices) that reflect the
        coordinates' placement in the original data files for use with ax.tripcolor

        INPUTS:
        start_lat1,2,3 -- Arrays of starting latitudes {np.array, list}
        start_lons1,2,3 -- Arrays of starting longitudes {np.array, list}
        start_id1,2,3 -- Array of starting IDs corresponding to start_lats1,2,3 and start_lons1,2,3 {np.array, list}

        OUTPUTS:
        new_lat -- Array of latitude values at the positions they were orignally in, in the data file
        new_lon -- Array of longitude values at the positions they were originally in, in the data file

        """

        (start_lat1_temp,
        start_lat2_temp,
        start_lat3_temp) = (self.start_lat1[min_index:max_index],
                                 self.start_lat2[min_index:max_index],
                                 self.start_lat3[min_index:max_index])

        (start_lon1_temp,
         start_lon2_temp,
         start_lon3_temp) = (self.start_lon1[min_index:max_index],
                                 self.start_lon2[min_index:max_index],
                                 self.start_lon3[min_index:max_index])

        (end_lat1_temp,
         end_lat2_temp,
         end_lat3_temp) = (self.end_lat1[min_index:max_index],
                                 self.end_lat2[min_index:max_index],
                                 self.end_lat3[min_index:max_index])

        (end_lon1_temp,
         end_lon2_temp,
         end_lon3_temp) = (self.end_lon1[min_index:max_index],
                                 self.end_lon2[min_index:max_index],
                                 self.end_lon3[min_index:max_index])

        (ids1,ids2,ids3) = (self.ids1[min_index:max_index],
                                 self.ids2[min_index:max_index],
                                 self.ids3[min_index:max_index])

        # Combined list of start IDs
        start_ids = np.hstack((ids1, ids2, ids3))

        # Skipping blank data files
        if len(start_ids) == 0:
            return 0, 0

        # Initializing new lists of coordinates
        LatVector, LonVector = ([np.nan] * (max(start_ids) + 1) for i in range(2))

        if EndPoint is True:
            for i in range(len(start_lat1_temp)):
                LatVector[ids1[i]] = end_lat1_temp[i]
                LatVector[ids2[i]] = end_lat2_temp[i]
                LatVector[ids3[i]] = end_lat3_temp[i]

            for i in range(len(start_lon1_temp)):
                LonVector[ids1[i]] = end_lon1_temp[i]
                LonVector[ids2[i]] = end_lon2_temp[i]
                LonVector[ids3[i]] = end_lon3_temp[i]

        else:

            for i in range(len(start_lat1_temp)):
                LatVector[ids1[i]] = start_lat1_temp[i]
                LatVector[ids2[i]] = start_lat2_temp[i]
                LatVector[ids3[i]] = start_lat3_temp[i]

            for i in range(len(start_lon1_temp)):
                LonVector[ids1[i]] = start_lon1_temp[i]
                LonVector[ids2[i]] = start_lon2_temp[i]
                LonVector[ids3[i]] = start_lon3_temp[i]

        return LatVector, LonVector
