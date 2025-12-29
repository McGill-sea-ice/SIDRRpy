"""
Author: Mathieu Plante, Lekima Yakuden
GitHub: mathieuslplante, LekiYak

--------------------------------------------------------------------------------
Code to investigate SIDRR dataset format and caracteristics
--------------------------------------------------------------------------------

This code is used to load the SIDRR data from the daily netcdf files,
plot the included data and determine the spatio-temporal coverage.

SIDRR data computed from SAR image pairs are stored in daily netcdf files,
according to the acquistion time of the earliest image from the pair.
SIDRR data spanning a given date are found accross several netcdf files.

To load all data valid for a specific date, it is necessary to:

1. Span cdf files from days prior to the analysis initial time,
2. Keep track of the previously loaded data until past the
   acquisition time of the latest image from the pair.

"""

# Loading from default packages
import os
import sys
parent = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,parent)
sys.path.append(r'src/')
from time import strftime
import configparser
from datetime import datetime, date, time, timedelta
import time
import numpy as np
#from netCDF4 import Dataset

# Code from src files
from visualisation import visualisation
from LoadDataset   import SID_dataset
from TimeUtil import TimeUtil
from Statistics_objects import Coverage_map, Data_Distribution
from haversine import haversine
from PostProcessed_Data import PostProcessed_Data

if __name__ == '__main__':


    #------------------------------------------------------------------
    # Reading the  namelist
    #------------------------------------------------------------------

    # Reading config
    config = configparser.ConfigParser()
    config.read('namelist.ini')

    path = config['IO']['netcdf_path']
    TimeTool = TimeUtil(config = config['Date_options'])
    Period = config['Date_options']['Period']

    # Start the clock
    start_time = time.time()
    Delta_days_init = 0

    refTime = TimeTool.ThisTime

    #------------------------------------------------------------------
    # STARTING ANALYSIS
    #------------------------------------------------------------------

    # Iterating over each day
    for ThisTime in TimeTool.daterange():

        #------------------------------------------------------------------
        # Fetch data
        #------------------------------------------------------------------

        # Update time and data paths
        TimeTool.ThisTime = ThisTime
        TimeTool.t = TimeTool.time_ref_number(date_pt = ThisTime)
        TimeTool.NextTime = TimeTool.ThisTime + timedelta(seconds=TimeTool.tstep*60*60)

        ThisTime_str = TimeTool.ThisTime.strftime("%Y%m%d")
        NextTime_str = TimeTool.NextTime.strftime("%Y%m%d")
        TimeTool.ThisTimeFile = "SIDRR_%s.nc" % ThisTime_str
        filePath = path + TimeTool.ThisTimeFile

        #Load visualisation tool
        visuals = visualisation(config=config)

        #Load new Data from SIDRR and stack with data from carrier
        Data = SID_dataset(FileName= filePath, config=config)
        Data.start_time = Data.start_time
        Data.end_time = Data.end_time 


        #Postprocessing = PostProcessed_Data(Data = Data)
        #Data_filtered = Postprocessing.S2Nfilter()
        #visuals.plot_deformations(data = Data_filtered, datestring = ThisTime_str + "s2n_filtered")
        #Data_filtered.Make_CDF_output(label = "S2Nfilter/", TimeString = ThisTime_str)
        #del Postprocessing

        for n in range(3,4):

            #A: simple smoothing
            Postprocessing = PostProcessed_Data(Data = Data)
            Data_isoSmoothed = Postprocessing.Kernel_smoothing(n = n)
            visuals.plot_deformations(data = Data_isoSmoothed, datestring = ThisTime_str + "Isotropic_smoothed_n%s" % n)
            visuals.plot_data_structure(data=Data_isoSmoothed, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Isotropic_smoothed_n%s" % n, show_shear =1)
            del Postprocessing

            Postprocessing = PostProcessed_Data(Data = Data)
            Data_anisoSmoothed = Postprocessing.Anisotropic_smoothing(n = n)
            visuals.plot_deformations(data = Data_anisoSmoothed, datestring = ThisTime_str + "Anisotropic_smoothed_n%s" % n)
            visuals.plot_data_structure(data=Data_anisoSmoothed, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Anisotropic_smoothed_n%s" % n, show_shear =1)
            del Postprocessing


            #B: Anisotropic smoothing
            Postprocessing = PostProcessed_Data(Data = Data)
            Data_filtered = Postprocessing.S2Nfilter()
            Data_filtered2 = Postprocessing.Anisotropic_smoothing(n = n)
            visuals.plot_deformations(data = Data_filtered2, datestring = ThisTime_str + "Anisotropic_filtered_n%s" % n)
            visuals.plot_data_structure(data=Data_filtered2, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Anisotropic_filtered_n%s" % n, show_shear =1)

            #C: Smoothing outside of LKFs
            Postprocessing2 = PostProcessed_Data(Data = Data)
            Data_filtered3 = Postprocessing2.Kernel_smoothing(xr_ref = Postprocessing.aniso_filtered_dataset, n = n)
            #visuals.plot_deformations(data = Data_filtered3, datestring = ThisTime_str + "Kernelfiltered_n%s" % n)
            visuals.plot_data_structure(data=Data_filtered3, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Kernelfiltered_n%s" % n, show_shear =1)

            #D: Combine the LKF deformations and out--of-LKFs deformaitons
            Data_filtered4 = Postprocessing2.Combined_filters(xr_in = Postprocessing.aniso_filtered_dataset)
            #visuals.plot_deformations(data = Data_filtered4, datestring = ThisTime_str + "Combined_filters_n%s" % n)
            visuals.plot_data_structure(data=Data_filtered4, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Combined_filters_n%s" % n, show_shear =1)

            #--------------------
            # Save results in post-processed datasets
            #--------------------

            Data_isoSmoothed.Make_CDF_output(label = "IsoSmoothed_n%s/" % n, TimeString = ThisTime_str)
            Data_anisoSmoothed.Make_CDF_output(label = "AnisoSmoothed_n%s/" % n, TimeString = ThisTime_str)
            Data_filtered2.Make_CDF_output(label = "Itkin_n%s/" % n, TimeString = ThisTime_str)
            Data_filtered3.Make_CDF_output(label = "Lowdefs_n%s/" % n, TimeString = ThisTime_str)
            Data_filtered4.Make_CDF_output(label = "CombinedFilters_n%s/" % n, TimeString = ThisTime_str)

            del Data_isoSmoothed
            del Data_anisoSmoothed
            del Data_filtered2
            del Data_filtered3
            del Data_filtered4
            del Postprocessing
            del Postprocessing2

            #--------------------
            # Repeat but with recomputed deformations instead of smoothing
            #--------------------

            #A: simple smoothing
            Postprocessing = PostProcessed_Data(Data = Data)
            Data_isoSmoothed = Postprocessing.Kernel_smoothing(n = n, Compute_defs = 1)
            visuals.plot_deformations(data = Data_isoSmoothed, datestring = ThisTime_str + "Isotropic_smoothed_n%s_recomputed" % n)
            visuals.plot_data_structure(data=Data_isoSmoothed, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Isotropic_smoothed_n%s_recomputed" % n, show_shear =1)
            del Postprocessing


            Postprocessing = PostProcessed_Data(Data = Data)
            Data_anisoSmoothed = Postprocessing.Anisotropic_smoothing(n = n, Compute_defs = 1)
            visuals.plot_deformations(data = Data_anisoSmoothed, datestring = ThisTime_str + "Anisotropic_smoothed_n%s_recomputed" % n)
            visuals.plot_data_structure(data=Data_anisoSmoothed, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Anisotropic_smoothed_n%s_recomputed" % n, show_shear =1)
            del Postprocessing

            #B: Anisotropic smoothing
            Postprocessing = PostProcessed_Data(Data = Data)
            Data_filtered = Postprocessing.S2Nfilter()
            Data_filtered2 = Postprocessing.Anisotropic_smoothing(n = n, Compute_defs = 1)
            visuals.plot_deformations(data = Data_filtered2, datestring = ThisTime_str + "Anisotropic_filtered_n%s_recomputed" % n)
            visuals.plot_data_structure(data=Data_filtered2, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Anisotropic_filtered_n%s_recomputed" % n, show_shear =1)

            #C: Smoothing outside of LKFs
            Postprocessing2 = PostProcessed_Data(Data = Data)
            Data_filtered3 = Postprocessing2.Kernel_smoothing(xr_ref = Postprocessing.aniso_filtered_dataset, Compute_defs = 1)
            #visuals.plot_deformations(data = Data_filtered3, datestring = ThisTime_str + "Kernelfiltered_n%s" % n)
            visuals.plot_data_structure(data=Data_filtered3, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Kernelfiltered_n%s_recomputed" % n, show_shear =1)

            #D: Combine the LKF deformations and out--of-LKFs deformaitons
            Data_filtered4 = Postprocessing2.Combined_filters(xr_in = Postprocessing.aniso_filtered_dataset)
            #visuals.plot_deformations(data = Data_filtered4, datestring = ThisTime_str + "Combined_filters_n%s" % n)
            visuals.plot_data_structure(data=Data_filtered4, idpair =  int(config['visualize']['idpair']), datestring = ThisTime_str + "Combined_filters_n%s_recomputed" % n, show_shear =1)

            #--------------------
            # Save results in post-processed datasets
            #--------------------

            Data_isoSmoothed.Make_CDF_output(label = "IsoSmoothed_n%s_recomputed/" % n, TimeString = ThisTime_str)
            Data_anisoSmoothed.Make_CDF_output(label = "AnisoSmoothed_n%s_recomputed/" % n, TimeString = ThisTime_str)
            Data_filtered2.Make_CDF_output(label = "Itkin_n%ss_recomputed/" % n, TimeString = ThisTime_str)
            Data_filtered3.Make_CDF_output(label = "Lowdefs_n%ss_recomputed/" % n, TimeString = ThisTime_str)
            Data_filtered4.Make_CDF_output(label = "CombinedFilters_n%ss_recomputed/" % n, TimeString = ThisTime_str)

            del Data_isoSmoothed
            del Data_anisoSmoothed
            del Data_filtered2
            del Data_filtered3
            del Data_filtered4
            del Postprocessing
            del Postprocessing2

        del visuals
        del Data
        del Data_filtered



    # Display the computation time
    print("--- %s seconds ---" % (time.time() - start_time))
