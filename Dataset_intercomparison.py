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

if __name__ == '__main__':


    #------------------------------------------------------------------
    # Reading the  namelist
    #------------------------------------------------------------------

    # Reading config
    config = configparser.ConfigParser()
    config.read('namelist_intercomparison.ini')

    path = config['IO']['Experiment_list_path']
    output_folder = config['IO']['output_folder']
    TimeTool = TimeUtil(config = config['Date_options'])

    List_legend = []
    ExperimentList = []
    with path as file:
        for line in file:
            Exp = line.split(" ")
            List_legend.append(Exp[0].strip())
            ExperimentList.append(Exp[1].strip())

    print(NamelistList)
    List_legend = List_legend[1:]
    ExperimentList = ExperimentList[1:]


    #------------------------------------------------------------------
    # Create objects for required data analysis in namelist
    #------------------------------------------------------------------

    #Initialize statistics objects
    #Initialise PDFs for the triangle areas.

#    if config['visualize']['show_spatial_scale_dist'] == 'True':
#        #Spatial scale distributions
#        A_dist_all = Data_Distribution(LeftBC= 0.0, RightBC = 20.0, nbins = 40, label = "combined")
#        A_dist_S1  = Data_Distribution(LeftBC= 0.0, RightBC = 20.0, nbins = 40, label = "S1")
#        A_dist_RCM = Data_Distribution(LeftBC= 0.0, RightBC = 20.0, nbins = 40, label = "RCM")
#
#    if config['visualize']['show_error_dist'] == 'True':
#        #Error distributions
#        errtot_dist_all  = Data_Distribution(LeftBC= 0.0, RightBC = 1.0, nbins = 100, label = "combined")
#        errtot_dist_S1   = Data_Distribution(LeftBC= 0.0, RightBC = 1.0, nbins = 100, label = "S1")
#        errtot_dist_RCM  = Data_Distribution(LeftBC= 0.0, RightBC = 1.0, nbins = 100, label = "RCM")
#        #Signal-to-noise distributions
#        s2n_dist_all     = Data_Distribution(LeftBC= 0.0, RightBC = 5.0, nbins = 50, label = "combined")
#        s2n_dist_S1      = Data_Distribution(LeftBC= 0.0, RightBC = 5.0, nbins = 50, label = "S1")
#        s2n_dist_RCM     = Data_Distribution(LeftBC= 0.0, RightBC = 5.0, nbins = 50, label = "RCM")




    #--------------------------------------------------------------
    # INITIALISING THE MEMORY TERM
    #    Loading data from earlier days (6), but valid for 1st analysis date.
    #    This step is necessary now that the data is stored according to
    #    the start time. Here, we fetch data from earlier date that are
    #    valid for the first date of the required analysis period.
    #
    #    We are thus building a Memory term containing data to be carried
    #    over the next step
    #
    #    - Data_Mem: Data memory carrier
    #--------------------------------------------------------------

    # Start the clock
    start_time = time.time()
    Delta_days_init = 0

    refTime = TimeTool.ThisTime
    Data_Mem = None

    #------------------------------------------------------------------
    # STARTING ANALYSIS
    #------------------------------------------------------------------

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

        for Exp in ExperimentList:
            filePath = Expp + TimeTool.ThisTimeFile

            #Load visualisation tool
            visuals = visualisation(config=config)

            #Load new Data from SIDRR and stack with data from carrier
            Data = SID_dataset(FileName= filePath, config=config)
            Data.start_time = Data.start_time
            Data.end_time = Data.end_time

            visuals.plot_SARscene_intercom(data=Data, idpair = [38,39,54], 
                                           datestring = ThisTime_str, n = n)



            #------------------------------------------------------------------
            # Area and errors analysis
            #------------------------------------------------------------------

            if config['visualize']['show_spatial_scale_dist'] == 'True':

                #Spatial distribution
                A_dist_all.add2hist_1D(Data = (Data.A[:]**0.5)/1000.0)
                A_dist_S1.add2hist_1D(Data = (Data.A[Data.satellite[:]==1]**0.5)/1000.0)
                A_dist_RCM.add2hist_1D(Data = (Data.A[Data.satellite[:]==0]**0.5)/1000.0)

            if config['visualize']['show_error_dist'] == 'True':
                #Error distribution
                errtot_dist_all.add2hist_1D(Data = Data.errtot[Data.errtot[:]>0.0])
                errtot_dist_S1.add2hist_1D( Data = Data.errtot[np.logical_and(Data.satellite==1 , Data.errtot>0.0)])
                errtot_dist_RCM.add2hist_1D(Data = Data.errtot[np.logical_and(Data.satellite==0 , Data.errtot>0.0)])
                #Signal-to-noise distribution
                s2n_dist_all.add2hist_1D(Data = Data.s2n[Data.s2n[:]>0.0])
                s2n_dist_S1.add2hist_1D( Data = Data.s2n[np.logical_and(Data.satellite==1, Data.s2n>0.0)])
                s2n_dist_RCM.add2hist_1D(Data = Data.s2n[np.logical_and(Data.satellite==0, Data.s2n>0.0)])

                #------------------------------------------------------------------
                # Show the results if specified in the namelist
                #------------------------------------------------------------------

                #Figure showing normal, shear and rotation rates. And Area. (Fig. 2 in Plante et al., 2024)
                if config['visualize']['plot_deformation'] == 'True':
                    visuals.plot_deformations(data = Data, datestring = ThisTime_str)
                    visuals.show_tripcolor_field(data=Data, Field = Data.A,
                                          title = "Triangle Areas", label = "Area",
                                          datestring=ThisTime_str)

                #Figure showing the errors in a given date and pair (Fig. 7 in Plante et al., 2024)
                if config['visualize']['plot_errors'] == 'True':
                    visuals.plot_errors(data = Data, datestring = ThisTime_str, idpair = int(config['visualize']['idpair']))

            del visuals
            del Data


    #--------------------------------------------------------------------------------
    # Make figures showing dataset characteristics
    #--------------------------------------------------------------------------------

    visuals = visualisation(config = config)
    # Make figure showing the spatio-temportal coverage of the SIDRR data in the analysed period
    # (Fig. 5 in Plante et al., 2024)
    if config['visualize']['show_spatial_coverage'] == 'True':
        visuals.show_spatial_coverage(distribution_2D = FreqMap,
                                      distribution_2D_2 = FreqMap_S1,
                                      distribution_2D_3 = FreqMap_RCM,
                                      datestring = TimeTool.StartDate_str + '_' + TimeTool.EndDate_str)

    # Make figure showing the distribution of data spatial scales
    # (Fig. 3 in Plante et al., 2024)
    if config['visualize']['show_spatial_scale_dist'] == 'True':
        visuals.plot_area_dist(dist1 = A_dist_S1,   dist2 = A_dist_RCM,dist3 = A_dist_all,
                               distDt1 = T_dist_S1, distDt2 = T_dist_RCM,distDt3 = T_dist_all,
                               datestring = TimeTool.StartDate_str + '_' + TimeTool.EndDate_str)
    # Make figure showing the distribution of propagation errors
    # (Fig. 6 in Plante et al., 2024)
    if config['visualize']['show_error_dist'] == 'True':
        visuals.plot_error_dist(dist1 = errtot_dist_S1, dist2 = errtot_dist_RCM, dist3 = errtot_dist_all,
                               dist_s1 = s2n_dist_S1, dist_s2 = s2n_dist_RCM, dist_s3 = s2n_dist_all, data = Data_Mem,
                               datestring = TimeTool.StartDate_str + '_' + TimeTool.EndDate_str)

    # Make figure showing the time series of area coveraged
    # (Fig. 4 in Plante et al., 2024)
    if config['visualize']['show_coverage_tseries'] == 'True':
        visuals.show_coverage_tseries(Data1 = FreqMap,
                                      Data2 = FreqMap_S1,
                                      Data3 = FreqMap_RCM,
                                      datestring = TimeTool.StartDate_str + '_' + TimeTool.EndDate_str)

    # Display the computation time
    print("--- %s seconds ---" % (time.time() - start_time))
