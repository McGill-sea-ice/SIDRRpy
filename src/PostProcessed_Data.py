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
import xarray as xr
import copy
from multiprocessing import Pool, shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from math import floor,atan2
import cartopy.crs as ccrs
import pyproj as pyproj

# Loads netCDF data
class PostProcessed_Data:

    def __init__(self,Data= None):
        """
        """

        print('--- Loading data (netcdf) ---')
        self.Data = copy.deepcopy(Data)
        a = len(self.Data.div)
        # Make dataset with all scaled variables
        div = xr.DataArray(Data.div.copy(), dims=['x'], name='div',coords={"x": np.arange(a)})
        shr = xr.DataArray(Data.shr.copy(), dims=['x'], name='shr',coords={"x": np.arange(a)})
        vrt = xr.DataArray(Data.vrt.copy(), dims=['x'], name='vrt',coords={"x": np.arange(a)})
        A = xr.DataArray(Data.A.copy(), dims=['x'], name='A',coords={"x": np.arange(a)})
        dudx = xr.DataArray(Data.dudx.copy()*Data.A.copy(), dims=['x'], name='dudx',coords={"x": np.arange(a)})
        dudy = xr.DataArray(Data.dudy.copy()*Data.A.copy(), dims=['x'], name='dudy',coords={"x": np.arange(a)})
        dvdx = xr.DataArray(Data.dvdx.copy()*Data.A.copy(), dims=['x'], name='dvdx',coords={"x": np.arange(a)})
        dvdy = xr.DataArray(Data.dvdy.copy()*Data.A.copy(), dims=['x'], name='dvdy',coords={"x": np.arange(a)})
        ids1 = xr.DataArray(Data.ids1.copy(), dims=['x'], name='ids1',coords={"x": np.arange(a)})
        ids2 = xr.DataArray(Data.ids2.copy(), dims=['x'], name='ids2',coords={"x": np.arange(a)})
        ids3 = xr.DataArray(Data.ids3.copy(), dims=['x'], name='ids3',coords={"x": np.arange(a)})
        idpair = xr.DataArray(Data.idpair.copy(), dims=['x'], name='idpair',coords={"x": np.arange(a)})
        tr_day = xr.DataArray(Data.day_flag.copy(), dims=['x'], name='tr_day',coords={"x": np.arange(a)})
        s2n = xr.DataArray(Data.s2n.copy(), dims=['x'], name='s2n',coords={"x": np.arange(a)})

        self.dataset = xr.Dataset({'div': div,
                                  'shr':  shr,
                                  'vrt':  vrt,
                                  'dudx': dudx,
                                  'dudy':  dudy,
                                  'dvdx':  dvdx,
                                  'dvdy':  dvdy,
                                  'ids1':  ids1,
                                  'ids2':  ids2,
                                  'ids3':  ids3,
                                  'idpair': idpair,
                                  'tr_day': tr_day,
                                  's2n': s2n,
                                  'A': A})
        self.Smoothed_dataset = np.nan
        self.aniso_filtered_dataset = np.nan
        self.n = 3
        self.manager = None
        self.shared_results = None
        self.Data_in = None
        self.min_index = None

        self.sigx = Data.trackingerror
        sigx = self.sigx

    def S2Nfilter(self, xr_in = None, xr_smooth = None):
        if xr_in is None:
            xr_in = self.dataset
        if xr_smooth is None:
            xr_smooth = self.Smoothed_dataset
        xr_out = xr_in.where(xr_in['s2n']>1.3,np.nan)
        self.xr_filtered = copy.deepcopy(xr_out)
        self.Data.div = np.array(xr_out['div']).copy()
        self.Data.shr = np.array(xr_out['shr']).copy()
        self.Data.vrt = np.array(xr_out['vrt']).copy()
        self.Data.A = np.array(xr_out['A']).copy()
        self.Data.dudx = np.array(xr_out['dudx']).copy()/np.array(xr_out['A']).copy()
        self.Data.dudy = np.array(xr_out['dudy']).copy()/np.array(xr_out['A']).copy()
        self.Data.dvdx = np.array(xr_out['dvdx']).copy()/np.array(xr_out['A']).copy()
        self.Data.dvdy = np.array(xr_out['dvdy']).copy()/np.array(xr_out['A']).copy()

        self.Data.dudx[np.isnan(np.array(xr_out['div']))] = np.nan
        self.Data.dudy[np.isnan(np.array(xr_out['div']))] = np.nan
        self.Data.dvdx[np.isnan(np.array(xr_out['div']))] = np.nan
        self.Data.dvdy[np.isnan(np.array(xr_out['div']))] = np.nan

        return self.Data


    def Kernel_smoothing(self, xr_in = None, xr_ref = None, n = None, Compute_defs = None):

        if n is None:
            n = 1
        if xr_in is None:
            xr_in = self.dataset
        if xr_ref is not None:
            Data_ref = np.array(xr_ref['div']).copy()
        else:
            Data_ref = self.Data.div.copy()*np.nan

        Nblocks = len(np.unique(self.Data.idpair))
        nb = 0
        print(len(self.Data.div))
        for nblock in np.unique(self.Data.idpair):
            print("Doing pairs %s/%s" % (nb, Nblocks))
            nb = nb+1
            condi = (self.Data.idpair== nblock)

            Data_shared = np.array([self.Data.div[condi].copy(),#0
                           self.Data.shr[condi].copy(),#1
                           self.Data.vrt[condi].copy(),#2
                           self.Data.A[condi].copy(),#3
                           self.Data.dudx[condi].copy(),#4
                           self.Data.dudy[condi].copy(),#5
                           self.Data.dvdx[condi].copy(),#6
                           self.Data.dvdy[condi].copy(),#7
                           self.Data.ids1[condi].copy(),#8
                           self.Data.ids2[condi].copy(),#9
                           self.Data.ids3[condi].copy(),#10
                           self.Data.idpair[condi].copy(),#11
                           self.Data.day_flag[condi].copy(),#12
                           self.Data.s2n[condi].copy(),#13
                           self.Data.errA[condi].copy(),#14
                           self.Data.errI[condi].copy(),#15
                           self.Data.errII[condi].copy(),#16
                           self.Data.errvrt[condi].copy(),#17
                           Data_ref[condi].copy()])#18
            print(Data_shared.shape)
            Data_in = Data_shared.copy()
            Data_shared = Data_shared*np.nan
            shm = shared_memory.SharedMemory(create=True, size=Data_shared.nbytes)
            shared_arr = np.ndarray(Data_shared.shape, dtype=Data_shared.dtype, buffer=shm.buf)
            np.copyto(shared_arr, Data_shared)
            print(Data_shared.shape)

            if Compute_defs is not None:

                condi_pts = (self.Data.pts_idpair==nblock)
                dt = self.Data.end_time[condi]- self.Data.start_time[condi] 
                Data_pts = np.array([self.Data.start_lat[condi_pts].copy(), #0
                           self.Data.start_lon[condi_pts].copy(), #1
                           self.Data.end_lat[condi_pts].copy(), #2
                           self.Data.end_lon[condi_pts].copy(), #3
                           self.Data.pts_dayflag[condi_pts].copy(), #4
                           self.Data.pts_idpair[condi_pts].copy()]) #5
            else:
                dt = np.arange(len(shared_arr[0,:]))*np.nan
                Data_pts = np.nan

            args = [(j, shm.name, shared_arr.shape, Data_in,n, Data_pts, dt[j], self.sigx) for j in range(0, len(shared_arr[0,:]))]
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(Paralleled_Kernel_smoothing, *arg) for arg in args]
                for future in as_completed(futures):
                    pass #future.result()  # handle exceptions if needed

            self.Data.div[condi] = shared_arr[0,:].copy()
            self.Data.shr[condi] = shared_arr[1,:].copy()
            self.Data.vrt[condi] = shared_arr[2,:].copy()
            self.Data.A[condi]   = shared_arr[3,:].copy()
            self.Data.dudx[condi] = shared_arr[4,:].copy()
            self.Data.dudy[condi] = shared_arr[5,:].copy()
            self.Data.dvdx[condi] = shared_arr[6,:].copy()
            self.Data.dvdy[condi] = shared_arr[7,:].copy()

            shm.close()
            shm.unlink()
            del(Data_shared)
            del(shared_arr)


        #----------------------------------
        #   Update the DS
        #----------------------------------
        a = len(self.Data.div)

        # Make dataset with all scaled variables
        div = xr.DataArray(self.Data.div.copy(), dims=['x'], name='div',coords={"x": np.arange(a)})
        shr = xr.DataArray(self.Data.shr.copy(), dims=['x'], name='shr',coords={"x": np.arange(a)})
        vrt = xr.DataArray(self.Data.vrt.copy(), dims=['x'], name='vrt',coords={"x": np.arange(a)})
        A = xr.DataArray(self.Data.A.copy(), dims=['x'], name='A',coords={"x": np.arange(a)})
        dudx = xr.DataArray(self.Data.dudx.copy()*self.Data.A.copy(), dims=['x'], name='dudx',coords={"x": np.arange(a)})
        dudy = xr.DataArray(self.Data.dudy.copy()*self.Data.A.copy(), dims=['x'], name='dudy',coords={"x": np.arange(a)})
        dvdx = xr.DataArray(self.Data.dvdx.copy()*self.Data.A.copy(), dims=['x'], name='dvdx',coords={"x": np.arange(a)})
        dvdy = xr.DataArray(self.Data.dvdy.copy()*self.Data.A.copy(), dims=['x'], name='dvdy',coords={"x": np.arange(a)})
        ids1 = xr.DataArray(self.Data.ids1.copy(), dims=['x'], name='ids1',coords={"x": np.arange(a)})
        ids2 = xr.DataArray(self.Data.ids2.copy(), dims=['x'], name='ids2',coords={"x": np.arange(a)})
        ids3 = xr.DataArray(self.Data.ids3.copy(), dims=['x'], name='ids3',coords={"x": np.arange(a)})
        idpair = xr.DataArray(self.Data.idpair.copy(), dims=['x'], name='idpair',coords={"x": np.arange(a)})
        tr_day = xr.DataArray(self.Data.day_flag.copy(), dims=['x'], name='tr_day',coords={"x": np.arange(a)})
        s2n = xr.DataArray(self.Data.s2n.copy(), dims=['x'], name='s2n',coords={"x": np.arange(a)})

        self.Smoothed_dataset = xr.Dataset({'div': div,
                                  'shr':  shr,
                                  'vrt':  vrt,
                                  'dudx': dudx,
                                  'dudy':  dudy,
                                  'dvdx':  dvdx,
                                  'dvdy':  dvdy,
                                  'ids1':  ids1,
                                  'ids2':  ids2,
                                  'ids3':  ids3,
                                  'idpair': idpair,
                                  'tr_day': tr_day,
                                  's2n': s2n,
                                  'A': A})

        return self.Data

    def Anisotropic_smoothing(self, xr_in = None, n = None, Compute_defs = None):

        if xr_in is None:
            xr_in = self.dataset

        #if self.xr_filtered is None:
        #    print("ERROR: Need s2n filtered data to use the anisotropic smoothing")
        #    return
        if n is not None:
            self.n = n
        a = len(self.Data.div)
        Nblocks = len(np.unique(self.Data.idpair))
        nb = 0
        for nblock in np.unique(self.Data.idpair):
            print("Doing pairs %s/%s" % (nb, Nblocks))
            nb = nb+1
            condi = (self.Data.idpair== nblock)

            Data_shared = np.array([self.Data.div[condi].copy(),#0
                           self.Data.shr[condi].copy(),#1
                           self.Data.vrt[condi].copy(),#2
                           self.Data.A[condi].copy(),#3
                           self.Data.dudx[condi].copy(),#4
                           self.Data.dudy[condi].copy(),#5
                           self.Data.dvdx[condi].copy(),#6
                           self.Data.dvdy[condi].copy(),#7
                           self.Data.ids1[condi].copy(),#8
                           self.Data.ids2[condi].copy(),#9
                           self.Data.ids3[condi].copy(),#10
                           self.Data.idpair[condi].copy(),#11
                           self.Data.day_flag[condi].copy(),#12
                           self.Data.s2n[condi].copy(),#13
                           self.Data.errA[condi].copy(),#14
                           self.Data.errI[condi].copy(),#15
                           self.Data.errII[condi].copy(),#16
                           self.Data.errvrt[condi].copy()])#17

            Data_in = Data_shared.copy()
            Data_shared = Data_shared*np.nan
            shm = shared_memory.SharedMemory(create=True, size=Data_shared.nbytes)
            shared_arr = np.ndarray(Data_shared.shape, dtype=Data_shared.dtype, buffer=shm.buf)
            np.copyto(shared_arr, Data_shared)


            if Compute_defs is not None:

                condi_pts = (self.Data.pts_idpair==nblock)
                dt = self.Data.end_time[condi]- self.Data.start_time[condi] 
                Data_pts = np.array([self.Data.start_lat[condi_pts].copy(), #0
                           self.Data.start_lon[condi_pts].copy(), #1
                           self.Data.end_lat[condi_pts].copy(), #2
                           self.Data.end_lon[condi_pts].copy(), #3
                           self.Data.pts_dayflag[condi_pts].copy(), #4
                           self.Data.pts_idpair[condi_pts].copy()]) #5
            else:
                dt = np.arange(len(shared_arr[0,:]))*np.nan
                Data_pts = np.nan

            args = [(j, n, shm.name, shared_arr.shape, Data_in, Data_pts, dt[j], self.sigx) for j in range(0, len(shared_arr[0,:]))]

            with ProcessPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(Paralleled_Anisotropic_smoothing, *arg) for arg in args]
                for future in as_completed(futures):
                    pass

            self.Data.div[condi] = shared_arr[0,:].copy()
            self.Data.shr[condi] = shared_arr[1,:].copy()
            self.Data.vrt[condi] = shared_arr[2,:].copy()
            self.Data.A[condi]   = shared_arr[3,:].copy()
            self.Data.dudx[condi] = shared_arr[4,:].copy()
            self.Data.dudy[condi] = shared_arr[5,:].copy()
            self.Data.dvdx[condi] = shared_arr[6,:].copy()
            self.Data.dvdy[condi] = shared_arr[7,:].copy()

            shm.close()
            shm.unlink()
            del(Data_shared)
            del(shared_arr)

        #----------------------------------
        #   Update the DS
        #----------------------------------

        a = len(self.Data.div)
        # Make dataset with all scaled variables
        div = xr.DataArray(self.Data.div.copy(), dims=['x'], name='div',coords={"x": np.arange(a)})
        shr = xr.DataArray(self.Data.shr.copy(), dims=['x'], name='shr',coords={"x": np.arange(a)})
        vrt = xr.DataArray(self.Data.vrt.copy(), dims=['x'], name='vrt',coords={"x": np.arange(a)})
        A = xr.DataArray(self.Data.A.copy(), dims=['x'], name='A',coords={"x": np.arange(a)})
        dudx = xr.DataArray(self.Data.dudx.copy()*self.Data.A.copy(), dims=['x'], name='dudx',coords={"x": np.arange(a)})
        dudy = xr.DataArray(self.Data.dudy.copy()*self.Data.A.copy(), dims=['x'], name='dudy',coords={"x": np.arange(a)})
        dvdx = xr.DataArray(self.Data.dvdx.copy()*self.Data.A.copy(), dims=['x'], name='dvdx',coords={"x": np.arange(a)})
        dvdy = xr.DataArray(self.Data.dvdy.copy()*self.Data.A.copy(), dims=['x'], name='dvdy',coords={"x": np.arange(a)})
        ids1 = xr.DataArray(self.Data.ids1.copy(), dims=['x'], name='ids1',coords={"x": np.arange(a)})
        ids2 = xr.DataArray(self.Data.ids2.copy(), dims=['x'], name='ids2',coords={"x": np.arange(a)})
        ids3 = xr.DataArray(self.Data.ids3.copy(), dims=['x'], name='ids3',coords={"x": np.arange(a)})
        idpair = xr.DataArray(self.Data.idpair.copy(), dims=['x'], name='idpair',coords={"x": np.arange(a)})
        tr_day = xr.DataArray(self.Data.day_flag.copy(), dims=['x'], name='tr_day',coords={"x": np.arange(a)})
        s2n = xr.DataArray(self.Data.s2n.copy(), dims=['x'], name='s2n',coords={"x": np.arange(a)})

        self.aniso_filtered_dataset = xr.Dataset({'div': div,
                                  'shr':  shr,
                                  'vrt':  vrt,
                                  'dudx': dudx,
                                  'dudy':  dudy,
                                  'dvdx':  dvdx,
                                  'dvdy':  dvdy,
                                  'ids1':  ids1,
                                  'ids2':  ids2,
                                  'ids3':  ids3,
                                  'idpair': idpair,
                                  'tr_day': tr_day,
                                  's2n': s2n,
                                  'A': A})


        return self.Data

    def Combined_filters(self, xr_in = None, xr_smooth = None):
        if xr_in is None:
            xr_in = self.aniso_filtered_dataset
        if xr_smooth is None:
            xr_smooth = self.Smoothed_dataset
        xr_out = xr_in.where(xr_in.notnull,xr_smooth)
        self.xr_filtered = xr_out
        self.Data.div = np.array(xr_out['div'])
        self.Data.shr = np.array(xr_out['shr'])
        self.Data.vrt = np.array(xr_out['vrt'])
        self.Data.A = np.array(xr_out['A'])
        self.Data.dudx = np.array(xr_out['dudx'])/ np.array(xr_out['A'])
        self.Data.dudy = np.array(xr_out['dudy'])/ np.array(xr_out['A'])
        self.Data.dvdx = np.array(xr_out['dvdx'])/ np.array(xr_out['A'])
        self.Data.dvdy = np.array(xr_out['dvdy'])/ np.array(xr_out['A'])

        return self.Data




def Paralleled_Anisotropic_smoothing(j, n, shm_name, shape, Data_in, Data_pts, dt, sigx):

    shm = shared_memory.SharedMemory(name=shm_name)
    shared_results = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
    if np.isnan(Data_in[0,j]):
        pass
    else:
        day_flag = Data_in[12,j].copy()
        idpair = Data_in[11,j].copy()
        condi = (Data_in[11,:] == idpair) & (Data_in[12,:] == day_flag)
        #------Create the pair xarray
        a = len(Data_in[0,condi])

        # Make dataset with all scaled variables
        div = xr.DataArray(Data_in[0,condi].copy(), dims=['x'], name='div',coords={"x": np.arange(a)})
        shr = xr.DataArray(Data_in[1,condi].copy(), dims=['x'], name='shr',coords={"x": np.arange(a)})
        vrt = xr.DataArray(Data_in[2,condi].copy(), dims=['x'], name='vrt',coords={"x": np.arange(a)})
        A =   xr.DataArray(Data_in[3,condi].copy(), dims=['x'], name='A',coords={"x": np.arange(a)})
        dudx = xr.DataArray(Data_in[4,condi].copy()*Data_in[3,condi].copy(), dims=['x'], name='dudx',coords={"x": np.arange(a)})
        dudy = xr.DataArray(Data_in[5,condi].copy()*Data_in[3,condi].copy(), dims=['x'], name='dudy',coords={"x": np.arange(a)})
        dvdx = xr.DataArray(Data_in[6,condi].copy()*Data_in[3,condi].copy(), dims=['x'], name='dvdx',coords={"x": np.arange(a)})
        dvdy = xr.DataArray(Data_in[7,condi].copy()*Data_in[3,condi].copy(), dims=['x'], name='dvdy',coords={"x": np.arange(a)})
        ids1 = xr.DataArray(Data_in[8,condi].copy(), dims=['x'], name='ids1',coords={"x": np.arange(a)})
        ids2 = xr.DataArray(Data_in[9,condi].copy(), dims=['x'], name='ids2',coords={"x": np.arange(a)})
        ids3 = xr.DataArray(Data_in[10,condi].copy(), dims=['x'], name='ids3',coords={"x": np.arange(a)})
        s2n = xr.DataArray(Data_in[13,condi].copy(), dims=['x'], name='s2n',coords={"x": np.arange(a)})

        xr_pair = xr.Dataset({'div': div,
                                  'shr':  shr,
                                  'vrt':  vrt,
                                  'dudx': dudx,
                                  'dudy':  dudy,
                                  'dvdx':  dvdx,
                                  'dvdy':  dvdy,
                                  'ids1':  ids1,
                                  'ids2':  ids2,
                                  'ids3':  ids3,
                                  's2n': s2n,
                                  'A': A})
        min_index = np.squeeze(np.where(condi))[0]
        max_index = np.squeeze(np.where(condi))[-1]+1
        i = j- min_index

        #----------------------------
        # Make a kernel for the smoothing
        #----------------------------
        #get the list of triangles to be averaged
        gen_num = np.array(xr_pair['ids1']).copy()*np.nan
        gen_num[i] = 1
        for k in range(1,n+1):
            #----Find neighbouring triangles----

            #Get the id of vertexes
            gen_num_da = xr.DataArray(gen_num.copy(), dims='x', coords={'x': xr_pair.coords['x']})
            xr_list = xr_pair.where(gen_num_da == k, drop = True)
            idlist = np.unique(np.concatenate([np.array(xr_list['ids1']),  xr_list['ids2'], xr_list['ids3']]))

            #Get triangles that are neighbour to this generation
            condi = np.logical_and(( np.array(xr_pair['ids1'].isin(idlist)).astype(int)+
            np.array(xr_pair['ids2'].isin(idlist)).astype(int)+
            np.array(xr_pair['ids3'].isin(idlist)).astype(int))==2, np.array(xr_pair['A'])>0.0)
            n_tri = len(np.array(np.array(xr_pair['ids1'])[condi]))

            #----Update the neighboor flag with new rows----
            gen_num[np.logical_and(condi,np.isnan(gen_num))] = k+1

            if n_tri == 0:
                break

        #----------------------------
        # Update the deformation data with the mean of selected triangles
        #----------------------------

        gen_num_xr = xr.DataArray(gen_num, dims='x', coords={'x': xr_pair.coords['x']})

        if np.isnan(dt):

            xr_tri = xr_pair.where(gen_num_xr.notnull, drop=True)
            n_tri = len(np.array(xr_tri['A']))
            Atot = np.sum(np.array(xr_tri['A']))

            if len(np.array(xr_tri['ids1']))>n:
                xr_tri = xr_tri.sum(skipna=True)
                shared_results[0,j] = np.array(xr_tri['dudx'])/Atot + np.array(xr_tri['dvdy'])/Atot
                shared_results[1,j] = ((  np.array(xr_tri['dudx'])/Atot - np.array(xr_tri['dvdy'])/Atot)**2.0 +
                            (np.array(xr_tri['dudy'])/Atot + np.array(xr_tri['dvdx'])/Atot)**2.0  )**0.5
                shared_results[2,j] = np.array(xr_tri['dvdx'])/Atot - np.array(xr_tri['dudy'])/Atot
                shared_results[3,j] = Atot
                shared_results[4,j] = np.array(xr_tri['dudx'])/Atot
                shared_results[5,j] = np.array(xr_tri['dudy'])/Atot
                shared_results[6,j] = np.array(xr_tri['dvdx'])/Atot
                shared_results[7,j] = np.array(xr_tri['dvdy'])/Atot
                #print(Data_in[0,j], shared_results[0,j], n_tri)
            else:
                shared_results[0:8,j]  = np.nan

        else:

        #-------------------------------
        # Recompute the deformations if requested
        #-------------------------------

            proj = ccrs.NorthPolarStereo(central_longitude=0)
            trans = ccrs.PlateCarree()

            #vertices vectors
            condi_pts = (Data_pts[5,:] == idpair) & (Data_pts[4,:] == day_flag)
            start_lats = Data_pts[0,condi_pts]
            start_lons = Data_pts[1,condi_pts]
            end_lats = Data_pts[2,condi_pts]
            end_lons = Data_pts[3,condi_pts]

            xr_tri = xr_pair.where(~np.isnan(gen_num_xr), drop=True)
            indices = np.unique(np.concatenate([np.array(xr_tri['ids1']),  np.array(xr_tri['ids2']), np.array(xr_tri['ids3'])]))
            start_lats = start_lats[indices.astype(int)].copy()
            start_lons = start_lons[indices.astype(int)].copy()
            end_lats = end_lats[indices.astype(int)].copy()
            end_lons = end_lons[indices.astype(int)].copy()

            # Change to XY values, with origin on the cloud of pts centroid
            new_coords_start = proj.transform_points(trans, np.array(start_lons).copy(), np.array(start_lats).copy())
            new_coords_end = proj.transform_points(trans, np.array(end_lons).copy(), np.array(end_lats).copy())
            sx_list=new_coords_start[:,0].copy()-(np.sum(new_coords_start[:,0].copy())/len(new_coords_start[:,0]))
            ex_list=new_coords_end[:,0].copy()-(np.sum(new_coords_start[:,0].copy())/len(new_coords_start[:,0]))
            sy_list=new_coords_start[:,1].copy()-(np.sum(new_coords_start[:,1].copy())/len(new_coords_start[:,1]))
            ey_list=new_coords_end[:,1].copy()-(np.sum(new_coords_start[:,1].copy())/len(new_coords_start[:,1]))

            #Sort according to the angle with the centroid
            angles = np.arctan2(sy_list, sx_list)
            sorted_indices = np.argsort(angles)
            a = len(sx_list)
            u_list =  xr.DataArray(((ex_list[sorted_indices] - sx_list[sorted_indices]) / dt).copy(), dims=['x'], name='u_list',coords={"x": np.arange(a)})
            v_list =  xr.DataArray(((ey_list[sorted_indices] - sy_list[sorted_indices]) / dt).copy(), dims=['x'], name='v_list',coords={"x": np.arange(a)})
            sx_list =  xr.DataArray(sx_list[sorted_indices].copy(), dims=['x'], name='sx_list',coords={"x": np.arange(a)})
            sy_list =  xr.DataArray(sy_list[sorted_indices].copy(), dims=['x'], name='sy_list',coords={"x": np.arange(a)})
            ex_list =  xr.DataArray(ex_list[sorted_indices].copy(), dims=['x'], name='ex_list',coords={"x": np.arange(a)})
            ey_list =  xr.DataArray(ey_list[sorted_indices].copy(), dims=['x'], name='ey_list',coords={"x": np.arange(a)})

            xr_pts = xr.Dataset({'sx_list': sx_list,
                             'sy_list': sy_list,
                             'ex_list': ex_list,
                             'ey_list': ey_list,
                             'u_list':  u_list,
                             'v_list':  v_list})

            test = 11
            # Remove concave angles by computing the cross product of segments at each point
            while test > 10:
                test += 1
                ux = np.array(xr_pts['sx_list']).copy()
                vy = np.array(xr_pts['sy_list']).copy()
                uxm = np.array(xr_pts['sx_list']).copy()
                uxm[1:] = np.array(xr_pts['sx_list'])[:-1].copy()
                uxm[0] = np.array(xr_pts['sx_list'])[-1].copy()
                vym = np.array(xr_pts['sy_list']).copy()
                vym[1:] = np.array(xr_pts['sy_list'])[:-1].copy()
                vym[0] = np.array(xr_pts['sy_list'])[-1].copy()
                uxp = np.array(xr_pts['sx_list']).copy().copy()
                uxp[:-1] = np.array(xr_pts['sx_list'])[1:].copy()
                uxp[-1] = np.array(xr_pts['sx_list'])[0].copy()
                vyp = np.array(xr_pts['sy_list']).copy()
                vyp[:-1] = np.array(xr_pts['sy_list'])[1:].copy()
                vyp[-1] = np.array(xr_pts['sy_list'])[0].copy()
         
                u1 = ux - uxm
                u2 = uxp - ux
                v1 = vy - vym
                v2 = vyp-vy
                cross = (u1*v2)-(v1*u2)
                if len(cross[cross<0]) > 0:
                    cross = xr.DataArray(cross.copy(), dims=['x'], name='cr',coords={'x': xr_pts.coords['x']})
                    xr_pts = xr_pts.where(cross>0, drop = True)
                else:
                    test = 0

            sx = np.array(xr_pts['sx_list']).copy()
            sy = np.array(xr_pts['sy_list']).copy()

            sxp = np.array(xr_pts['sx_list']).copy()
            sxp[:-1] = np.array(xr_pts['sx_list'])[1:].copy()
            sxp[-1] = np.array(xr_pts['sx_list'])[0].copy()

            syp = np.array(xr_pts['sy_list']).copy()
            syp[:-1] = np.array(xr_pts['sy_list'])[1:].copy()
            syp[-1] = np.array(xr_pts['sy_list'])[0].copy()

            u = np.array(xr_pts['u_list']).copy()
            v = np.array(xr_pts['v_list']).copy()
            um = np.array(xr_pts['u_list']).copy()
            um[1:] = np.array(xr_pts['u_list'])[:-1].copy()
            um[0] = np.array(xr_pts['u_list'])[-1].copy()
            vm = np.array(xr_pts['v_list']).copy()
            vm[1:] = np.array(xr_pts['v_list'])[:-1].copy()
            vm[0] = np.array(xr_pts['v_list'])[-1].copy()
            up = np.array(xr_pts['u_list']).copy().copy()
            up[:-1] = np.array(xr_pts['u_list'])[1:].copy()
            up[-1] = np.array(xr_pts['u_list'])[0].copy()
            vp = np.array(xr_pts['v_list']).copy()
            vp[:-1] = np.array(xr_pts['v_list'])[1:].copy()
            vp[-1] = np.array(xr_pts['v_list'])[0].copy()


            A = np.sum(( sx*syp - sxp*sy)/2.0)
            dudx = np.sum((up+u)*(syp-sy))/(2.0*A) 
            dudy = np.sum((up+u)*(sx-sxp))/(2.0*A)  #Note that sx-sxp = -Delta x here
            dvdx = np.sum((vp+v)*(syp-sy))/(2.0*A) 
            dvdy = np.sum((vp+v)*(sx-sxp))/(2.0*A) #Note that sx-sxp = -Delta x here

            eps_I = dudx + dvdy
            eps_II =  ((dudx - dvdy)**2 + (dudy + dvdx)**2)**0.5
            vrt = dvdx - dudy

            eps_tot = ( eps_I**2 + eps_II**2 )**0.5

            # Compute the propagation errors
            sxp2 = np.array(xr_pts['sx_list']).copy()
            sxp2[:-2] = np.array(xr_pts['sx_list'])[2:].copy()
            sxp2[-2] = np.array(xr_pts['sx_list'])[0].copy()
            sxp2[-1] = np.array(xr_pts['sx_list'])[1].copy()

            syp2 = np.array(xr_pts['sy_list']).copy()
            syp2[:-2] = np.array(xr_pts['sy_list'])[2:].copy()
            syp2[-2] = np.array(xr_pts['sy_list'])[0].copy()
            syp2[-1] = np.array(xr_pts['sy_list'])[1].copy()

            up2 = np.array(xr_pts['u_list']).copy()
            up2[:-2] = np.array(xr_pts['u_list'])[2:].copy()
            up2[-2] = np.array(xr_pts['u_list'])[0].copy()
            up2[-1] = np.array(xr_pts['u_list'])[1].copy()

            vp2 = np.array(xr_pts['v_list']).copy()
            vp2[:-2] = np.array(xr_pts['v_list'])[2:].copy()
            vp2[-2] = np.array(xr_pts['v_list'])[0].copy()
            vp2[-1] = np.array(xr_pts['v_list'])[1].copy()

            del11 = np.sum( (sy - syp2)**2.0/(2.0*A*A*dt*dt) + (u-up2)**2.0/(4.0*A*A) )
            del22 = np.sum( (sx - sxp2)**2.0/(2.0*A*A*dt*dt) + (v-vp2)**2.0/(4.0*A*A) )
            del12 = np.sum( (sx - sxp2)**2.0/(2.0*A*A*dt*dt) + (u-up2)**2.0/(4.0*A*A) )
            del21 = np.sum( (sy - syp2)**2.0/(2.0*A*A*dt*dt) + (v-vp2)**2.0/(4.0*A*A) )

            delA = np.sum( ((sx- sxp2)**2.0 + (syp2-sy)**2.0)/4.0 )
        
            del11 = dudx*dudx*delA/A**2.0 + del11*(sigx**2.0)
            del22 = dvdy*dvdy*delA/A**2.0 + del22*(sigx**2.0)
            del12 = dudy*dudy*delA/A**2.0 + del12*(sigx**2.0)
            del21 = dvdx*dvdx*delA/A**2.0 + del21*(sigx**2.0)
            delI = del11 + del22
            delvrt = del12 + del21
            term1 = (delI*(dudx - dvdy)**2.0 + (del12+del21)*(dudy + dvdx)**2.0) #this is eps_II^2 * delII^2
            term2 = eps_I**2.0 * delI
            epstot_deltot = term1 + term2

            if eps_II > 0.0:
                delII = (delI*(((dudx - dvdy)/eps_II)**2.0) + (del12+del21)*(((dudy + dvdx)/eps_II)**2.0))
                deltot = delII*((eps_II/eps_tot)**2.0) + delI*(( eps_I/eps_tot)**2.0)
                sig2noise = eps_tot**2.0 / epstot_deltot**0.5
            else:
                delII = 1000.0
                deltot = 1000.0
                sig2noise = 0.0

            #print("%s/%s" % (i,a1), j, len(xr_pts['sx_list']), eps_I,eps_II, vrt)
            shared_results[0,j] = eps_I*60*60*24
            shared_results[1,j] = eps_II*60*60*24
            shared_results[2,j] = vrt*60*60*24
            shared_results[3,j] = A
            shared_results[4,j] = dudx*60*60*24
            shared_results[5,j] = dudy*60*60*24
            shared_results[6,j] = dvdx*60*60*24
            shared_results[7,j] = dvdy*60*60*24
            shared_results[14,j] = delA**0.5
            shared_results[16,j] = (delI**0.5)*60*60*24
            shared_results[16,j] = (delII**0.5)*60*60*24
            shared_results[17,j] = (delvrt**0.5)*60*60*24

def Paralleled_Kernel_smoothing(j, shm_name, shape, Data_in, n, Data_pts, dt, sigx):

    shm = shared_memory.SharedMemory(name=shm_name)
    shared_results = np.ndarray(shape,  buffer=shm.buf)

    proj = ccrs.NorthPolarStereo(central_longitude=0)
    trans = ccrs.PlateCarree()
    if np.isnan(Data_in[0,j]):
        pass
    else:
        day_flag = Data_in[12,j].copy()
        idpair = Data_in[11,j].copy()
        condi = (Data_in[11,:] == idpair) & (Data_in[12,:] == day_flag)

        #------Create the pair xarray
        a = len(Data_in[0,condi])

        # Make dataset with all scaled variables
        div = xr.DataArray(Data_in[0,condi].copy(), dims=['x'], name='div',coords={"x": np.arange(a)})
        shr = xr.DataArray(Data_in[1,condi].copy(), dims=['x'], name='shr',coords={"x": np.arange(a)})
        vrt = xr.DataArray(Data_in[2,condi].copy(), dims=['x'], name='vrt',coords={"x": np.arange(a)})
        A = xr.DataArray(Data_in[3,condi].copy(), dims=['x'], name='A',coords={"x": np.arange(a)})
        dudx = xr.DataArray(Data_in[4,condi].copy()*Data_in[3,condi].copy(), dims=['x'], name='dudx',coords={"x": np.arange(a)})
        dudy = xr.DataArray(Data_in[5,condi].copy()*Data_in[3,condi].copy(), dims=['x'], name='dudy',coords={"x": np.arange(a)})
        dvdx = xr.DataArray(Data_in[6,condi].copy()*Data_in[3,condi].copy(), dims=['x'], name='dvdx',coords={"x": np.arange(a)})
        dvdy = xr.DataArray(Data_in[7,condi].copy()*Data_in[3,condi].copy(), dims=['x'], name='dvdy',coords={"x": np.arange(a)})
        ids1 = xr.DataArray(Data_in[8,condi].copy(), dims=['x'], name='ids1',coords={"x": np.arange(a)})
        ids2 = xr.DataArray(Data_in[9,condi].copy(), dims=['x'], name='ids2',coords={"x": np.arange(a)})
        ids3 = xr.DataArray(Data_in[10,condi].copy(), dims=['x'], name='ids3',coords={"x": np.arange(a)})
        s2n = xr.DataArray(Data_in[13,condi].copy(), dims=['x'], name='s2n',coords={"x": np.arange(a)})
        ref = xr.DataArray(Data_in[18,condi].copy(), dims=['x'], name='ref',coords={"x": np.arange(a)})

        xr_pair = xr.Dataset({'div': div,
                                  'shr':  shr,
                                  'vrt':  vrt,
                                  'dudx': dudx,
                                  'dudy':  dudy,
                                  'dvdx':  dvdx,
                                  'dvdy':  dvdy,
                                  'ids1':  ids1,
                                  'ids2':  ids2,
                                  'ids3':  ids3,
                                  's2n': s2n,
                                  'A': A,
                                  'ref': ref})

        min_index = np.squeeze(np.where(condi))[0]
        max_index = np.squeeze(np.where(condi))[-1]+1
        i = j- min_index

        xr_list = xr_pair.where(xr_pair['ref'].notnull(), drop = True) #Select values that are already post-processed
        indices= np.unique(np.concatenate([np.array(xr_list['ids1']),  xr_list['ids2'], xr_list['ids3']]))

        xr_pair = xr_pair.where(xr_pair['ref'].isnull(), np.nan) #Drop the values if the are already post-processed



        #----------------------------
        # Make a kernel for the smoothing
        #----------------------------
        #get the list of triangles to be averaged
        gen_num = np.array(xr_pair['ids1']).copy()*np.nan
        #n = 0:
        gen_num[i] = 0
        for k in range(1,n+1):
            #----Find neighbouring triangles----

            #Get the id of vertexes
            gen_num_da = xr.DataArray(gen_num.copy(), dims='x', coords={'x': xr_pair.coords['x']})
            xr_list = xr_pair.where(gen_num_da == k-1, drop = True)
            idlist = np.unique(np.concatenate((np.array(xr_list['ids1']),np.array(xr_list['ids2']),np.array(xr_list['ids3']))))
            idlist = idlist[~np.isin(idlist,indices)]
            if len(idlist) == 0:
                break
            if k == 1:
                idlist = idlist[0] #Keeping only ids1 (or ids2,3 if 1 is nan)

            #selct all triangles with one of either vertices
            condi = np.logical_and(np.logical_or(np.array(xr_pair['ids1'].isin(idlist)).astype(int),
                                                   np.logical_or(np.array(xr_pair['ids2'].isin(idlist)).astype(int),
                                                                 np.array(xr_pair['ids3'].isin(idlist)).astype(int))),
                                   np.isnan(gen_num)) #excluding points that are already selected from n<k
            n_tri = len(np.array(np.array(xr_pair['ids1'])[condi]))

            #----Update the neighboor flag with new rows----
            if n_tri == 0:
                break
            gen_num[condi] = k

        gen_num_xr = xr.DataArray(gen_num, dims='x', coords={'x': xr_pair.coords['x']})

        #----------------------------
        # Update the deformation data with the mean of selected triangles
        #----------------------------

        if np.isnan(dt):
            xr_tri = xr_pair.where(~np.isnan(gen_num_xr), drop=True)
            n_tri = len(np.array(xr_tri['A']))
            xr_tri = xr_tri.sum(skipna=True)
            Atot = np.array(xr_tri['A'])
            if Atot> 0.0:
                shared_results[0,j] = np.array(xr_tri['dudx'])/Atot + np.array(xr_tri['dvdy'])/Atot
                shared_results[1,j] = ((  np.array(xr_tri['dudx'])/Atot - np.array(xr_tri['dvdy'])/Atot)**2.0 +
                            (np.array(xr_tri['dudy'])/Atot + np.array(xr_tri['dvdx'])/Atot)**2.0  )**0.5
                shared_results[2,j] = np.array(xr_tri['dvdx'])/Atot - np.array(xr_tri['dudy'])/Atot
                shared_results[3,j] = np.array(xr_tri['A'])
                shared_results[4,j] = np.array(xr_tri['dudx'])/Atot
                shared_results[5,j] = np.array(xr_tri['dudy'])/Atot
                shared_results[6,j] = np.array(xr_tri['dvdx'])/Atot
                shared_results[7,j] = np.array(xr_tri['dvdy'])/Atot
                #print(Data_in[0,j], shared_results[0,j], n_tri)
            else:
                #print(Data_in[:,j])
                #print(gen_num[j])
                shared_results[:,j]  = np.nan

        else:

        #-------------------------------
        # Recompute the deformations if requested
        #-------------------------------

            #print("Data_pts is: ", Data_pts.shape)
            #vertices vectors
            condi_pts = np.logical_and(Data_pts[5,:] == idpair , Data_pts[4,:] == day_flag)
            #print("Data_pts is: ", len(condi_pts))
            start_lats = Data_pts[0,condi_pts]
            start_lons = Data_pts[1,condi_pts]
            end_lats = Data_pts[2,condi_pts]
            end_lons = Data_pts[3,condi_pts]

            xr_tri = xr_pair.where(~np.isnan(gen_num_xr), drop=True)
            n_tri = len(np.array(xr_tri['A']))
            #print(n_tri)
            indices = np.unique(np.concatenate((np.array(xr_tri['ids1']),  np.array(xr_tri['ids2']), np.array(xr_tri['ids3']))))
            #print(n_tri,indices)
            start_lats = start_lats[indices.astype(int)].copy()
            start_lons = start_lons[indices.astype(int)].copy()
            end_lats = end_lats[indices.astype(int)].copy()
            end_lons = end_lons[indices.astype(int)].copy()

            # Change to XY values, with origin on the cloud of pts centroid
            new_coords_start = proj.transform_points(trans, np.array(start_lons).copy(), np.array(start_lats).copy())
            new_coords_end = proj.transform_points(trans, np.array(end_lons).copy(), np.array(end_lats).copy())
            sx_list=new_coords_start[:,0].copy()-(np.sum(new_coords_start[:,0].copy())/len(new_coords_start[:,0]))
            ex_list=new_coords_end[:,0].copy()-(np.sum(new_coords_start[:,0].copy())/len(new_coords_start[:,0]))
            sy_list=new_coords_start[:,1].copy()-(np.sum(new_coords_start[:,1].copy())/len(new_coords_start[:,1]))
            ey_list=new_coords_end[:,1].copy()-(np.sum(new_coords_start[:,1].copy())/len(new_coords_start[:,1]))

            #Sort according to the angle with the centroid
            angles = np.arctan2(sy_list, sx_list)
            sorted_indices = np.argsort(angles)
            a = len(sx_list)

            u_list =  xr.DataArray(((ex_list[sorted_indices] - sx_list[sorted_indices]) / dt).copy(), dims=['x'], name='u_list',coords={"x": np.arange(a)})
            v_list =  xr.DataArray(((ey_list[sorted_indices] - sy_list[sorted_indices]) / dt).copy(), dims=['x'], name='v_list',coords={"x": np.arange(a)})
            sx_list =  xr.DataArray(sx_list[sorted_indices].copy(), dims=['x'], name='sx_list',coords={"x": np.arange(a)})
            sy_list =  xr.DataArray(sy_list[sorted_indices].copy(), dims=['x'], name='sy_list',coords={"x": np.arange(a)})
            ex_list =  xr.DataArray(ex_list[sorted_indices].copy(), dims=['x'], name='ex_list',coords={"x": np.arange(a)})
            ey_list =  xr.DataArray(ey_list[sorted_indices].copy(), dims=['x'], name='ey_list',coords={"x": np.arange(a)})

            xr_pts = xr.Dataset({'sx_list': sx_list,
                             'sy_list': sy_list,
                             'ex_list': ex_list,
                             'ey_list': ey_list,
                             'u_list':  u_list,
                             'v_list':  v_list})

            test = 11
            # Remove concave angles by computing the cross product of segments at each point
            while test > 10:
                test += 1
                ux = np.array(xr_pts['sx_list']).copy()
                vy = np.array(xr_pts['sy_list']).copy()
                uxm = np.array(xr_pts['sx_list']).copy()
                uxm[1:] = np.array(xr_pts['sx_list'])[:-1].copy()
                uxm[0] = np.array(xr_pts['sx_list'])[-1].copy()
                vym = np.array(xr_pts['sy_list']).copy()
                vym[1:] = np.array(xr_pts['sy_list'])[:-1].copy()
                vym[0] = np.array(xr_pts['sy_list'])[-1].copy()
                uxp = np.array(xr_pts['sx_list']).copy().copy()
                uxp[:-1] = np.array(xr_pts['sx_list'])[1:].copy()
                uxp[-1] = np.array(xr_pts['sx_list'])[0].copy()
                vyp = np.array(xr_pts['sy_list']).copy()
                vyp[:-1] = np.array(xr_pts['sy_list'])[1:].copy()
                vyp[-1] = np.array(xr_pts['sy_list'])[0].copy()
         
                u1 = ux - uxm
                u2 = uxp - ux
                v1 = vy - vym
                v2 = vyp-vy
                cross = (u1*v2)-(v1*u2)
                #print(cross)
                if len(cross[cross<0]) > 0:
                    cross = xr.DataArray(cross.copy(), dims=['x'], name='cr',coords={'x': xr_pts.coords['x']})
                    xr_pts = xr_pts.where(cross>0, drop = True)
                else:
                    test = 0

            sx = np.array(xr_pts['sx_list']).copy()
            sy = np.array(xr_pts['sy_list']).copy()

            sxp = np.array(xr_pts['sx_list']).copy()
            sxp[:-1] = np.array(xr_pts['sx_list'])[1:].copy()
            sxp[-1] = np.array(xr_pts['sx_list'])[0].copy()

            syp = np.array(xr_pts['sy_list']).copy()
            syp[:-1] = np.array(xr_pts['sy_list'])[1:].copy()
            syp[-1] = np.array(xr_pts['sy_list'])[0].copy()

            u = np.array(xr_pts['u_list']).copy()
            v = np.array(xr_pts['v_list']).copy()
            um = np.array(xr_pts['u_list']).copy()
            um[1:] = np.array(xr_pts['u_list'])[:-1].copy()
            um[0] = np.array(xr_pts['u_list'])[-1].copy()
            vm = np.array(xr_pts['v_list']).copy()
            vm[1:] = np.array(xr_pts['v_list'])[:-1].copy()
            vm[0] = np.array(xr_pts['v_list'])[-1].copy()
            up = np.array(xr_pts['u_list']).copy().copy()
            up[:-1] = np.array(xr_pts['u_list'])[1:].copy()
            up[-1] = np.array(xr_pts['u_list'])[0].copy()
            vp = np.array(xr_pts['v_list']).copy()
            vp[:-1] = np.array(xr_pts['v_list'])[1:].copy()
            vp[-1] = np.array(xr_pts['v_list'])[0].copy()


            A = np.sum(( sx*syp - sxp*sy)/2.0)
            dudx = np.sum((up+u)*(syp-sy))/(2.0*A) 
            dudy = np.sum((up+u)*(sx-sxp))/(2.0*A)  #Note that sx-sxp = -Delta x here
            dvdx = np.sum((vp+v)*(syp-sy))/(2.0*A) 
            dvdy = np.sum((vp+v)*(sx-sxp))/(2.0*A) #Note that sx-sxp = -Delta x here

            eps_I = dudx + dvdy
            eps_II =  ((dudx - dvdy)**2 + (dudy + dvdx)**2)**0.5
            vrt = dvdx - dudy

            eps_tot = ( eps_I**2 + eps_II**2 )**0.5
            # Compute the propagation errors

            sxp2 = np.array(xr_pts['sx_list']).copy()
            sxp2[:-2] = np.array(xr_pts['sx_list'])[2:].copy()
            sxp2[-2] = np.array(xr_pts['sx_list'])[0].copy()
            sxp2[-1] = np.array(xr_pts['sx_list'])[1].copy()

            syp2 = np.array(xr_pts['sy_list']).copy()
            syp2[:-2] = np.array(xr_pts['sy_list'])[2:].copy()
            syp2[-2] = np.array(xr_pts['sy_list'])[0].copy()
            syp2[-1] = np.array(xr_pts['sy_list'])[1].copy()

            up2 = np.array(xr_pts['u_list']).copy()
            up2[:-2] = np.array(xr_pts['u_list'])[2:].copy()
            up2[-2] = np.array(xr_pts['u_list'])[0].copy()
            up2[-1] = np.array(xr_pts['u_list'])[1].copy()

            vp2 = np.array(xr_pts['v_list']).copy()
            vp2[:-2] = np.array(xr_pts['v_list'])[2:].copy()
            vp2[-2] = np.array(xr_pts['v_list'])[0].copy()
            vp2[-1] = np.array(xr_pts['v_list'])[1].copy()

            del11 = np.sum( (sy - syp2)**2.0/(2.0*A*A*dt*dt) + (u-up2)**2.0/(4.0*A*A) )
            del22 = np.sum( (sx - sxp2)**2.0/(2.0*A*A*dt*dt) + (v-vp2)**2.0/(4.0*A*A) )
            del12 = np.sum( (sx - sxp2)**2.0/(2.0*A*A*dt*dt) + (u-up2)**2.0/(4.0*A*A) )
            del21 = np.sum( (sy - syp2)**2.0/(2.0*A*A*dt*dt) + (v-vp2)**2.0/(4.0*A*A) )

            delA = np.sum( ((sx- sxp2)**2.0 + (syp2-sy)**2.0)/4.0 )
        
            del11 = dudx*dudx*delA/A**2.0 + del11*(sigx**2.0)
            del22 = dvdy*dvdy*delA/A**2.0 + del22*(sigx**2.0)
            del12 = dudy*dudy*delA/A**2.0 + del12*(sigx**2.0)
            del21 = dvdx*dvdx*delA/A**2.0 + del21*(sigx**2.0)
            delI = del11 + del22
            delvrt = del12 + del21
            term1 = (delI*(dudx - dvdy)**2.0 + (del12+del21)*(dudy + dvdx)**2.0) #this is eps_II^2 * delII^2
            term2 = eps_I**2.0 * delI
            epstot_deltot = term1 + term2
            if eps_II > 0.0:
                delII = (delI*(((dudx - dvdy)/eps_II)**2.0) + (del12+del21)*(((dudy + dvdx)/eps_II)**2.0))
                deltot = delII*((eps_II/eps_tot)**2.0) + delI*(( eps_I/eps_tot)**2.0)
                sig2noise = eps_tot**2.0 / epstot_deltot**0.5
            else:
                delII = 1000.0
                deltot = 1000.0
                sig2noise = 0.0

            #print("%s/%s" % (i,a1), j, len(xr_pts['sx_list']), eps_I,eps_II, vrt)
            shared_results[0,j] = eps_I*60*60*24
            shared_results[1,j] = eps_II*60*60*24
            shared_results[2,j] = vrt*60*60*24
            shared_results[3,j] = A
            shared_results[4,j] = dudx*60*60*24
            shared_results[5,j] = dudy*60*60*24
            shared_results[6,j] = dvdx*60*60*24
            shared_results[7,j] = dvdy*60*60*24
            shared_results[14,j] = delA**0.5
            shared_results[16,j] = (delI**0.5)*60*60*24
            shared_results[16,j] = (delII**0.5)*60*60*24
            shared_results[17,j] = (delvrt**0.5)*60*60*24
            #print(j, Data_in[0,j], shared_results[0,j], n_tri)
