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
from math import floor

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


    def Kernel_smoothing(self, xr_in = None, xr_ref = None, n = None):

        if n is None:
            n = 1
        if xr_in is None:
            xr_in = self.dataset
        if xr_ref is not None:
            Data_ref = np.array(xr_ref['div']).copy()
        else:
            Data_ref = self.Data.div.copy()*np.nan
#                    indices = self.list_indices(xr_pair = xr_pair_ref)
#                    xr_pair = xr_pair.where(~xr_pair['ids1'].isin(indices), drop = True)
        Nblocks = len(np.unique(self.Data.idpair))
        nb = 0
        print(len(self.Data.div))
        for nblock in np.unique(self.Data.idpair):
            print("Doing pairs %s/%s" % (nb, Nblocks))
            nb = nb+1
            condi = (self.Data.idpair== nblock)

            Data_shared = np.array([self.Data.div[condi].copy(),
                           self.Data.shr[condi].copy(),
                           self.Data.vrt[condi].copy(),
                           self.Data.A[condi].copy(),
                           self.Data.dudx[condi].copy(),
                           self.Data.dudy[condi].copy(),
                           self.Data.dvdx[condi].copy(),
                           self.Data.dvdy[condi].copy(),
                           self.Data.ids1[condi].copy(),
                           self.Data.ids2[condi].copy(),
                           self.Data.ids3[condi].copy(),
                           self.Data.idpair[condi].copy(),
                           self.Data.day_flag[condi].copy(),
                           self.Data.s2n[condi].copy(),
                           self.Data.A[condi].copy(),
                           self.Data.dudx[condi].copy(),
                           self.Data.dudy[condi].copy(),
                           self.Data.dvdx[condi].copy(),
                           self.Data.dvdy[condi].copy(),
                           Data_ref[condi].copy()])
            print(Data_shared.shape)
            Data_in = Data_shared.copy()
            shm = shared_memory.SharedMemory(create=True, size=Data_shared.nbytes)
            shared_arr = np.ndarray(Data_shared.shape, dtype=Data_shared.dtype, buffer=shm.buf)
            np.copyto(shared_arr, Data_shared)
            print(Data_shared.shape)

            args = [(j, shm.name, shared_arr.shape, Data_in,n) for j in range(0, len(shared_arr[0,:]))]
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(Paralleled_Kernel_smoothing, *arg) for arg in args]
#                for _ in tqdm(as_completed(futures), total=len(futures)):
#                    pass  # or collect results
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

    def Anisotropic_smoothing(self, xr_in = None, n = None):

        if xr_in is None:
            xr_in = self.dataset

        if self.xr_filtered is None:
            print("ERROR: Need s2n filtered data to use the anisotropic smoothing")
            return
        if n is not None:
            self.n = n
        a = len(self.Data.div)
        Nblocks = len(np.unique(self.Data.idpair))
        nb = 0
        for nblock in np.unique(self.Data.idpair):
            print("Doing pairs %s/%s" % (nb, Nblocks))
            nb = nb+1
            condi = (self.Data.idpair== nblock)

            Data_shared = np.array([self.Data.div[condi].copy(),
                           self.Data.shr[condi].copy(),
                           self.Data.vrt[condi].copy(),
                           self.Data.A[condi].copy(),
                           self.Data.dudx[condi].copy(),
                           self.Data.dudy[condi].copy(),
                           self.Data.dvdx[condi].copy(),
                           self.Data.dvdy[condi].copy(),
                           self.Data.ids1[condi].copy(),
                           self.Data.ids2[condi].copy(),
                           self.Data.ids3[condi].copy(),
                           self.Data.idpair[condi].copy(),
                           self.Data.day_flag[condi].copy(),
                           self.Data.s2n[condi].copy(),
                           self.Data.A[condi].copy(),
                           self.Data.dudx[condi].copy(),
                           self.Data.dudy[condi].copy(),
                           self.Data.dvdx[condi].copy(),
                           self.Data.dvdy[condi].copy()])
            Data_in = Data_shared.copy()
                  
            shm = shared_memory.SharedMemory(create=True, size=Data_shared.nbytes)
            shared_arr = np.ndarray(Data_shared.shape, dtype=Data_shared.dtype, buffer=shm.buf)
            np.copyto(shared_arr, Data_shared)
            args = [(j, n, shm.name, shared_arr.shape, Data_in) for j in range(0, len(shared_arr[0,:]))]

            with ProcessPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(Paralleled_Anisotropic_smoothing, *arg) for arg in args]
#                for _ in tqdm(as_completed(futures), total=len(futures)):
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




def Paralleled_Anisotropic_smoothing(j, n, shm_name, shape, Data_in):

    shm = shared_memory.SharedMemory(name=shm_name)
    shared_results = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)

    if np.isnan(Data_in[15,j]):
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
        A =   xr.DataArray(Data_in[14,condi].copy(), dims=['x'], name='A',coords={"x": np.arange(a)})
        dudx = xr.DataArray(Data_in[15,condi].copy()*Data_in[14,condi].copy(), dims=['x'], name='dudx',coords={"x": np.arange(a)})
        dudy = xr.DataArray(Data_in[16,condi].copy()*Data_in[14,condi].copy(), dims=['x'], name='dudy',coords={"x": np.arange(a)})
        dvdx = xr.DataArray(Data_in[17,condi].copy()*Data_in[14,condi].copy(), dims=['x'], name='dvdx',coords={"x": np.arange(a)})
        dvdy = xr.DataArray(Data_in[18,condi].copy()*Data_in[14,condi].copy(), dims=['x'], name='dvdy',coords={"x": np.arange(a)})
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
        gen_num[i] = 0
        for k in range(1,n+1):
            #----Find neighbouring triangles----

            #Get the id of vertexes
            gen_num_da = xr.DataArray(gen_num.copy(), dims='x', coords={'x': xr_pair.coords['x']})
            xr_list = xr_pair.where(gen_num_da == k-1, drop = True)
            idlist = np.unique(np.concatenate([np.array(xr_list['ids1']),  xr_list['ids2'], xr_list['ids3']]))
            #Get triangles that are neighbour to this generation
            condi = np.logical_and(( np.array(xr_pair['ids1'].isin(idlist)).astype(int)+
            np.array(xr_pair['ids2'].isin(idlist)).astype(int)+
            np.array(xr_pair['ids3'].isin(idlist)).astype(int))==2, np.array(xr_pair['s2n'])>1.3)
            n_tri = len(np.array(np.array(xr_pair['ids1'])[condi]))
            #----Update the neighboor flag with new rows----
            gen_num[np.logical_and(condi,np.isnan(gen_num))] = k 

            if n_tri == 0:
                break



        #print(shared_results[4,j], shared_results[15,j])
        #----------------------------
        # Update the deformation data with the mean of selected triangles
        #----------------------------

        gen_num_xr = xr.DataArray(gen_num, dims='x', coords={'x': xr_pair.coords['x']})
        xr_tri = xr_pair.where(~np.isnan(gen_num_xr), drop=True)
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
            #print("A: ",shared_results[14,j],shared_results[3,j])
            #print("dudx: ",shared_results[15,j],np.array(xr_tri['dudx']),shared_results[4,j])
        else:
            #print("skipped! ", k)
            shared_results[0:8,j]  = np.nan
        #print(shared_results[0,j], shared_results[14,j])


def Paralleled_Kernel_smoothing(j, shm_name, shape, Data_in,n):

    shm = shared_memory.SharedMemory(name=shm_name)
    shared_results = np.ndarray(shape,  buffer=shm.buf)

    if np.isnan(Data_in[15,j]):
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
        A = xr.DataArray(Data_in[14,condi].copy(), dims=['x'], name='A',coords={"x": np.arange(a)})
        dudx = xr.DataArray(Data_in[15,condi].copy()*Data_in[14,condi].copy(), dims=['x'], name='dudx',coords={"x": np.arange(a)})
        dudy = xr.DataArray(Data_in[16,condi].copy()*Data_in[14,condi].copy(), dims=['x'], name='dudy',coords={"x": np.arange(a)})
        dvdx = xr.DataArray(Data_in[17,condi].copy()*Data_in[14,condi].copy(), dims=['x'], name='dvdx',coords={"x": np.arange(a)})
        dvdy = xr.DataArray(Data_in[18,condi].copy()*Data_in[14,condi].copy(), dims=['x'], name='dvdy',coords={"x": np.arange(a)})
        ids1 = xr.DataArray(Data_in[8,condi].copy(), dims=['x'], name='ids1',coords={"x": np.arange(a)})
        ids2 = xr.DataArray(Data_in[9,condi].copy(), dims=['x'], name='ids2',coords={"x": np.arange(a)})
        ids3 = xr.DataArray(Data_in[10,condi].copy(), dims=['x'], name='ids3',coords={"x": np.arange(a)})
        s2n = xr.DataArray(Data_in[13,condi].copy(), dims=['x'], name='s2n',coords={"x": np.arange(a)})
        ref = xr.DataArray(Data_in[19,condi].copy(), dims=['x'], name='ref',coords={"x": np.arange(a)})

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

        #The following was to drop high s2n values, but it is wrong...
        #indices = np.concatenate([np.array(xr_list['ids1']),  xr_list['ids2'], xr_list['ids3']])
        #xr_pair = xr_pair.where(~xr_pair['ids1'].isin(indices), drop = True)
        #xr_pair = xr_pair.where(~xr_pair['ids2'].isin(indices), drop = True)
        #xr_pair = xr_pair.where(~xr_pair['ids3'].isin(indices), drop = True)


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

        #----------------------------
        # Update the deformation data with the mean of selected triangles
        #----------------------------

        gen_num_xr = xr.DataArray(gen_num, dims='x', coords={'x': xr_pair.coords['x']})
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
        else:
            print(Data_in[:,j])
            print(gen_num[j])
            print(xr_tri.values)
            shared_results[:,j]  = np.nan






