# -*- coding: utf-8 -*-
"""
Title: AHTTR (Automated High-Throughput Tomography Reconstruction)
Authors: Elliott Marsden, Dillon S. Watring, Ashley D. Spear
License:
    Software License Agreement (BSD License)
    
    Copyright 2022 Elliott Marsden. All rights reserved.
    Copyright 2022 Dillon S. Watring. All rights reserved.
    Copyright 2022 Ashley D. Spear. All rights reserved.
    
    THE BSD LICENSE
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    1. Redistributions of the source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in the binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
       
    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ''AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND OR ANY THEORY OF LIABILITY, WHETHER IN CONTRAST,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

""" 

import numpy as np
import datetime
import cv2 as cv2
import tifffile as tif
import os
import matplotlib.pyplot as plt
import yaml
import sys
import AHTTR_img_util as imu
import tomopy
import time
from skimage import exposure
from skimage import feature
import shutil
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu, threshold_yen

class AHTTR:
    """
    Automated High-Throughput Tomography Reconstruction:
    Reconstructs raw tomography data and removes duplicates between volluns \n
    Args:
        yamlPath   - Required if recon == True  : Tomo collection variables (str)
        recon      - Optional  : Determines CoR and reconstructs raw tomo data (bool)
        plots      - Optional  : Displays visualizations of reconstruction processes for 3 seconds (bool)
        focusAlg   - Optional  : Algorithm used for CoR focus scoring (str)
        matchHist  - Optional  : matches each images histogram to the first image in the combined folder (bool)
    """
    def __init__(self, yamlPath = "", recon = False, plots = False, focusAlg = "canny", matchHist = False):
        
        self.img_util_path = "C:/Users/Dillon/Documents/AHTTR/AHTTR_img_util/"
        self.dxchange_path = "C:/Users/Dillon/anaconda3/envs/AHTTR/Lib/site-packages/dxchange/"
        self.yamlPath = yamlPath
        self.focusAlg = focusAlg
        self.recon = recon
        self.plots = plots
        self.matchHist = matchHist
        if self.recon == True:
            self.yamlDict = self.readYaml()
    
    def readYaml(self):
        """
        Desc: 
            Reads .yaml file
        
        Retrieves yaml file keys pertaining to an individual vollun

        Returns:
            A dict mapping keys to the corresponding vollun dict containing
            all reconstruction parameters
        
        Raises:
            ValueError: Attempted to read file other than .yaml
        """
        with open(self.yamlPath, "r") as samples:
            try:
                yam = yaml.safe_load(samples)
                return yam
            except yaml.YAMLError as exc:
                raise ValueError("Only .yaml files are allowed")
    
    def printYamlKeys(self):
        """
        Desc: 
            Retrieves and prints yaml file keys 
        """
        if self.recon == True:
            for key in self.yamlDict:
                print(key)
    
    def getYamlKeys(self):
        """
        Desc:
            Retrieves yaml file keys and stores them in a list
        
        Returns:
            list of vollun keys
        
        """
        return [key for key in self.yamlDict]
    
    def reconVollun(self, key):
        """
        Desc: 
            Reconstructs a single vollun 
        
        Args:
            vollun key from the given .yaml file
        
        """
        AHTTR_recon(self.yamlDict[key])
        
    def reconAllVollun(self):
        """
        Desc: Reconstructs all volluns 
        
        Reconstructs all volluns listed in the given .yaml file
        
        """
        for key in self.yamlDict:
            self.reconVollun(key)
        
    def printPredictedDuplicate(self, v1Path, v2Path, overlap, refOverlap):
        """
        Desc: 
            Reconstructs all volluns 
        
        Args:
            v1Path: Path to folder to be screened for duplicates (e.g. Vollun 1)
            v2Path: Path to folder containing reference images (e.g. Vollun 2)
            overlap: Image search depth beginning from the end of the v1Path image stack
            refOverlap: Reference image depth 
        
        This function searches for duplicate images between two vollun stacks.
        The first image stack (v1Path) is screened for duplicates with search depth "overlap"
        using reference images from (v2Path). The matching image paths are printed
        to the console. The user will then have the option to delete the predicted
        duplicates from the two image stacks. If plots == True, th euser can
        visualize the predicted match
    
        Returns:
            Predicted first duplicate image and reference image file paths
            are printed to the console. User will be prompted with an option
            to delete the duplicate images
        
        """
        predict = AHTTR_dup_removal(overlap, refOverlap, '')
        predict.printDuplicatePath(v1Path, v2Path)
        
    def combineAllVolluns(self, overlap, refOverlap, parentPath, savePath):
        """
        Desc: 
            combines vollun images into a combined folder and removes duplicates
            between stacks.
        
        Args:
            overlap: image search depth beginning from the end of the screened image stack
            refOverlap: reference image stack search depth 
            parentPath: path to folder containing vollun folders
            savePath: save path for combined stack folder
        """
        combine = AHTTR_dup_removal(overlap, refOverlap, savePath)
        combine.combineAllStacks(parentPath)
    
    def makeDir(self, path):
        """
        Desc: 
            creates directory
        
        Args:
            path: directory path
        
        """
        try:
            os.makedirs(path)
        except FileExistsError: # directory already exists
            pass
    
    def getPaths(self, parent):
        """
        Desc: 
            creates and returns list of tiff file paths
        
        Args:
            parent: tiff stack parent folder
        
        """
        files = sorted(os.listdir(parent), key=len)
        files = [i for i in files if not i.startswith(".")]
        files = [i for i in files if i.endswith(".tif") or i.endswith(".tiff")]
        return [os.path.join(parent, i).replace("\\", "/") for i in files]
     
    def printProgressBar(self, startTime, iteration, total, prefix = "", suffix = "", decimals = 1, length = 100, fill = "█", printEnd = "\r"):
        """
        Desc:
            Call in a loop to create terminal progress bar
        Args:
            startTime   - Required  : start time.time() (float)
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        
        function sourced and modified from:
            https://stackoverflow.com/a/34325723/19088515
        """
        runTime = int(time.time()-startTime)
        clock = datetime.timedelta(seconds=runTime)
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% {suffix} {clock}", end = printEnd)
        if iteration == total: # Print New Line on Complete
            print()
            
class AHTTR_recon(AHTTR):
    def __init__(self, yamlDict):
        """
        AHTTR raw tomography data reconstruction:
        Predicts and uses the optimal center of rotation to reconstruct raw tomo data \n
        Args:
            yamlDict - dictionary created during parent class init 
        """
        super().__init__()
        self.sampleName = yamlDict["path_output"].split("/")
        self.sampleName = self.sampleName[-3]
        self.vollunName = self.sampleName[-1]
        self.corSteps = [1.0, 0.1]
        self.volluns = list()
        self.CoR = 3.0
        self.canny_sigma = 7
        self.vollun_count = 1
        self.steps = [1.0, 0.1]        # rotation axis (after stitching)  [69,6]
        self.rot_step = 1.0
        try:
            self.stroi = yamlDict["stroi"]
            self.shift = yamlDict["shift"]
            self.psroi = yamlDict["psroi"]# roi for projection (x, y, w, h) [ 26, 76, 1864, 1042 ]
        except KeyError:
            pass
        
        self.Notes = yamlDict["Notes"]
        self.raw_data_filename = yamlDict["path_input"] #User defines where the raw images are
        self.target_filepath = yamlDict["path_output"]
        self.path_input = yamlDict["path_input"]
        self.prefix = yamlDict["prefix"]
        self.digit = yamlDict["digit"]
        #self.post_recon_roi = [285, 160, 750, 960]
        self.post_recon_roi = None
        self.nwhite1 = yamlDict["nwhite1"]
        self.nproj = yamlDict["nproj"]
        self.nwhite2 = yamlDict["nwhite2"]
        self.ndark = yamlDict["ndark"]
        self.theta_start = yamlDict["theta_start"]
        self.theta_step = yamlDict["theta_step"]  #0.125,
        self.clim = yamlDict["clim"]   # optional, default: [0, 65535]
        self.rot_angle = yamlDict["rot_angle"]     # rotation of the projection (optional, default: 0)
        self.focus_check_roi = yamlDict["focus_check_roi"]                  # usually leave it blank
        self.nstitch = yamlDict["nstitch"]
        self.roi = yamlDict["roi"]
        self.scno = yamlDict["scno"][0] #scan number
        self.post_recon_rot = 0.0 # post-reconstruction rotation (in deg, +ccw, -cw)
        self.scno_offset = self.scnoOffset()
        self.reconRawData()
        
    def reconRawData(self):
        """
        Desc: 
            stores raw data image paths and begins CoR selection
        
        Args:
            None
        
        """
        Sample_Raw_Data = self.getPaths(self.raw_data_filename) #raw tomo data
        self.vollun_count = int(len(Sample_Raw_Data)/self.scno_offset)
        self.makeDir(self.target_filepath) # make vollun save path  
        recon_start_time = time.time()
        print(f"Starting to reconstruct Sample: {self.sampleName} Vollun: {self.vollunName}")
        self.corSelect() # determine CoR
        runTime = int(time.time()-recon_start_time)
        clock = datetime.timedelta(seconds=runTime)
        print(f"Sample: {self.sampleName} Vollun: {self.vollunName} reconstructed in: {clock}")
    
    def corSelect(self):
        """
        Desc: 
            predicts the CoR using step sizes of 1.0 and 0.1
            1. Using step size 1.0 and default center of 3.0, 21 images 
            (+/- 10 on either side of the given center) for the top, middle, 
            and bottom slices are reconstructed with varying centers -7.0 -> 13.0
            
            2. The most focused/sharp image is used as the new center
            and another set of 21 images for each slice are generated with a 
            finer step size of 0.1
            
            3. The CoRmost focused image from 2. is used for the full vollun 
            reconstruction
        
        Args:
            None
        
        """
        for step in self.corSteps:
            self.APS_reconstruction("sub", step)
            self.CoR = self.predictCoR(step)
            print(f"Step size: {step}")
            print(f"COR PREDICTION: {self.CoR}")
            
        self.APS_reconstruction("full", step)
    
    def scnoOffset(self):
        """
        Desc: 
            offset added to scno for the next vollun
        """
        offset = self.nwhite1 + self.nproj + self.nwhite2 + self.ndark
        return offset
    
    def CannyFocusScore(self, path, COR_Arr):
        """
        Desc: 
            canny focus scoring algorithm
            
            uses the sum of canny edge pixels as the focus score
        
        Args:
            path: path to the image to be scored
            COR_Arr: array of potential CoR
        
        """
        img = tif.imread(path)
        l_x, l_y = img.shape[0], img.shape[1]
        X, Y = np.ogrid[:l_x, :l_y]
        disk_mask = (X - (l_x) / 2)**2 + (Y - (l_y) / 2)**2 > (l_x / 2.2)**2
        
        p2, p98 = np.percentile(img, (2, 98))
        image = exposure.rescale_intensity(img, in_range=(p2, p98))
        edges2 = feature.canny(image, sigma=self.canny_sigma)
        edges2 = edges2.astype("uint8")
        edges2[disk_mask] = 0
        score = cv2.countNonZero(edges2)
        
        if self.plots == True:
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(image, cmap="gray")
            ax1.title.set_text("contours: "+ str(score) + "  COR: " + str(COR_Arr))
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(edges2, cmap="gray")
            ax2.title.set_text("SCORE: "+ str(score) + \
                "  CoR: " + str(COR_Arr) + " Sigma: " +str(self.canny_sigma))
            plt.show()
            plt.pause(3)
            plt.close()
        
        return score
    
    def sort_key(self, e):
        """
        Desc: 
            key used to sort directory of top, middle, and bottom slice image paths
            
        Args:
            e: image path
            
        Returns:
            sorted array
        
        """
        e = e.split("_")
        e = e[-1]
        e = e.split(".")
        e = e[0]
        e = e.split("-")
        x = e[0] 
        e = "".join(e)
        e = int(e)
        x = int(x)
        array = [x, e]
        
        return array

    def predictCoR(self, step):
        """
        Desc: 
            Generates 63 images for the top, middle, and bottom vollun slices
            and selects the best CoR using the given focus scoring algorithm.
            
        Args:
            step: step size for array of potential CoR
            
        Returns:
            best CoR (float)
        
        """
        import time
        start_time = time.time()
        print()
        print("scoring...")
        
        cur_STEP = step
        cur_COR = self.CoR
        COR_imgs = self.getPaths(self.target_filepath)
        COR_imgs.sort(key = self.sort_key)
        COR_imgs_len = len(COR_imgs)
        sub_imgs = COR_imgs_len/3
        
        COR_Arr = []
        begin = cur_COR - cur_STEP * 10
        end = cur_COR + cur_STEP * 10
        for rotcenter in np.linspace(begin, end, 21):
            COR_Arr.append(round(rotcenter, 1))
        
        startTime = time.time()
        self.printProgressBar(startTime, 0, 63, prefix = f"Image {0}/{63}", suffix ="", length = 50)
        count = 1
        for i in range(3):
            focus_scores = []
            plot_imgs = []
            c_id = 0
            
            for j in range(i*int(sub_imgs), (i+1)*int(sub_imgs)):
                n_path = (os.path.join(self.target_filepath, COR_imgs[j]))
                n_path.replace("\\","/")
                focus = self.CannyFocusScore(n_path, COR_Arr[c_id])
                focus_scores.append(focus)
                c_id += 1
                self.printProgressBar(startTime, count, 63, prefix = f"Image {count}/{63}", suffix ="", length = 50)
                count += 1
                
            focus_scores = focus_scores / np.max(focus_scores)
            if i == 0:
                fscore = focus_scores
            else:
                fscore = np.vstack((fscore,focus_scores))
        rotcen = np.array(COR_Arr)
        print()
        if self.plots == True:
            fig, ax = plt.subplots(3, sharex=True)
            ax[0].plot(rotcen, fscore[0,:],"o-", color="blue")
            ax[0].grid(True, ls="--")
            ax[0].legend(loc=0)
            #ax[0].set_title("image rotation is %2.3f deg" % rot_angle)
            ax[1].plot(rotcen, fscore[1,:],"o-", color="green")
            ax[1].grid(True, ls="--")
            ax[1].legend(loc=0)
            ax[2].plot(rotcen, fscore[2,:],"o-", color="red")
            ax[2].grid(True, ls="--")
            ax[2].legend(loc=0)
            ax[2].set_xlabel("rotation axis (px)")
            fig.suptitle("rotation axis offset", fontsize=17)
            fmax = np.max(fscore)
            fig2, ax2 = plt.subplots()
            ax2.plot(rotcen, fscore[0,:] / fmax, "o-",  color="blue")
            ax2.plot(rotcen, fscore[1,:] / fmax, "o-",  color="green")
            ax2.plot(rotcen, fscore[2,:] / fmax, "o-", color="red")
            ax2.grid(True, ls="--")
            ax2.legend(("top","mid","bot"), loc=0)
            ax2.set_xlabel("center of rotation")
            ax2.set_ylabel("normalized non-zero pixel count")
            
            plt.show()
            plt.pause(3)
            plt.close()
        
        ALL_COR_Array = []
        
        for r in range(fscore.shape[1]):
            ALL_COR_Array.append(np.sum(fscore[:,r]))
        
        max_score = np.max(ALL_COR_Array)
        index_max_score = np.where(ALL_COR_Array == max_score)
        predicted_COR = rotcen[index_max_score]
        return round(predicted_COR[0], 2)
    
    def APS_tomo_par_init(self):
        """
        Desc:
            Initialize parameters for reconstruction from experimental file/log
    
        Args:
            None
    
        Return:
            par: output parameter file
        """
        par = dict()
        par["Notes"] = self.Notes
        par["prefix"] = self.prefix
        par["rot_axis"] = self.CoR
        par["nstitch"] = self.nstitch
        par["nproj"] = self.nproj
        try:
            par["psroi"] = self.psroi
        except AttributeError:
            pass
        
        par["roi"] = self.roi
        par["post_recon_roi"] = self.post_recon_roi
        par["post_recon_rot"] = self.post_recon_rot
        par["focus_check_roi"] = self.focus_check_roi
        # default assignment if not available
        par["path_output"] = self.target_filepath#, cfg["recon_dir/"])
        par["rot_angle"]   = (self.rot_angle, 0.0)
        par["clim"]        = (self.clim, [0, 65535])
        
        # default value if not assigned (TO-DO)
        fname = os.path.join(self.path_input, self.prefix + "_" + str(1).zfill(self.digit) + ".tif") # create file template
        par["input_template"] = fname
    
        #par["roi"] = [cfg["roi_x"][0], cfg["roi_y"][0], cfg["roi_x"][1], cfg["roi_y"][1]]  # [top_left_x, top_left_y, width, height)
        par["theta"] = [float(self.theta_start) , self.theta_start + self.theta_step * (self.nproj -1)]
        par["theta_step"] = self.theta_step
    
        if par["nstitch"] > 1:
            par["stroi"] = np.asarray(self.stroi)
            par["shift"] = np.asarray(self.shift)
        else:
            par["stroi"] = None
            par["shift"] = None
    
        par["ind_w1"] = list() # calculate file index
        par["ind_pj"] = list()
        par["ind_w2"] = list()
        par["ind_dk"] = list()
    
        for i in range(0,self.nstitch):
            print(f"scno: {self.scno}")
            par["ind_w1"].append(list(range(self.scno,
                                            self.scno + self.nwhite1)))
            par["ind_pj"].append(list(range(self.scno + self.nwhite1,
                                            self.scno + self.nwhite1 + self.nproj)))
            par["ind_w2"].append(list(range(self.scno + self.nwhite1 + self.nproj,
                                            self.scno + self.nwhite1 + self.nproj + self.nwhite2)))
            par["ind_dk"].append(list(range(self.scno + self.nwhite1 + self.nproj + self.nwhite2,
                                            self.scno + self.nwhite1 + self.nproj + self.nwhite2 + self.ndark)))
        return par
    
    def APS_reconstruction(self, portion, step):
        """
        Desc: 
            APS tomopy: reconstructs raw tomography data
            Author: Andrew Chuang
            
        Args:
            portion: 'full' or 'sub' reconstruction
            step: if 'sub', step size for array of potential CoR
        
        """
        import time
        start_time = time.time()
        sys.path.append(self.img_util_path)
        sys.path.append(self.dxchange_path)
        import dxchange.reader as dxreader
        import dxchange.writer as dxwriter
        import concurrent.futures
    
        opt = self.APS_tomo_par_init() # initialize parameters
        rec_type = portion # assign value based on input files
        show_rec_img = 0 # show rec img to check rot_center
        clean_up_folder = 1 # clean up destination folder for resconstructed files
        rot_center = opt["rot_axis"]
        outpath    = opt["path_output"]
        fprefix    = self.prefix
        rot_angle  = opt["rot_angle"][0]
        clim       = opt["clim"]
        nstitch    = opt["nstitch"]
        roi        = opt["roi"]
        th_start   = opt["theta"][0]
        th_end     = opt["theta"][1]
        th_step    = opt["theta_step"]
        postrecroi = opt["post_recon_roi"]
        postrecrot = opt["post_recon_rot"]
        focochkroi = opt["focus_check_roi"]
        rc_step = step # rec will be done +-10 steps around rot_center
        medfilter_size = None
        
        c_scaling16 = 65536/(clim[0][1] - clim[0][0] + 1)
        crop_x = (roi[0], roi[0] + roi[2])
        crop_y = (roi[1], roi[1] + roi[3])
        fname = opt["input_template"] # prepare file name & index
        
        if nstitch > 4:
            print("Do not support stitch > 4!!")
        elif nstitch == 3:
            ind_wf1 = opt["ind_w1"][0] # generate index of files
            ind_wf2 = opt["ind_w1"][1]
            ind_wf3 = opt["ind_w1"][2]
            ind_pj1 = opt["ind_pj"][0]
            ind_pj2 = opt["ind_pj"][1]
            ind_pj3 = opt["ind_pj"][2]
            ind_dk1 = opt["ind_dk"][0]
            
            tic = time.time()
            print("Start reading files....")
            wf1 = dxreader.read_tiff_stack(fname, ind=ind_wf1, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            wf2 = dxreader.read_tiff_stack(fname, ind=ind_wf2, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            wf3 = dxreader.read_tiff_stack(fname, ind=ind_wf3, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            pj1 = dxreader.read_tiff_stack(fname, ind=ind_pj1, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            pj2 = dxreader.read_tiff_stack(fname, ind=ind_pj2, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            pj3 = dxreader.read_tiff_stack(fname, ind=ind_pj3, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            dk1 = dxreader.read_tiff_stack(fname, ind=ind_dk1, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            print("Done. %.4f sec" % (time.time() - tic))
        
            projn = list() # normalize image first before stitching
            projn.append(tomopy.normalize(pj1, wf1, dk1))
            projn.append(tomopy.normalize(pj2, wf2, dk1))
            projn.append(tomopy.normalize(pj3, wf3, dk1))
            
            del pj1, pj2, pj3, wf1, wf2, wf3, dk1 # clean-up space
            
            shift = opt["shift"].astype(np.float32)
            stroi = opt["stroi"].astype(np.float32)
            psroi = opt["psroi"]
        
            if rec_type != "full": # determine ROI for post stitching images
                sino = (50, roi[3] - 50 - psroi[2] - psroi[3] + 1, (roi[3] - 100 - - psroi[2] - psroi[3]) // 2)
                #sino = (150, roi[3] - 150 - psroi[2] - psroi[3] + 1, (roi[3] - 300 - - psroi[2] - psroi[3]) // 2)
                #sino = (100, roi[3] - 100 - psroi[2] - psroi[3] + 1, (roi[3] - 200 - psroi[2] - psroi[3]) // 2)
            else:
                sino = None    
            
            tic = time.time()
            print("Start stitching files....")
            proj = imu.stitcher(projn, shift, stroi, axis=0, slc=sino, psroi=psroi)
            print("Done. %.4f sec" % (time.time() - tic))
            del projn # clean-up space
            
        elif nstitch == 2:
            ind_wf1 = opt["ind_w2"][0] # generate index of files
            ind_wf2 = opt["ind_w2"][1]
            ind_pj1 = opt["ind_pj"][0]
            ind_pj2 = opt["ind_pj"][1]
            ind_dk1 = opt["ind_dk"][0]
        
            tic = time.time()
            print("Start reading files....")
            wf1 = dxreader.read_tiff_stack(fname, ind=ind_wf1, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            wf2 = dxreader.read_tiff_stack(fname, ind=ind_wf2, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            pj1 = dxreader.read_tiff_stack(fname, ind=ind_pj1, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            pj2 = dxreader.read_tiff_stack(fname, ind=ind_pj2, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            dk1 = dxreader.read_tiff_stack(fname, ind=ind_dk1, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
            print("Done. %.4f sec" % (time.time() - tic))        
            print("Size of projection is", pj1.shape) # check quality of the measurement
            projn = list() # normalize image first before stitching
            projn.append(tomopy.normalize(pj1, wf1, dk1))
            projn.append(tomopy.normalize(pj2, wf2, dk1))
            del pj1, pj2, wf1, wf2, dk1 # clean-up space
            
            shift = opt["shift"].astype(np.float32)
            stroi = opt["stroi"].astype(np.float32)
            psroi = opt["psroi"]
        
            if rec_type != "full": # determine ROI for post stitching images
                #sino = (50, roi[3] - 50 - psroi[2] - psroi[3] + 1, (roi[3] - 100 - psroi[2] - psroi[3]) // 2)
                sino = (100, roi[3] - 100 - psroi[2] - psroi[3] + 1, (roi[3] - 200 - psroi[2] - psroi[3]) // 2)
            else:
                sino = None    
            
            tic = time.time()
            print("Start stitching files....")
            proj = imu.stitcher(projn, shift, stroi, axis=0, slc=sino, psroi=psroi)
            print("Done. %.4f sec" % (time.time() - tic))
            del projn # clean-up space
            
        else:
            # determine ROI for each projection
            if rec_type != "full":
                sino = (roi[1] + 50, roi[1] + roi[3] - 50 + 1, (roi[3] - 100) // 2)
            else:
                sino = crop_y
            #print(type(scno_start), type(nwhite1))
            ind_white1 = opt["ind_w1"][0]
            ind_proj   = opt["ind_pj"][0]
            ind_white2 = opt["ind_w2"][0]
            ind_dark   = opt["ind_dk"][0]
            tic = time.time()
            #print("Start reading files and rotate image %.3f degree...." % rot_angle)
            #white1 = dxreader.read_tiff_stack(fname, ind=ind_white1, digit=None, slc=(sino, crop_x))#, angle=rot_angle, mblur=medfilter_size)
            proj = dxreader.read_tiff_stack(fname, ind=ind_proj, digit=None, slc=(sino, crop_x), angle=rot_angle, mblur=medfilter_size)
            white2 = dxreader.read_tiff_stack(fname, ind=ind_white2, digit=None, slc=(sino, crop_x), angle=rot_angle, mblur=medfilter_size)
            dark = dxreader.read_tiff_stack(fname, ind=ind_dark, digit=None, slc=(sino, crop_x), angle=rot_angle, mblur=medfilter_size)
            proj[proj==0] = 1
            #print("Done. %.4f sec" % (time.time() - tic))
            
            # check quality of the measurement
            #print("Size of projection is", proj.shape)
        
            # Flat-field correction of raw data.
            tic = time.time()
            #print("Normalization....")
            proj = tomopy.normalize(proj, white2, dark)
            #print("Done. %.4f sec" % (time.time() - tic))
            del white2, dark # clean-up space
        
        theta = tomopy.angles(proj.shape[0], ang1=th_start, ang2=th_end) # Set data collection angles as equally spaced between theta_start ~ theta_end (in degrees.)
        #print(theta)
        # Ring removal.
        tic = time.time()
        #print("Apply Ring removal filter...")
        #proj = tomopy.remove_stripe_fw(proj)      # pretty effective, bg distortion
        proj = tomopy.remove_stripe_ti(proj)      # pretty effective, bg distortion Was using this on last
        #proj = tomopy.remove_stripe_sf(proj)      # less useful, but won"t distort background
        print(proj.shape, proj.dtype, proj.max(), proj.min())
        tic = time.time()
        print("Calculate minus_log of projection...")
        # make sure no negative intensity
        proj[proj<=0] = 0.0001
        proj = tomopy.minus_log(proj)
        #print("Done. %.4f sec" % (time.time() - tic))
        print(proj.shape, proj.dtype, proj.max(), proj.min())
        print("Create & Cleanup destination folder...")
        
        if not os.path.exists(outpath): # create output folder if not exist
            os.makedirs(outpath)
        else:
            if clean_up_folder: # clean folder if requested
                filelist = os.listdir(outpath)
                for files in filelist:
                    os.remove(os.path.join(outpath, files))
        	
        	# debug (save sinogram)
        #dxwriter.write_tiff_stack(np.einsum("jik->ijk", proj), fname= outpath + "sinogram")
        	
        if rec_type != "full":
            import time
            start_time = time.time()
            centers = {
            "0.1": tomopy.find_center_pc(proj[0], proj[1800], tol=0.5),   # for theta step = 0.1
            "0.125": tomopy.find_center_pc(proj[0], proj[1440], tol=0.5),   # for theta step = 0.125 #used this for dillons data
            "0.15": tomopy.find_center_pc(proj[0], proj[1200], tol=0.5),   # for theta step = 0.15
            "0.12": tomopy.find_center_pc(proj[0], proj[900], tol=0.5),   # for theta step = 0.2
            "0.25": tomopy.find_center_pc(proj[0], proj[720], tol=0.5)    # for theta step = 0.25
            }
            #print("Expected rotation center: ", cen - proj.shape[2]/2, th_step)
            cen = centers[str(self.theta_step)]
            
            #cv2.namedWindow("Check Proj", cv2.WINDOW_NORMAL)
            #cv2.selectROI("Check Proj", proj[:,0,:], True)
        
            print("Do 3 slices, ", list(range(sino[0], sino[1], sino[2])), ", to find Rot center")
            print(postrecroi)
            if postrecroi != None:
                recdimx = postrecroi[2]
                recdimy = postrecroi[3]
            else:
                recdimx = abs(proj.shape[2])
                recdimy = abs(proj.shape[2])
        
            if show_rec_img:
                rectop = np.zeros((recdimy, recdimx, 21), np.float32)
                recmid = np.zeros((recdimy, recdimx, 21), np.float32)
                recbot = np.zeros((recdimy, recdimx, 21), np.float32)
        
            if focochkroi != None:
                xs = focochkroi[0]
                xe = focochkroi[0] + focochkroi[2]
                ys = focochkroi[1]
                ye = focochkroi[1] + focochkroi[3]
            else:
                xs = 0
                xe = recdimx
                ys = 0
                ye = recdimy
                
            _, nslice, projw = proj.shape
            focus_score = np.zeros((4, 21))
            i = 0
            count = 3
            tic = time.time()
            print("rotate reconstructed image", postrecrot, "degree")
            print()
            self.printProgressBar(tic, 0, 63, prefix = f"Reconstructing image: {0}/{63}", suffix ="", length = 50)
            for rotcenter in np.linspace(rot_center - rc_step * 10, rot_center + rc_step * 10, 21):
                rec = tomopy.recon(proj, theta, center=proj.shape[2]/2 + rotcenter, algorithm="gridrec")
                #rec = tomopy.recon(proj, theta, center=proj.shape[2]/2 + rotcenter, algorithm="art", num_iter=10)
                rec = rec * 10E7
                # tomopy.misc.corr.circ_mask(arr, axis, ratio=1, val=0.0, ncore=None) # apply cirular mask
                # rec = tomopy.misc.corr.circ_mask(rec, axis=0, ratio=1.0, val=0.0)
                if not (postrecrot == None or postrecrot == 0): # rotate the image (unbounded, i.e image will trim outside original size)
                    (nimg, h, w) = rec.shape[:] # grab the dimensions of the image and then determine the center
                    (cX, cY) = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D((cX, cY), postrecrot, 1) # grab the rotation matrix, then grab the sine and cosine
                    for k in range(0, nimg): # perform the actual rotation and return the image
                        rec[k,:,:] = cv2.warpAffine(rec[k,:,:], M, (w, h))
        
                if postrecroi != None: # crop the reconstructed image
                    rec = rec[:, postrecroi[1]:postrecroi[1]+postrecroi[3], postrecroi[0]:postrecroi[0]+postrecroi[2]]
                self.printProgressBar(tic, count, 63, prefix = f"Reconstructing image: {count}/{63}", suffix ="", length = 50)
                
                count += 3
                dxwriter.write_tiff_stack(rec, fname= outpath+"/"+fprefix)
    
            if show_rec_img: # convert the data to uint16 and normalize to its MIN/MAX or clim
                rectop = cv2.normalize(rectop, None, alpha=-clim[0]*c_scaling16,\
                    beta=(65535-clim[0])*c_scaling16, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
                recmid = cv2.normalize(recmid, None, alpha=-clim[0]*c_scaling16,\
                    beta=(65535-clim[0])*c_scaling16, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
                recbot = cv2.normalize(recbot, None, alpha=-clim[0]*c_scaling16,\
                    beta=(65535-clim[0])*c_scaling16, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        
                window1 = imu.PanZoomWindow(rectop, "top layer")
                window2 = imu.PanZoomWindow(recmid, "middle layer")
                window3 = imu.PanZoomWindow(recbot, "bottom layer")
        
                print("User 'q' or ESC to quite the window.")
                key = 27
                while key != ord("q") and key != 27:  # 27 = escape key
                    key = cv2.waitKey(5)  # User can press "q" or ESC to exit
                cv2.destroyAllWindows()
                     
        else:
            print("Do full volume reconstruction with Rot_center = %f" % rot_center)
            tic = time.time() # Reconstruct object using Gridrec algorithm.
            print("Start reconstruction....")
            recs = tomopy.recon(proj, theta, center=proj.shape[2]/2 + rot_center, algorithm="gridrec")
            print("Done. %.4f sec" % (time.time() - tic))
    
            if not (postrecrot == None or postrecrot == 0): # rotate the image (unbounded, i.e image will trim outside original size)
                (nimg, h, w) = recs.shape[:] # grab the dimensions of the image and then determine the center
                (cX, cY) = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D((cX, cY), postrecrot, 1) # grab the rotation matrix, then grab the sine and cosine
                
                for k in range(0, nimg): # perform the actual rotation and return the image
                    recs[k,:,:] = cv2.warpAffine(recs[k,:,:], M, (w, h), cv2.INTER_NEAREST_EXACT)
        
            #recs = tomopy.circ_mask(recs, axis=0, ratio=0.95) # Mask each reconstructed slice with a circle.
            recs = tomopy.misc.corr.circ_mask(recs, axis=0, ratio=1.0, val=0.0)
    
            if postrecroi != None: # crop the reconstructed image
                recs = recs[:, postrecroi[1]:postrecroi[1] + postrecroi[3], postrecroi[0]:postrecroi[0] + postrecroi[2]]
        
            del proj # clean up space
            
            recs = (recs*10E7).astype(np.int32) # for very large dataset, this step could crash the reconstruction due to memory issue.
            tic = time.time() # Write data as stack of TIFFs.
            print("Start saving files....")
            dxwriter.write_tiff_stack(recs, fname= outpath + "/" + fprefix)
            print("Done. %.4f sec" % (time.time() - tic))

class AHTTR_dup_removal(AHTTR):
    """
        AHTTR vollun duplicate/overlap removal:
        Finds matching images between vollun stacks and combines stacks without duplicates \n
        
        Args:
            defined during parent class init 
        """
    def __init__(self, overlap, refOverlap, savePath):
        super().__init__()
        self.overlap = overlap
        self.refOverlap = refOverlap
        self.savePath = savePath
        
    def ImgProc(self, image_path, ref_image_path):
        """
        Desc: 
            applies high-pass filter and calculates intersection over union
            for between images
            
        Args:
            image_path: potential duplicate image path
            ref_image_path: reference image path
        
        Returns:
            IOU score
        
        """
        img = tif.imread(image_path)
        ref_image = tif.imread(ref_image_path)
        img = match_histograms(img, ref_image)
        img = exposure.rescale_intensity(img, out_range=(0.0, 1.0))
        ref = exposure.rescale_intensity(ref_image, out_range=(0.0, 1.0))
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_ref = cv2.dft(np.float32(ref), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        dft_shift_ref = np.fft.fftshift(dft_ref)
        
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        
        # circular mask added to center of transformation
        # removing low frequency components of the image
        # radius ""rad"" can be tuned to your image set
        # good starting range = 10-20 
        mask = np.ones((rows, cols, 2), np.uint8)
        rad = 20
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= rad*rad
        mask[mask_area] = 0
        
       
        fshift = dft_shift * mask # high pass filter is applied
        fshift_ref = dft_shift_ref * mask
        f_ishift = np.fft.ifftshift(fshift)
        f_ishift_ref = np.fft.ifftshift(fshift_ref)
       
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        img_back_ref = cv2.idft(f_ishift_ref)
        img_back_ref = cv2.magnitude(img_back_ref[:, :, 0], img_back_ref[:, :, 1])
        
        l_x, l_y = img_back.shape[0], img_back.shape[1]
        X, Y = np.ogrid[:l_x, :l_y]
        outer_disk_mask = (X - (l_x) / 2)**2 + (Y - (l_y) / 2)**2 > (l_x / 2.2)**2 # mask is applied to remove any artifact from beam window
        
        img_back[outer_disk_mask] = 0
        img_back_ref[outer_disk_mask] = 0
        img_back = match_histograms(img_back, img_back_ref)
        
        thresh_ref = threshold_yen(img_back_ref)
        thresh = threshold_yen(img_back) 
        img_back_ref[img_back_ref<=thresh_ref]=0
        img_back_ref[img_back_ref>thresh_ref]=1
        img_back[img_back<=thresh]=0
        img_back[img_back>thresh]=1
        
        intersection = img_back*img_back_ref
        union = img_back + img_back_ref
        IOU = intersection.sum()/float(union.sum()) # iou returned as similarity score
        
        # optional visualization 
        # can be used to tune fourier mask
        # commented out below
        """
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(img_back, cmap="gray")
        ax1.title.set_text("img_back")
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(img_back_ref, cmap="gray")
        ax2.title.set_text("img_back_ref")
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(intersection, cmap="gray")
        ax3.title.set_text("intersection")
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(union, cmap="gray")
        ax4.title.set_text("union")
        plt.show()
        """
        
        return IOU
    
    def sorted_alphanumeric(self, data):
        """
        Desc: 
            sorts image paths with alphanumeric key 
            
        Args:
            image directory
        
        function sourced from:
        https://stackoverflow.com/a/48030307/19088515
        """
        import re
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split("([0-9]+)", key) ] 
        return sorted(data, key=alphanum_key)
    
    def printDuplicatePath(self, v1Path, v2Path):
        """
        Desc: 
            Prints paths to matching images and prompts user to delete all
            duplicate images
            
        Args:
            v1Path: potential duplicate image path
            v2Path: reference image path
        
        """
        sampleName = v1Path.split("/") 
        vol1Name = v1Path.split("/")
        vol1Name = vol1Name[-1]
        vol2Name = v2Path.split("/")
        vol2Name = vol2Name[-1]
        sec2 =  self.sorted_alphanumeric(os.listdir(v2Path)) # reference image directory
        sec1 = self.sorted_alphanumeric(os.listdir(v1Path)) # screened image directory
        bestDuplicate, bestRefIndex = self.findMatch(v1Path, v2Path)
        index_best_duplicate = sec1.index(bestDuplicate)
        matchPath = os.path.join(v1Path, bestDuplicate).replace("\\","/")
        refPath = os.path.join(v2Path, sec2[bestRefIndex]).replace("\\","/")
        print()
        print(f"Vollun {vol1Name} Image: \n {matchPath}")
        print("Is most similar to:")
        print(f"Vollun {vol2Name} Image: \n {refPath}")
        print()
        print("All images between these two paths are likely duplicates")
        print()
        print("Would you like to delete the duplicate images? [Y] or [N]:")
        string = str(input())
        if string == 'Y':
            matchStart = os.path.join(v1Path, sec1[index_best_duplicate]).replace("\\","/")
            matchEnd = os.path.join(v1Path, sec1[-1]).replace("\\","/")
            if bestRefIndex > 0:
                refStart = os.path.join(v2Path, sec2[0]).replace("\\","/")
                refEnd = os.path.join(v2Path, sec2[bestRefIndex]).replace("\\","/")
                print(f'are you sure you want to delete images: \n {matchStart} \nthrough \n {matchEnd}')
                print('and')
                print(f'{refStart} \nthrough \n {refEnd}\n [Y] or [N]:?')
            else:
                print(f'are you sure you want to delete images: \n {matchStart} \nthrough \n {matchEnd} \n [Y] or [N]:?')
            secondChance = str(input())
            if secondChance == 'Y':
                print('deleting duplicates...')
                for f in range(index_best_duplicate, len(sec1)):
                    culprit = (os.path.join(v1Path, sec1[f]).replace("\\","/")) 
                    os.remove(culprit)
                
                if bestRefIndex > 0:
                    for f in range(bestRefIndex):
                        culprit = (os.path.join(v2Path, sec2[f]).replace("\\","/")) 
                        os.remove(culprit)

    def combineAllStacks(self, parent):
        """
        Desc: 
            combines all samples vollun stacks
            
        Args:
            parent: parent folder of all samples
        
        """
        samples = os.listdir(parent)
        for s in samples:
            path = os.path.join(parent, s).replace("\\","/")
            self.combineSingleStack(path)
            
    def combineSingleStack(self, parent):
        """
        Desc: 
            combines all volluns in sample stack
            
        Args:
            parent: parent folder of all vollun stacks
        
        """
        sampleName = parent.split("/")
        sampleName = sampleName[-1]
        s_off = 0
        S_Sec = os.listdir(parent)
        S_Sec_end = len(S_Sec) # vollun count
        comb_path = (os.path.join(parent, self.savePath).replace("\\","/")) 
        self.makeDir(comb_path) # create combined folder path
        for x in range(S_Sec_end-1): # screens for duplicates between neighbor sections/volluns
            refPath = os.path.join(parent, S_Sec[x+1]).replace("\\","/")
            sec2 = os.listdir(refPath) # reference section
            sec1 = os.listdir((os.path.join(parent, S_Sec[x]).replace("\\","/"))) # section to be screened for duplicates
                         
            for w in range(len(sec1)): # copies each image in sample section folder to combined folder   
                OG = (os.path.join(parent, S_Sec[x], sec1[w])).replace("\\","/")
                shutil.copy(OG, comb_path) 
                RENAMED = (os.path.join(comb_path, S_Sec[x] + sec1[w])).replace("\\","/")
                comb_OG = (os.path.join(comb_path, sec1[w])).replace("\\","/")
                os.rename(comb_OG, RENAMED)
            
            bestDuplicate, bestRefIndex = self.findMatch(comb_path, refPath)
            comb_sec = os.listdir(comb_path, ) # combined folder file locations
            comb_end = len(comb_sec)
            
            index_best_duplicate =  comb_sec.index(bestDuplicate)
            s_off = bestRefIndex
            
            for f in range(index_best_duplicate, comb_end):
                culprit = (os.path.join(parent, self.savePath, comb_sec[f]).replace("\\","/")) 
                os.remove(culprit) # remove duplicate
                        
            if x == (S_Sec_end-2): # final stack doesnt have duplicates 
                end_sec = os.listdir((os.path.join(parent, S_Sec[S_Sec_end-1]).replace("\\","/")))                   
                for w in range(s_off, len(end_sec)):                      
                   OG = (os.path.join(os.path.join(parent, S_Sec[S_Sec_end-1], end_sec[w])).replace("\\","/"))
                   shutil.copy(OG, comb_path) 
                   RENAMED = (os.path.join(comb_path, S_Sec[S_Sec_end-1]+end_sec[w])).replace("\\","/")
                   comb_OG = (os.path.join(comb_path, end_sec[w])).replace("\\","/")
                   os.rename(comb_OG, RENAMED)
                
                comb_sec = os.listdir(comb_path)
                comb_end = len(comb_sec)
                hist_ref = tif.imread((os.path.join(parent, self.savePath, comb_sec[0]).replace("\\","/")))
                
                if self.matchHist == True: # combined stack images are histogram matched to first image in stack                      
                    for q in range(1, comb_end):                         
                        match_path = (os.path.join(parent, self.savePath, comb_sec[q]).replace("\\","/"))
                        comb_stack_im = tif.imread(match_path)
                        comb_stack_im = match_histograms(comb_stack_im, hist_ref)
                        tif.imsave(match_path, comb_stack_im.astype("float32"))
    
    def findMatch(self, v1Path, v2Path):
        """
        Desc: 
            finds best match between duplicate and reference images
            
        Args:
            v1Path: parent folder of stack to be screened for duplicates
            v2Path: parent folder of reference stack
        
        """
        sampleName = v1Path.split("/") 
        vol1Name = v1Path.split("/")
        vol1Name = vol1Name[-1]
        vol2Name = v2Path.split("/")
        vol2Name = vol2Name[-1]
        sec2 =  self.sorted_alphanumeric(os.listdir(v2Path)) # reference image directory
        sec1 = self.sorted_alphanumeric(os.listdir(v1Path)) # section screened against reference for duplicates
        v1_end = len(sec1) # determines the number of images in the combined folder
        temp_images_dup = [] # file name storage for o_lap images
        temp_images_ref = [] # file name storage for reference images
        best_matches = [] # file names for best match from each reference image
        best_scores = [] # corresponding iou scores for best_matches images
        count = len(sec1)-self.overlap
        bar_length = self.refOverlap*self.overlap
        print("Comparing images...")
        tic = time.time()
        main_count = 1
        self.printProgressBar(tic, 0, bar_length, prefix = f"Comparing {vol2Name}: {1} {vol1Name}: {count}", suffix ="", length = 50)
        for r in range(self.refOverlap):
            sim_list = list() # image similarity (iou) scores  
            img2_path = (os.path.join(v2Path, sec2[r]).replace("\\","/"))
            temp_images_ref.append(sec2[r])
            temp_images =[]
            decline = 0
            count = len(sec1)-self.overlap+1 #starting image for progress bar
            for i in range(v1_end-self.overlap, v1_end):                           
                img1_path = (os.path.join(v1Path, sec1[i]).replace("\\","/"))
                temp_images.append(sec1[i])
                score = self.ImgProc(img1_path, img2_path)                    
                if i == v1_end-self.overlap:
                    Temp = score                        
                if score < Temp:
                    decline += 1                            
                if score > Temp:
                    decline = 0
                if decline == 25: # iou score does not improve 
                    main_count = (r+1)*self.overlap
                    self.printProgressBar(tic, main_count, bar_length, prefix = f"Comparing {vol2Name}: {r+1} {vol1Name}: {count}", suffix ="", length = 50)
                    break
                sim_list.append(score)
                Temp = max(sim_list)
                self.printProgressBar(tic, main_count, bar_length, prefix = f"Comparing {vol2Name}: {r+1} {vol1Name}: {count}", suffix ="", length = 50)
                count += 1 # count used for loading bar
                main_count += 1 # count used for loading bar
                
            temp_images_dup.append(temp_images)
            max_sim = max(sim_list) 
            index_max_sim = sim_list.index(max_sim)
            best_scores.append(max_sim) 
            best_matches.append(temp_images[index_max_sim])
            
            if self.plots == True: # Optional plot of similarity scores   
                o_lap_index = []
                for ov in range(len(sim_list)):
                   o_lap_index.append(ov)
                
                plt.plot(o_lap_index, sim_list)
                plt.title("IOU scores")
                plt.ylabel("score")
                plt.show()
                plt.pause(3)
                plt.close()
    
        overall_max = max(best_scores) 
        index_max_sim = best_scores.index(overall_max)
        best_duplicate = best_matches[index_max_sim]
        index_best_duplicate =  sec1.index(best_duplicate)
        s_off = index_max_sim
        
        if self.plots == True: # Optional comparison of reference and predicted match images
                match = tif.imread(os.path.join(v1Path, best_duplicate).replace("\\","/"))
                reference = tif.imread((os.path.join(v1Path, sec2[s_off]).replace("\\","/")))             
                self.comparePlot(match, reference)
        
        return best_duplicate, index_max_sim
    
    def comparePlot(self, match, reference):
        """
        Desc: 
            plots comparison between predicted match and reference image
            
        Args:
            match: match image
            reference: reference image
        
        """
        img = match_histograms(match, reference)                 
        p2, p98 = np.percentile(img, (2, 98))
        r2, r98 = np.percentile(reference, (2, 98))
        match = exposure.rescale_intensity(match, in_range=(p2, p98))
        reference = exposure.rescale_intensity(reference, in_range=(r2, r98)) 
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(reference, cmap="gray")
        ax1.title.set_text("Reference Image")
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(match, cmap="gray")
        ax2.title.set_text("Predicted Match")
        plt.show()
        plt.pause(5)
        plt.close()
                    
        