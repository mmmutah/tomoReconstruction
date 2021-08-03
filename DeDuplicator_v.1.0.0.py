#!/usr/bin/env python
# coding: utf-8

"""
Title: DeDuplicator
Authors: Elliott Marsden, Dillon S. Watring, Ashley D. Spear
License:
    Software License Agreement (BSD License)
    
    Copyright 2021 Elliott Marsden. All rights reserved.
    Copyright 2021 Dillon S. Watring. All rights reserved.
    Copyright 2021 Ashley D. Spear. All rights reserved.
    
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
#-----------------------------------------------------------------------------
import shutil
import os
import cv2
import tifffile as tif
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu, threshold_yen

#-----------------------------------------------------------------------------
# USER INPUT

# TARGET PARENT FOLDER, SHOULD CONTAIN UNCOMBINED SAMPLE FOLDERS
fname = "D:/APS Tomo" 
# DUPLICATE SEARCH DEPTH
o_lap = 90   
# REFERENCE IMAGE DEPTH
ro_lap = 5 
# FOLDER CREATED TO STORE COMBINED FOLDERS, WILL BE PLACED IN TARGET DIRECTORY
c_path = "Z_For_test-Combined_Sample_Stacks_" 

#-----------------------------------------------------------------------------
# FUNCTIONS

# WHAT DOES THIS FUNCTION DO
def ImgProc(image_path, ref_image_path):
    
    img = tif.imread(image_path)
    ref_image = tif.imread(ref_image_path)
    
    img = match_histograms(img, ref_image)
    
    img = exposure.rescale_intensity(img, out_range=(0.0, 1.0))
    ref = exposure.rescale_intensity(ref_image, out_range=(0.0, 1.0))
    
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_ref = cv2.dft(np.float32(ref), flags=cv2.DFT_COMPLEX_OUTPUT)

    # REARRANGES A FOURIER TRANSFORM X BY SHIFTING THE ZERO-FREQUENCY
    # COMPONENT TO THE CENTER OF THE ARRAY
    
    dft_shift = np.fft.fftshift(dft)
    dft_shift_ref = np.fft.fftshift(dft_ref)
    
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    
    # CIRCULAR MASK ADDED TO CENTER OF TRANSFORMATION
    # REMOVING LOW FREQUENCY COMPONENTS OF THE IMAGE
    # RADIUS ''rad'' CAN BE TUNED TO YOUR IMAGE SET
    # GOOD STARTING RANGE = 10-20 
    mask = np.ones((rows, cols, 2), np.uint8)
    rad = 20
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= rad*rad
    mask[mask_area] = 0
    
    # HIGH PASS FILTER IS APPLIED
    fshift = dft_shift * mask
    fshift_ref = dft_shift_ref * mask
    #fshift_mask_mag = 
    #200 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    #fshift_mask_mag_ref = 
    #200 * np.log(cv2.magnitude(fshift_ref[:, :, 0], fshift_ref[:, :, 1]))
    
    f_ishift = np.fft.ifftshift(fshift)
    f_ishift_ref = np.fft.ifftshift(fshift_ref)
   
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    img_back_ref = cv2.idft(f_ishift_ref)
    img_back_ref = cv2.magnitude(img_back_ref[:, :, 0], img_back_ref[:, :, 1])
    
    
    # MASK IS APPLIED TO REMOVE ANY ARTIFACT FROM BEAM WINDOW
    l_x, l_y = img_back.shape[0], img_back.shape[1]
    X, Y = np.ogrid[:l_x, :l_y]
    outer_disk_mask = (X - (l_x) / 2)**2 + (Y - (l_y) / 2)**2 > (l_x / 2.2)**2
    
    img_back[outer_disk_mask] = 0
    img_back_ref[outer_disk_mask] = 0
    
    # MATCH HISTOGRAMS
    img_back = match_histograms(img_back, img_back_ref)
    
    # DETERMINING THRESHOLDS
    # REFERENCE IMAGE THRESHOLD
    thresh_ref = threshold_yen(img_back_ref)
    # POTENTIAL DUPLICATE IMAGE THRESHOLD
    thresh = threshold_yen(img_back) 
    
    
    # CONVERTING IMAGES TO BINARY FORMAT
    img_back_ref[img_back_ref<=thresh_ref]=0
    img_back_ref[img_back_ref>thresh_ref]=1
    
    img_back[img_back<=thresh]=0
    img_back[img_back>thresh]=1
    
    
    # CALCULATING INTERSECTION OVER UNION SCORE
    # LOGICAL AND
    intersection = img_back*img_back_ref
    # LOGICAL OR
    union = img_back + img_back_ref

    # IOU RETURNED AS SIMILARITY SCORE
    IOU = intersection.sum()/float(union.sum())
    
    # OPTIONAL VISUALIZATION 
    # CAN BE USED TO TUNE FOURIER MASK
    # COMMENTED OUT BELOW
    """
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(img_back, cmap='gray')
    ax1.title.set_text('img_back')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(img_back_ref, cmap='gray')
    ax2.title.set_text('img_back_ref')
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(intersection, cmap='gray')
    ax3.title.set_text('intersection')
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(union, cmap='gray')
    ax4.title.set_text('union')
    plt.show()
    """
    
    return IOU

#-----------------------------------------------------------------------------
import time
start_time = time.time()

if __name__ == "__main__":
    # DIRECTORY OF ALL SAMPLE PARENT FILES IS CREATED
    S_Par = os.listdir(fname) 
    # NUMBER OF SAMPLE PARENT FILES IS DETERMINED AND USED FOR LOOPING
    S_Par_end = len(S_Par) 
    
    try: 
        # NEW FOLDER IS ADDED TO DIRECTORY
        os.mkdir(os.path.join(fname, c_path).replace("\\","/"))
    except OSError as error: 
        print(error) 
    
    # LOOP THROUGH ALL SAMPLE PARENT FOLDERS           
    for z in range(S_Par_end):
        s_off = 0
        # LIST OF SAMPLE SECTIONS GENERATED    
        S_Sec = os.listdir((os.path.join(fname, S_Par[z]).replace("\\","/")))
        # NUMBER OF SAMPLE SECTIONS DETERMINED AND USED FOR LOOP BOUNDARY
        S_Sec_end = len(S_Sec)
            
        # PATH FOR COMBINED SAMPLE FOLDER, EACH SAMPLE WILL HAVE UNIQUE FOLDER                                        
        comb_path = (os.path.join(fname, c_path, S_Par[z]).replace("\\","/")) 
        try: 
    	    # SAMPLE COMBINED FOLDER GENERATED
            os.mkdir(comb_path.replace("\\","/"))
        except OSError as error: 
            print(error) 
         
        # SCREENS FOR DUPLICATES BETWEEN NEIGHBOR SECTIONS/VOLLUNS
        for x in range(S_Sec_end-1):
            # REFERENCE IMAGE DIRECTORY
            sec2 = os.listdir((os.path.join(fname, \
                S_Par[z],S_Sec[x+1]).replace("\\","/")))
            # SAMPLE SECTION TO BE SCREENED FOR DUPLICATES
            sec1 = os.listdir((os.path.join(fname, \
                S_Par[z],S_Sec[x]).replace("\\","/")))
            # COPIES EACH IMAGE IN SAMPLE SECTION FOLDER TO COMBINED FOLDER                
            for w in range(s_off, len(sec1)):
                OG = (os.path.join(fname, S_Par[z], \
                    S_Sec[x], sec1[w])).replace("\\","/")
                shutil.copy(OG, comb_path) 
                RENAMED = (os.path.join(comb_path, S_Sec[x] + \
                    sec1[w])).replace("\\","/")
                comb_OG = (os.path.join(comb_path, \
                    sec1[w])).replace("\\","/")
                os.rename(comb_OG, RENAMED)
                    
            # LIST OF FILE LOCATIONS FOR EACH IMAGE IN THE SAMPLE COMBINED
            comb_sec = os.listdir(comb_path) 
            # DETERMINES THE NUMBER OF IMAGES IN THE COMBINED FOLDER
            comb_end = len(comb_sec)
            # FILE NAME STORAGE FOR o_lap IMAGES
            temp_images_dup = [] 
            # FILE NAME STORAGE FOR REFERENCE IMAGES
            temp_images_ref = [] 
            # FILE NAMES FOR BEST MATCH FROM EACH REFERENCE IMAGE
            best_matches = [] 
            # CORRESPONDING IOU SCORES FOR BEST_MATCHES IMAGES
            best_scores = []
           
            for r in range(ro_lap):
                # STORAGE FOR IMAGE SIMILARITY (IOU) SCORES        
                sim_list = list()
                img2_path = (os.path.join(fname, S_Par[z], \
                    S_Sec[x+1], sec2[r]).replace("\\","/"))
                temp_images_ref.append(sec2[r])
                temp_images =[]
                decline = 0
                # o_lap NUMBER OF IMAGES IS SCREENED FOR DUPLICATES
                for i in range(comb_end-o_lap, comb_end):                         
                    img1_path = (os.path.join(fname, c_path, \
                        S_Par[z], \
                        comb_sec[i]).replace("\\","/"))
                    temp_images.append(comb_sec[i])
                    score = ImgProc(img1_path, img2_path)                    
                    if i == comb_end-o_lap:
                        Temp = score                        
                    if score < Temp:
                        decline += 1                            
                    if score > Temp:
                        decline = 0
                    #DECLINE USED TO EXIT FOR LOOP IF IOU SCORE DOES NOT IMPROVE                            
                    if decline == 10:
                        break
                    # SIMILARITY SCORE IS DETERMINED AND STORED IN SIM_LIST                        
                    sim_list.append(score)
                    Temp = max(sim_list)
                    
                temp_images_dup.append(temp_images)      
                # MAXIMUM VALUE OF THE SIM_LIST IS DETERMINED
                max_sim = max(sim_list) 
                # MOST SIMILAR IMAGE LOCATION IS DETERMINED
                index_max_sim = sim_list.index(max_sim)
                best_scores.append(max_sim) 
                best_matches.append(temp_images[index_max_sim])
                   
                # Optional plot of similarity scores
                """
                o_lap_index = []
                for ov in range(len(sim_list)):
                   o_lap_index.append(ov)
                       
                plt.plot(o_lap_index, sim_list)
                plt.title('Scores')
                plt.ylabel('score')
                plt.show()
                """
            # MAXIMUM VALUE OF THE SIM_LIST IS DETERMINED 
            overall_max = max(best_scores) 
            index_max_sim = best_scores.index(overall_max)
            best_duplicate = best_matches[index_max_sim]
            index_best_duplicate =  comb_sec.index(best_duplicate)
            s_off = index_max_sim
            
            # Optional comparison of reference and predicted match images
            #"""
            match = tif.imread(os.path.join(fname, c_path, \
                S_Par[z], best_duplicate).replace("\\","/"))
            reference = tif.imread((os.path.join(fname, S_Par[z], S_Sec[x+1],\
                sec2[s_off]).replace("\\","/")))             
            img = match_histograms(match, reference)                 
            p2, p98 = np.percentile(img, (2, 98))
            r2, r98 = np.percentile(reference, (2, 98))
            match = exposure.rescale_intensity(match, in_range=(p2, p98))
            reference = exposure.rescale_intensity(reference, \
                in_range=(r2, r98)) 
            
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(reference, cmap='gray')
            ax1.title.set_text('Reference Image')
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(match, cmap='gray')
            ax2.title.set_text('Predicted Match')
            plt.show()
            #"""
                
            for f in range(index_best_duplicate, comb_end):
                # IMAGE DETERMINED TO BE THE FIRST DUPLICATE IS REMOVED
                culprit = (os.path.join(fname, c_path, \
                    S_Par[z], comb_sec[f]).replace("\\","/"))
                os.remove(culprit) 
            # FINAL VOLLUN TRANSFER TO COMBINED STACK             
            if x == (S_Sec_end-2):
                end_sec = os.listdir((os.path.join(fname, S_Par[z],S_Sec[\
                    S_Sec_end-1]).replace("\\","/")))                   
                for w in range(s_off, len(end_sec)):                      
                   OG = (os.path.join(os.path.join(fname, S_Par[z],S_Sec[\
                       S_Sec_end-1], \
                       end_sec[w])).replace("\\","/"))
                   shutil.copy(OG, comb_path) 
                   RENAMED = (os.path.join(comb_path,\
                       S_Sec[S_Sec_end-1]+\
                       end_sec[w])).replace("\\","/")
                   comb_OG = (os.path.join(comb_path,\
                    end_sec[w])).replace("\\","/")
                   os.rename(comb_OG, RENAMED)
                # COMBINED STACK HISTOGRAM MATCHING
                # LIST OF FILE LOCATIONS FOR EACH IMAGE
                comb_sec = os.listdir(comb_path)
                comb_end = len(comb_sec)
                hist_ref = tif.imread((os.path.join(fname, c_path, S_Par[z],\
                    comb_sec[0]).replace("\\","/")))
                # ALL IMAGES IN STACK ARE HISTOGRAM MATCHED TO FIRST IMAGE IN STACK                    
                for q in range(1, comb_end):                         
                    match_path = (os.path.join(fname, c_path,S_Par[z],\
                        comb_sec[q]).replace("\\","/"))
                    comb_stack_im = tif.imread(match_path)
                    comb_stack_im = match_histograms(comb_stack_im, hist_ref)
                    tif.imsave(match_path, comb_stack_im.astype('float32'))
    
                print("--- %s seconds ---" % (time.time() - start_time))
    
   