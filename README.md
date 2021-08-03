# **Automated Tomography Reconstruction**

## **Project Description:**
This automation project is intended to relieve the immense time commitment required to reconstruct raw tomography data into usable image data. 

Processing tomography data has been a historically tedious process that uses time otherwise spent on more critical aspects of research projects. The repetive and time-consuming nature of tomography reconstruction make it an exellent candidate for automation. Removing the need for manual reconstruction enables researchers to increase their sample batch sizes without incurring an overwhelming amount of manual reconstruction hours. 

This project is intended to be a comprehensive approach to tomography reconstruction. Because parties interested in this project may have image data in varying stages of reconstruction, each stage of the pipeline is provided as a stand-alone program in addition to the full reconstruction pipeline. The implementation of the stand-alone programs and full pipeline are described in their respective folders.

## **The Reconstruction Process:**

### **1. Center of Rotation Selection:** 
> Perfect sample alignment within the beam window is difficult. The center of the sample often varies slightly from the beam window center. 
This program tests a range of rotation centers and reconstructs the raw data using the center that produces the clearest images. 
### **2. Vollun Stack Combination and Duplicate Removal:**
> Because measured sample lengths often exceed the dimensions of the beam window, samples are initially imaged and reconstructed in sections (volluns). 
To prevent data loss, the sections overlap one another. While this ensures a complete representation of the sample, it creates duplicate images in the overlap regions.
After the initital reconstruction completed in step 1, this program removes any duplicate images and combines the vollun stacks into a single stack of images.  
### **3. Combined Stack Binary Segmentation:**
> Optional, but is useful for 3D sample visualization. This program uses machine learning to complete a binary segmentation of the sample cross-section images.
Tomograpy produces noisy images, and traditional thresholding techniques are unable to differentiate between sample and background. 
The results obtained using machine learning are excellent and highly recommended for anyone interested in binary segmentation. 
The entire process including training data generation is described in the segmentation folder. 

## **Development:**
All programs were developed using Python 3.8 within the Spyder IDE.

[Spyder download](https://www.spyder-ide.org/)

## **Program Use/Download:**
It is recommended that all programs are run using the Spyder IDE. To use the programs, simply download the desired folder and follow the program specific README file. 
