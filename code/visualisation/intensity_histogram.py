import os
import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt

##normalize mri image based on
#https://github.com/jcreinhold/intensity-normalization/blob/master/intensity_normalization/normalize/zscore.py

def zscore_normalize(img):
    """
        Normalization of image based on mean and std.
    """
    mean = img.mean()
    std = img.std()
    normalized_image = (img - mean) / std
    return normalized_image



def get_acquisition_histogram(in_dir,klass,slices):

    """
        this function will plot the histogram bins for each aquisition 
        for a certain slice FOR ALL PATIENTS

        it loops over the slices, sums the intensities and afterwards divides 
        each intensity by the number of patients * the nr of slices
        this way you will have the average intensity of each pixel
    """
    count_t1 = []
    count_t2 = [] 
    count_t2_flair = [] 
    
    patients_dir = os.path.join(in_dir, klass)
    patients = os.listdir(patients_dir)
    
    if type(slices) == tuple:
        slices = range(slices[0], slices[1])
    if type(slices) == int:
        slices = [slices]
    for index,patient in enumerate(patients):
        patient = str(patient)
        t1_path = os.path.join(in_dir, klass, patient,'T1.mha')
        t2_path = os.path.join(in_dir, klass, patient,'T2.mha')
        t2_flair_path = os.path.join(in_dir, klass, patient,'T2-FLAIR.mha')

        ## Load all images as numpy arrays
        t1_array = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
        t2_array = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
        t2_flair_array = sitk.GetArrayFromImage(sitk.ReadImage(t2_flair_path))
        
        for slice_number in slices:
            ## obtain the slices
            t1_slice = t1_array[slice_number,:,:]
            t2_slice = t2_array[slice_number,:,:]
            t2_flair_slice = t2_flair_array[slice_number,:,:]


            ## normalize the images
            t1_slice_normalized = zscore_normalize(t1_slice)
            t2_slice_normalized = zscore_normalize(t2_slice)
            t2_flair_slice_normalized = zscore_normalize(t2_flair_slice)

            if index == 0 :
                count_t1 = t1_slice_normalized
                count_t2 = t2_slice_normalized
                count_t2_flair = t2_flair_slice_normalized
            else:
                count_t1 += t1_slice_normalized
                count_t2 += t2_slice_normalized
                count_t2_flair += t2_flair_slice_normalized  
        

    count_t1 = count_t1 / (len(patients) * len(slices))
    count_t2 = count_t2 / (len(patients) * len(slices))
    count_t2_flair = count_t2_flair / (len(patients) *len(slices))
    fig,ax = plt.subplots(1,3, sharey=True, figsize=(14,14))

    ax[0].hist(count_t1,range=(0,count_t1.max()))
    ax[0].set_title("Average pixel intensities for T1 Acquistion \n of all {} patients and {} slices".format(klass, str(len(slices))))
    ax[1].hist(count_t2,range=(0,count_t2.max()))
    ax[1].set_title("Average pixel intensities for T2 Acquistion \n of all {} patients and {} slices".format(klass, str(len(slices))))
    ax[2].hist(count_t2_flair,range=(0,count_t2_flair.max()))
    ax[2].set_title("Average pixel intensities for T2 Flair Acquistion \n of all {} patients and {} slices".format(klass, str(len(slices))))

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.65,
                    wspace=0.35)
    fig.suptitle("Overview of Average Pixel Intensities \n of the three aquisitions for {} patients ".format(klass), y=1.06)
    fig.tight_layout()

