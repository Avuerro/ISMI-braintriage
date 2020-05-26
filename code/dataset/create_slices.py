import torch
import numpy as np
import SimpleITK as sitk
from tqdm.notebook import tqdm
import csv
import os

def generate_slice_data(IN_DIR,DATA_DIR):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    patients = []
     
    for klass in os.listdir(IN_DIR):
        for patient in tqdm(os.listdir(os.path.join(IN_DIR, klass)), desc="Patients"):
            
            ## Check if patient has not been processed yet
            if patient not in patients:
                ## Paths to the image files
                t1_path = os.path.join(IN_DIR, klass, patient,'T1.mha')
                t2_path = os.path.join(IN_DIR, klass, patient,'T2.mha')
                t2_flair_path = os.path.join(IN_DIR, klass, patient,'T2-FLAIR.mha')

                ## Load all images as numpy arrays
                t1_array = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
                t2_array = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
                t2_flair_array = sitk.GetArrayFromImage(sitk.ReadImage(t2_flair_path))
                
                n_slices = t1_array.shape[0]
                
                for slice_number in range(n_slices):

                    with open(DATA_DIR+'/labels_slices.csv', 'a') as csvfile:      
                        w = csv.writer(csvfile, delimiter=',')
                        ## Write 0 if "normal", 1 if "abnormal"
                        w.writerow([patient, slice_number, klass == "abnormal"])

                    t1_slice = t1_array[slice_number,:,:]
                    t2_slice = t2_array[slice_number,:,:]
                    t2_flair_slice = t2_flair_array[slice_number,:,:]
                    comb_data = np.array([t1_slice, t2_slice, t2_flair_slice])

                    ## Save as torch tensor for quick loading during training
                    torch.save(torch.from_numpy(comb_data.astype('float32')), os.path.join(DATA_DIR, f"{patient}_{slice_number}.pt"))
            else:
                continue
                
            patients.append(patient)