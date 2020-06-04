import numpy as np

def create_k_strat_folds(normal_patients, abnormal_patients, k = 5):
    """
    Creates k stratified folds by creating 5 folds of each array.
    
        - normal_patients: array with IDs of normal classified patients
        - abnormal_patients: array with IDs of abnormal classified patients
        - k: number of folds to create
        
    If np.split cannot create k folds, it will throw an error
    """
    normal_folds = np.split(normal_patients, k)
    abnormal_folds = np.split(abnormal_patients, k)
    folds = np.concatenate((normal_folds, abnormal_folds), axis = 1)
        
    return folds

def get_train_and_val(folds, val_fold = 4):
    """
    From the folds created in create_k_strat_folds, 
    this function returns a train_set with indices that are in training folds
    and a val_set with indices from the validation fold.
    
        - folds: 2D-array with folds created in create_k_strat_folds
        - val_fold: integer that indicates which fold is the validation fold
        
    Keep in mind that val_fold cannot be larger than k in create_k_strat_folds!!
    """
    val_set = folds[val_fold]
    train_set= folds[np.arange(len(folds))!=val_fold]
    
    return train_set.flatten(), val_set