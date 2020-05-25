import torch
import matplotlib.pyplot as plt


def plot_slices(patient,range_of_slices,DATA_DIR):

    if type(range_of_slices) == tuple:
        assert range_of_slices[0] < range_of_slices[1]
        size_of_range = range_of_slices[1] - range_of_slices[0]
        nr_of_rows = size_of_range // 4 + 1
        nr_of_cols = 4 # max 4 plots per row 

        subplots_to_delete = (4 - size_of_range%4) - 1 ## subplots of last row that will turn out empty..
        
        fig,axs = plt.subplots(nr_of_rows,nr_of_cols, figsize=(16,16))
        for index,slice_number in enumerate(range(range_of_slices[0],range_of_slices[1]+1)): #inclusive..
            X = torch.load(DATA_DIR + '/' + str(patient) + '_' + str(slice_number) + '.pt')
            if nr_of_rows > 1:
                axes = axs[ index//4 , index%4 ]
            else:
                axes = axs[index]
            axes.imshow(X.data.numpy()[2,:,:])
            axes.set_title("Patient {} slice number {}".format(patient,slice_number))
            
        # make sure we don't have these empty white subplots
        for empty_subplot in range(3, 4 - subplots_to_delete - 1, -1):
            fig.delaxes(axs[nr_of_rows-1][empty_subplot])
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.65,
                    wspace=0.35)
    else:
        slice_number =range_of_slices
        X=torch.load(DATA_DIR + '/' + str(patient) + '_' + str(slice_number) + '.pt')

        plt.figure()
        plt.imshow(X.data.numpy()[2,:,:]) 
        plt.title("Patient {} slice number {}".format(patient,slice_number))
    plt.show()