import numpy as np
import scipy
from scipy import signal,ndimage 
import matplotlib.pyplot as plt

def gaussian_2d(sigma_mm, voxel_size):
    """
    Paramaters
    ----------
    Input:
    sigma_mm: 
    voxel_size: voxel_size
    
    Output:
    kernel: kernel
    x : matrix of x coordinates of the filter
    y : matrix of y coordinates of the filter
    """
    bound = sigma_mm*3
    x,y = np.ogrid[-bound : bound : voxel_size[0], -bound : bound : voxel_size[1]]
    
    kernel = 1/(2*np.pi*sigma_mm**2) * np.exp(-(x**2+y**2)/(2*sigma_mm**2))   
    
    return kernel, x, y 

def laplacian_of_gaussian(g):
    gx, gy = np.gradient(g)
    gxx = np.gradient(gx)[0]
    gyy = np.gradient(gy)[1]
    LoG = gxx + gyy
    return LoG,gxx,gyy


def get_brain(brain_slice):
 
    gaussian_blob = gaussian_2d(5, [0.5,0.5])[0]
    brain_slice_blurred = scipy.signal.convolve(brain_slice, gaussian_blob, method="fft", mode="same") 
    rescaled_brain_slice_blurred =(((brain_slice_blurred - brain_slice_blurred.min()) * (1/(brain_slice_blurred.max() - brain_slice_blurred.min()))) * 255).astype('uint8')
    variable_threshold = rescaled_brain_slice_blurred.max() * 0.45
    rescaled_brain_slice_blurred[rescaled_brain_slice_blurred >= variable_threshold ] = 255
    rescaled_brain_slice_blurred[rescaled_brain_slice_blurred < 200 ] = 0

    return rescaled_brain_slice_blurred



def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)





def bright_blobs_plotter(image,slices,seed_points=None):
    voxel_size = [0.5, 0.5]
    sigmas = [5,8,12,50] # We choose 5, 8 and 12 because trachea is 10 to 25mm wide 
                     # and 50 to weaken the effect of large irrelevant areas
    if type(slices) == tuple:
        slices = range(slices[0], slices[1])
    if type(slices) == int:
        slices = [slices]
    for slice_idx in slices:
        ct_slice = image[slice_idx,:,:]
        ct_slice_conv = np.zeros(ct_slice.shape)
        # Compute indices of image that are part of the body
        body_idcs = get_brain(ct_slice)

        # Loop over all sigma values and create average convolved image
        for sigma in sigmas:
            gaussian_blob = gaussian_2d(sigma,voxel_size)[0]
            LoG = laplacian_of_gaussian(gaussian_blob)[0]
            # Use method="same" to avoid size conflict error
            ct_slice_conv += scipy.signal.convolve(ct_slice, LoG, method="fft", mode="same")
        ct_slice_conv_avg = ct_slice_conv/float(len(sigmas))
        # Normalize the convolved slice
        ct_slice_conv_avg = (ct_slice_conv_avg - ct_slice_conv_avg.flatten().min())/(ct_slice_conv_avg.flatten().max() - ct_slice_conv_avg.flatten().min())

        # Show original slice
        plt.subplot(1,2,1); plt.imshow(ct_slice, cmap='gray')
        fig = plt.gcf()
        ax = fig.gca()
        ax.set_title("Original slice")

        # Find all indices that are above threshold and part of body
        threshold = 0.35
        max_idcs_y, max_idcs_x = np.where(ct_slice_conv_avg < threshold)
        max_idcs = list(zip(max_idcs_x,max_idcs_y))
        max_idcs_filtered = [idx for idx in max_idcs if body_idcs[idx[0],idx[1]]]

        # Find closest (or first) seed point
        if len(max_idcs_filtered) != 0:
            # First iteration: choose first seedpoint
            if seed_points is None:
                seed_points = [max_idcs_filtered[0]+(slice_idx,)]
                cur_closest_idx = max_idcs_filtered[0]
            # Other iterations: seek prediction closest to previous seedpoint
            else: # Get X,Y coordinates with [:2]
                cur_closest_idx = max_idcs_filtered[0] 
                cur_min_dist = euclidean_dist(seed_points[-1][:2], max_idcs_filtered[0])
                for idx in max_idcs:
                    if euclidean_dist(seed_points[-1][:2], idx) < cur_min_dist:
                        cur_min_dist = euclidean_dist(seed_points[-1][:2], idx)
                        cur_closest_idx = idx
                seed_points.append(cur_closest_idx+(slice_idx,))

        # Plot convolved image and convolved image with pink dots that signify possible abnormality locations
        plt.subplot(1,2,2); plt.imshow(ct_slice_conv_avg, cmap='gray')
        fig = plt.gcf()
        ax = fig.gca()
        ax.set_title("Convolved slice with blue dots as brightspots \n and red point as seed point")
        for idx in max_idcs_filtered:
            ax.add_artist(plt.Circle(idx,5,color="blue", fill = False, linewidth=2))
        # Draw seed point red
        if len(max_idcs_filtered) != 0:
            ax.add_artist(plt.Circle(cur_closest_idx,5,color="red"))
        plt.show()

    print("Seed points are ", seed_points)