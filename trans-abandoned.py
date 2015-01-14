# This file is abandoned 

import numpy as np
from scipy.interpolate import interpn

def _iminterpolate(t1, t2):
    for i in range(t1.shape[3]):
        l = t1[:,:,:, i]
        mgrid = np.mgrid([range(l.shape[0]), range(l.shape[1]), range(l.shape[2])]) # Coordinate Image
        my = mgird[0,:,:,:] 
        mx = mgird[1,:,:,:] 
        mz = mgird[2,:,:,:] 
        x_prime = mx + t[:,:,:,0]
        y_prime = my + t[:,:,:,1]
        z_prime = mz + t[:,:,:,2]
        interpn((mx,my,mz), l, [x_prime, y_prime, z_prime])
    

# Find out where the points are going
def transinterpolate(mgrid, t1, t2):
    my = mgird[0,:,:,:] 
    mx = mgird[1,:,:,:] 
    mz = mgird[2,:,:,:] 

    iminterpolate(mx+t1[:,:,:,0], t2)
    iminterpolate(my+t1[:,:,:,1], t2)
    iminterpolate(mx+t1[:,:,:,2], t2)

     
# Compose two transforms to output deformation fields
def compose(t1, t2):
    # Remove the identity dimension
    t1 = t1.squeeze()
    t2 = t2.squeeze()

    # see if two transform are of the same dimensions
    assert t1.shape == t2.shape

    mgrid = np.mgrid([range(t1.shape[0]), range(t1.shape[1]), range(t1.shape[2])]) # Coordinate Image
    tp = iminterpolate(mgrid, t1, t2)
    tv = tp - mgrid

    # TODO:Zero vectors going outside the image

    return tv


if __name__ == '__main__':
    transa = '/home/siqi/workspace/ContentBasedRetrieval/PyRegival/tests/testdata/4092cMCI-GRAPPA2/results/transformed/322535-278831/SyNQuick/transid1/out1Wrap.nii.gz'
    transb = '/home/siqi/workspace/ContentBasedRetrieval/PyRegival/tests/testdata/4092cMCI-GRAPPA2/results/transformed/278831-241691/SyNQuick/transid0/out1Warp.nii.gz'
    data1 = nib.load(transa).get_data()
    data2 = nib.load(transb).get_data()
    tv = compose(data1, data2)
