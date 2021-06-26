# for checking the input directories do not contain any corrupt images by 
# opening every file and counting errors and deleting the offending image

import matplotlib.image as mpimg
import os

print("checking files")
path = "./images/"

errors = 0

for impath in [i for i in os.listdir(path) if i.endswith(".png")]:
    try:
        im = mpimg.imread(path+impath)[:,:,:3]
    except RuntimeError:
        print(f'detect error {impath}')
        errors += 1
        continue
print("all images checked")
del im
assert(errors==0)
