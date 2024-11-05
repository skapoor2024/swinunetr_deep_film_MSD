"""
purpose of this file is to help create the dataset list require
for the msd based files to be used with the test function in 
msd_test
"""

import os
msd_fold = './10_Decathlon'
file_path = './dataset/dataset_list/msd_test.txt'
with open(file_path, 'w') as txt_file:

    for fold in os.listdir(msd_fold):
        gg = fold[4:6]
        if gg in ['03','06','07','08','09','10']:
            ts_path = os.path.join(msd_fold,fold,'imagesTs')
            ts_path_2 = os.path.join('10_Decathlon',fold,'imagesTs')
            for file in os.listdir(ts_path):
                if not file.startswith('.') and file.endswith('.nii.gz'):
                    if  file in ['liver_141.nii.gz','liver_156.nii.gz','liver_160.nii.gz','liver_161.nii.gz','liver_162.nii.gz','liver_164.nii.gz','liver_167.nii.gz','liver_182.nii.gz','liver_189.nii.gz','liver_190.nii.gz','hepaticvessel_247.nii.gz']:
                        continue
                    path_to_add = os.path.join(ts_path_2,file)+'\n'
                    txt_file.write(path_to_add)