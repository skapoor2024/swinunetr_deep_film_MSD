import os

amos_test_img = './09_AMOS/amos22/imagesTs'
amos_file_path = './dataset/dataset_list/amostest_test.txt'

with open(amos_file_path,'w') as txt_file:

    for file in os.listdir(amos_test_img):

        if not file.startswith('.') and file.endswith('.nii.gz'):

            img_path = os.path.join(amos_test_img[2:],file)

            path_to_write = img_path+'\n'
            txt_file.write(path_to_write)