import os
def gen(dir,txt):
    f = open(txt,'w+')
    files = os.listdir(dir)
    for file in files:
        if file.endswith(".png"):
            print(file)
            f.writelines(file[:-4] + '\n')
    f.close

gen('datasets/kitti/testing/image_2',"datasets/kitti/testing/ImageSets/test.txt")