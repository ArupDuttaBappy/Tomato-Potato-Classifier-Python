import os

i = 0
base_path = "dataset_potato/"
for filename in os.listdir(base_path):
    file_dest = "potato" + "." + str(i) + ".jpg"
    my_source = base_path + filename
    file_dest = base_path + file_dest
    os.rename(my_source, file_dest)
    i += 1
