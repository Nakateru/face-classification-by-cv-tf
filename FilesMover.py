"""
Move images from img_path to ok_path if which names are same as in faces_path

"""
import glob
import re
import shutil

img_path = './faces/mikana/'
faces_path = './faces/mikana_f/'
ok_path = './faces/mikana_ok/'

ff_list = []
oo_list = []

f_list = glob.glob(faces_path + '*.jpg')
o_list = glob.glob(img_path + '*.jpg')

print(len(f_list), 'in faces_path')
print(len(o_list), 'in img_path')

for i in f_list:
    ff_list.append(re.split(r'[\\]', i)[-1])

for i in o_list:
    oo_list.append(re.split(r'[\\]', i)[-1])

# print(ff_list)
# print(oo_list)

for i in oo_list:
    if i in ff_list:
        shutil.move(img_path + i, ok_path)
        print('Moved', i)

print('Done')
