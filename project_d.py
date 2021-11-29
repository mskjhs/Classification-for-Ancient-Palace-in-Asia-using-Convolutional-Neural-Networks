import os
import shutil

# change image name for Korea
file_path = r'/home/mskjhs/PycharmProjects/untitled2/project/clean/Korea'
file_names = os.listdir(file_path)
i = 0
for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i) + 'korea.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1

# change image name for China
file_path = r'/home/mskjhs/PycharmProjects/untitled2/project/clean/China'
file_names = os.listdir(file_path)
i = 0
for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i) + 'china.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1

# change image name for Japan
file_path = r'/home/mskjhs/PycharmProjects/untitled2/project/clean/Japan'
file_names = os.listdir(file_path)
i = 0
for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i) + 'japan.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1

# # New naming for each dataset
# Korea_data = '/home/mskjhs/PycharmProjects/untitled2/project/clean/Korea'
# China_data = '/home/mskjhs/PycharmProjects/untitled2/project/clean/China'
# Japan_data = '/home/mskjhs/PycharmProjects/untitled2/project/clean/Japan'
#
# base = './home/mskjhs/PycharmProjects/untitled2/projects_real/'
# if not os.path.isdir(base):
#     os.makedirs(base)
#
# # make each dir
# train_dir = os.path.join(base, 'train')
# os.mkdir(train_dir)
# validation_dir = os.path.join(base, 'validation')
# os.mkdir(validation_dir)
# test_dir = os.path.join(base, 'test')
# os.mkdir(test_dir)
#
# # make Africa and India
# train_Korea = os.path.join(train_dir, 'Korea')
# os.mkdir(train_Korea)
# train_China = os.path.join(train_dir, 'China')
# os.mkdir(train_China)
# train_Japan = os.path.join(train_dir, 'Japan')
# os.mkdir(train_Japan)
#
# validation_Korea = os.path.join(validation_dir, 'Korea')
# os.mkdir(validation_Korea)
# validation_China = os.path.join(validation_dir, 'China')
# os.mkdir(validation_China)
# validation_Japan = os.path.join(validation_dir, 'Japan')
# os.mkdir(validation_Japan)
#
# test_Korea = os.path.join(test_dir, 'Korea')
# os.mkdir(test_Korea)
# test_China = os.path.join(test_dir, 'China')
# os.mkdir(test_China)
# test_Japan = os.path.join(test_dir, 'Japan')
# os.mkdir(test_Japan)
#
# # Copy
# fnames = ['{}Korea.jpg'.format(i) for i in range(480)]
# for fname in fnames:
#     src = os.path.join(Korea_data, fname)
#     dst = os.path.join(train_Korea, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['{}Korea.jpg'.format(i) for i in range(480, 640)]
# for fname in fnames:
#     src = os.path.join(Korea_data, fname)
#     dst = os.path.join(validation_Korea, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['{}Korea.jpg'.format(i) for i in range(640, 800)]
# for fname in fnames:
#     src = os.path.join(Korea_data, fname)
#     dst = os.path.join(test_Korea, fname)
#     shutil.copyfile(src, dst)
#
#
# fnames = ['{}China.jpg'.format(i) for i in range(480)]
# for fname in fnames:
#     src = os.path.join(China_data, fname)
#     dst = os.path.join(train_China, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['{}China.jpg'.format(i) for i in range(480, 640)]
# for fname in fnames:
#     src = os.path.join(China_data, fname)
#     dst = os.path.join(validation_China, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['{}China.jpg'.format(i) for i in range(640, 800)]
# for fname in fnames:
#     src = os.path.join(China_data, fname)
#     dst = os.path.join(test_China, fname)
#     shutil.copyfile(src, dst)
#
#
# fnames = ['{}Japan.jpg'.format(i) for i in range(480)]
# for fname in fnames:
#     src = os.path.join(Japan_data, fname)
#     dst = os.path.join(train_Japan, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['{}Japan.jpg'.format(i) for i in range(480, 640)]
# for fname in fnames:
#     src = os.path.join(Japan_data, fname)
#     dst = os.path.join(validation_Japan, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['{}Japan.jpg'.format(i) for i in range(640, 800)]
# for fname in fnames:
#     src = os.path.join(Japan_data, fname)
#     dst = os.path.join(test_Japan, fname)
#     shutil.copyfile(src, dst)