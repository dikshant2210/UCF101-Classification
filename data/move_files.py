import os
import shutil


PATH_TO_DATA = 'UCF-101-Inception-Features/'
PATH_TO_TRAIN = '../data/train'
PATH_TO_TEST = '../data/test'
PATH_TO_VALIDATION = '../data/validation'


def read_files(category):
    files = os.listdir(os.path.join(PATH_TO_DATA, category))
    number_of_files = len(files)
    return number_of_files, files


def split(category):
    number_of_files, files = read_files(category)
    train_index = int(number_of_files * 0.7)
    validation_index = train_index + int(number_of_files * 0.2)
    train = files[: train_index]
    validation = files[train_index: validation_index]
    test = files[validation_index:]
    return train, validation, test


def store_file(files, folder, category):
    path = ''
    if folder == 'train':
        path = PATH_TO_TRAIN
    elif folder == 'validation':
        path = PATH_TO_VALIDATION
    elif folder == 'test':
        path = PATH_TO_TEST

    os.mkdir(os.path.join(path, category))
    for file in files:
        source = os.path.join(PATH_TO_DATA, category, file)
        destination = os.path.join(path, category, file)
        shutil.copy(source, destination)


categories = os.listdir(PATH_TO_DATA)
for category in categories:
    train, validation, test = split(category)
    store_file(train, 'train', category)
    store_file(validation, 'validation', category)
    store_file(test, 'test', category)
    print(category + ' done!')
