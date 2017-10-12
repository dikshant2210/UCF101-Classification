import numpy as np
import os
import imageio
import utils
import pickle as pkl
from extractor import Extractor
import time


NUMBER_OF_FRAMES = 40

path_to_read_data = 'data/UCF-101'
path_to_store_data = 'data/UCF-101-Inception-Features'

categories = os.listdir(path_to_read_data)
model = Extractor()


def get_activations(path_to_video):
    try:
        video = imageio.get_reader(path_to_video)
        frames = [frame for frame in video]
        frame_length = len(frames)
        frames = utils.rescale_list(frames, NUMBER_OF_FRAMES)
        sequence = list()
        try:
            for frame in frames:
                features = model.extract(frame.astype(np.float64))
                sequence.append(features)
        except TypeError:
            print(path_to_video, frame_length)
        return np.array(sequence)
    except:
        print(path_to_video)
        return []


def store_activations(category):
    video_files = os.listdir(os.path.join(path_to_read_data, category))
    path_to_dir = os.path.join(path_to_store_data, category)
    if not os.path.isdir(path_to_dir):
        os.mkdir(path_to_dir)
    for video in video_files:
        path = os.path.join(path_to_read_data, category, video)

        store_name = video.replace('.avi', '.pkl')
        path_to_store_file = os.path.join(path_to_store_data, category, store_name)
        if os.path.isfile(path_to_store_file):
            print(store_name + ' stored.')
            continue

        sequence = get_activations(path)
        if len(sequence) != 0:
            with open(path_to_store_file, 'wb') as file:
                pkl.dump(sequence, file, protocol=pkl.HIGHEST_PROTOCOL)
            print(store_name + ' stored.')
        else:
            print(store_name + ' not stored.')


start = time.time()
for category in categories:
    store_activations(category)
    checkpoint = time.time()
    print('---------------------*****---------------------')
    print(category + ' done!')
    print('Time elapsed: {}s'.format(checkpoint - start))
    print('---------------------*****---------------------')
