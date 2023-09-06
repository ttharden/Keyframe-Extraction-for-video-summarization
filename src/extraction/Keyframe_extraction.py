import pickle
import cv2
import numpy as np

from Kmeans_improvment import kmeans_silhouette
from save_keyframe import save_frames
from Redundancy import redundancy


def scen_keyframe_extraction(scenes_path, features_path, video_path, save_path, folder_path):
    # Get lens segmentation data
    number_list = []
    with open(scenes_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line)
            numbers = line.strip().split(' ')
            print(numbers)
            number_list.extend([int(number) for number in numbers])

    # Read inference data from local
    with open(features_path, 'rb') as file:
        features = pickle.load(file)

    features = np.asarray(features)
    # print(len(features))

    # Clustering at each shot to obtain keyframe sequence numbers
    keyframe_index = []
    for i in range(0, len(number_list) - 1, 2):
        start = number_list[i]
        end = number_list[i + 1]
        # print(start, end)
        sub_features = features[start:end]
        best_labels, best_centers, k, index = kmeans_silhouette(sub_features)
        # print(index)
        final_index = [x + start for x in index]
        # final_index.sort()
        # print("clustering：" + str(keyframe_index))
        # print(start, end)
        final_index = redundancy(video_path, final_index, 0.8)
        # print(final_index)
        keyframe_index += final_index
    keyframe_index.sort()
    print("final_index：" + str(keyframe_index))

    # save keyframe
    save_frames(keyframe_index, video_path, save_path, folder_path)




