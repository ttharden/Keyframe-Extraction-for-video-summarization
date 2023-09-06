import cv2
import numpy as np


def redundancy(video_path, keyframe_index, threshold):
    # colour histogram
    def color_histogram(img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 255, 0, 255, 0, 255])
        return hist.flatten()

    init_number = 0
    final_index = []

    # print(keyframe_index)

    final_index.append(keyframe_index[init_number])

    # List for storing colour histograms
    histograms = []
    # open the video
    video = cv2.VideoCapture(video_path)

    # Iterate through the list of frame numbers
    for frame_index in keyframe_index:
        # Setting the current frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read current frame
        ret, frame = video.read()

        if ret:
            # Calculate the colour histogram
            hist = color_histogram(frame)
            histograms.append(hist)

    # Releasing the video
    video.release()
    histogram = np.array(histograms)
    new_histogram = []
    mid_index = []

    # Filter pure colour frames, low information frames
    for i in range(len(histogram)):
        peak_count = np.sum(histogram[i] > 0)
        # print(i, peak_count)
        if peak_count > 10:
            new_histogram.append(histogram[i])
            mid_index.append(keyframe_index[i])
    # print(mid_index)

    # Get the global similarity matrix
    simis = []
    for i in range(len(mid_index)):
        simi = []
        base = new_histogram[i]
        cv2.normalize(base, base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        for j in range(len(mid_index)):
            lat = new_histogram[j]
            cv2.normalize(lat, lat, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            similarity = np.dot(base, lat) / (np.linalg.norm(base) * np.linalg.norm(lat))
            simi.append(similarity)
        simis.append(simi)
    del_index = []
    # print(simis)
    for i in range(len(mid_index)):
        for j in range(i + 1, len(mid_index)):
            if mid_index[i] in del_index or mid_index[j] in del_index:
                continue
            if simis[i][j] > threshold:
                print(mid_index[j], simis[i][j], mid_index[i])
                del_index.append(mid_index[j])
    set_mid_index = set(mid_index)
    set_del_index = set(del_index)
    set_final_index = set_mid_index - set_del_index
    final_index = list(set_final_index)
    final_index.sort()
    print(final_index)

    return final_index

