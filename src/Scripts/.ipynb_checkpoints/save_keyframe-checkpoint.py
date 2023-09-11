import os
import cv2


def save_frames(keyframe_indexes, video_path, save_path, folder_name):
    # open video
    cap = cv2.VideoCapture(video_path)

    # Creating a folder path for saving images
    folder_path = os.path.join(save_path, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Setting the current frame number
    current_index = 0

    # Cyclic reading of the video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 判断当前帧是否为关键帧，如果是，则保存帧图像
        if current_index in keyframe_indexes:
            file_name = '{}.jpg'.format(current_index)
            file_path = os.path.join(folder_path, file_name)
            cv2.imwrite(file_path, frame)

        current_index += 1

    # release
    cap.release()
