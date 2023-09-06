# Description

In this project we use keyframe extraction skimming for video summarization.

we present a large model based sequential keyframe extraction, dubbed LMSKE, to extract minimal keyframes to sum up a given video with their sequences maintained. First, The large model TransNetV2 [11] was utilized to conduct shot segmentations, and the large model CLIP was employed to extract semantic features for each frame within each shot.Second, an adaptive clustering method is devised to automatically determine the optimal clusters, based on which we performed candidate keyframe selection and redundancy elimination shot by shot. Finally, a keyframe set was obtained by concatenating keyframes of all shots in chronological order.

# Method
## Shot segmentations
We use the large model TransNetV2 for segmentation. Code from https://github.com/soCzech/TransNetV2. By this step, we will get the shot segmentation result of the video and the result will be saved locally in txt form. The front indicates the start frame of the shot and the back indicates the end frame of the shot.

![case](images/scenes.png) 
## Feature Extraction
We use the large-scale model CLIP to extract semantic features for each frame in each shot. Code from https://www.modelscope.cn/models/damo/multi-modal_clip-vit-large-patch14_336_zh/summary. The video is subjected to feature extraction through the CLIP large model to obtain a 768-dimensional feature vector for each frame of the video. We save the features of the whole video locally in the form of .pkl for subsequent use.
## Clustering
We designed an adaptive clustering method to automatically determine the best clustering results. The code can be found in our repository. [Kmeans_improvment.py](src/extraction/Kmeans_improvment.py)
## Redundancy
After obtaining the clustering results, we perform shot-by-shot selection and redundancy elimination of candidate keyframes. In terms of de-redundancy, we mainly divide it into two aspects, on the one hand, it is for solid colour frames or low information frames, and on the other hand, it is for frames with a high degree of similarity. The code can be found in our repository. [Redundancy.py](src/extraction/Redundancy.py)

# Evaluation
## benchmark dataset
The dataset we use is a benchmark dataset we built ourselves called TvSum20, which is used to evaluate the performance of the keyframe extraction method. Dataset from https://github.com/ttharden/Keyframe-extraction
## scripts
We use a script to evaluate the extracted keyframes. The code can be found in our repository. [Evaluation.py](src/extraction/Evaluation.py)

# Case
This is a case of keyframe extraction by our proposed method. Below are examples of keyframe sequences extracted by different methods：![case](images/githubcase.png)
