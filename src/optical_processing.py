import numpy as np
import cv2

from scipy.spatial.distance import pdist, squareform
from scipy.signal import convolve2d, firwin, filtfilt

def join_convexity_defects(thresh):
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)  # assuming the largest contour is the heart
    cv2.drawContours(thresh, contours, -1, (0, 255, 0), 1)

    # Convexity defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    defects = defects[np.argsort(defects[:, 0, -1])][-2:]

    # draw a line between the two points of the convex segment
    points = [tuple(contour[f][0]) for f in defects[:, 0, 2]]
    cv2.line(thresh, points[0], points[1], [0, 0, 0], 2)
    p1 = points[0]
    p2 = points[1]
    d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return thresh, d

def get_ventricles_mask(video):
    image = video[0]  # take the first frame, heart the same in all frames

    # create mask in cv2
    img = np.nan_to_num(image, nan=-10)
    normalized_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_8bit = (normalized_img * 255).astype(np.uint8)
    _, thresh = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY)

    atria_area = 1
    ventricles_area = 21
    area_ratio = ventricles_area / atria_area
    dist_between_points = 0
    repeats = 0

    # join the convexity defects if the cropped regions are too small and too close together
    while dist_between_points < 25 and area_ratio > 10:
        if repeats > 1:
            # only repeat twice
            break
        thresh, dist_between_points = join_convexity_defects(thresh)

        # contour the ventricles and the atria
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ventricles = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [ventricles], -1, (255), thickness=cv2.FILLED)

        # calculate the area of the atria
        atria = cv2.bitwise_xor(thresh, mask)
        atria_area = np.sum(atria == 255)
        ventricles_area = np.sum(mask == 255)
        area_ratio = ventricles_area / atria_area
        repeats += 1

    if area_ratio < 3:
        # if the ventricles are too small, use the original mask
        mask = (image == image).astype(float)

    # create mask for ventricles
    mask = (mask > 0).astype(float)
    mask[mask == 0] = np.nan
    return mask

def optical_data_preprocessing(videos, fs=395.4):
    processed = []

    for video in videos:

        # remove ventricles
        mask = get_ventricles_mask(video)
        video = video * (mask == 1)

        # preprocessing
        filt = spatial_filtering(video, kernel_size=3)
        filt = temporal_filtering(filt, fs, 30.0, axis=0)  # lpf 100 Hz
        processed.append(filt)
    return processed

def design_lowpass(sampling_rate, cutoff_frequency, numtaps=100):
    nyquist_rate = sampling_rate / 2.0
    return firwin(numtaps, cutoff_frequency / nyquist_rate)


def temporal_filtering(data, sampling_rate, cutoff_frequency, axis):
    fir_coefficients = design_lowpass(sampling_rate, cutoff_frequency)
    return filtfilt(fir_coefficients, 1.0, data, axis)  # ensure zero phase distortion

def mean_std_norm(s, axis):
    s = s - np.mean(s, axis=axis)
    s = s / (np.std(s, axis=axis))
    return s

def spatial_filtering(video, kernel_size=3):
    filtered_video = []
    for frame in video:
        filtered_frame = spatial_binning(frame, kernel_size)
        filtered_video.append(filtered_frame)
    return np.array(filtered_video)

def spatial_binning(image, kernel_size=3):
    # Define a 3x3 uniform kernel
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2

    # Use convolve2d to apply the kernel on the image
    binned_image = convolve2d(image, kernel, mode='same')
    return binned_image


def downsample_video(video, temporal_factor, spatial_factor):
    """downsample the spatio-temporal resolution of the video

    Args:
        video (np.array): (L, W, D) numpy array of the video
        temporal_factor (int): factor to downsample the temporal resolution of the video
        spatial_factor (int): factor to downsample the spatial resolution of the video

    Returns:
        np.array: (L', W', D') numpy array of the downsampled video
    """
    L, W, D = video.shape

    # calculate the new dimensions after downsampling
    L_new = L // temporal_factor
    W_new = W // spatial_factor
    D_new = D // spatial_factor

    # initialise downsampled video array
    downsampled_video = np.empty((L_new, W_new, D_new), dtype=video.dtype)

    # downsample the video
    for i in range(L_new):
        start_frame = i * temporal_factor
        end_frame = start_frame + temporal_factor
        selected_frames = video[start_frame:end_frame]
        downsampled_frame = cv2.resize(selected_frames.mean(axis=0), (D_new, W_new))
        downsampled_video[i] = downsampled_frame
    return downsampled_video

def downsample_videos(videos, temporal_factor, spatial_factor):
    """downsample the spatio-temporal resolution of the video

    Args:
        videos (list): list of numpy arrays of the voltage data
        temporal_factor (int): factor to downsample the temporal resolution of the video
        spatial_factor (int): factor to downsample the spatial resolution of the video

    Returns:
        list: list of numpy arrays of the downsampled videos
    """
    all_signals = []
    for v in range(len(videos)):
        v = downsample_video(videos[v], temporal_factor, spatial_factor)
        all_signals.append(v)
    return all_signals

def crop_videos_to_heart(videos):
    """crop the videos to the heart surface

    Args:
        videos (list): list of numpy arrays of the voltage data

    Returns:
        list: list of numpy arrays of the cropped videos
    """
    vertices = get_vertices_without_nans(videos)
    
    cropped_videos = []
    for v in range(len(videos)):
        # calculate border indices for cropping
        max_border = vertices[v].max(axis=0)
        min_border = vertices[v].min(axis=0)
        cropped_video = videos[v][:, min_border[0]:max_border[0], min_border[1]:max_border[1]]
        cropped_videos.append(cropped_video)
    return cropped_videos

def get_vertices_without_nans(signals):
    """get the vertices of the heart surface where the signal is not nan

    Args:
        signals (list): list of numpy arrays of the voltage data

    Returns:
        list: list of numpy arrays of the vertices on the heart surface
    """
    all_vertices = []
    for v in range(len(signals)):
        inds_x, inds_y = np.where(signals[v][0, :, :] == signals[v][0, :, :])

        # get the x,y coordinates of each vertex
        vertices = np.array(list(zip(inds_x, inds_y)))
        all_vertices.append(vertices)
    return all_vertices

def get_signals(vertices, videos):
    """get the signals from the vertices on the heart surface (nans removed)

    Args:
        vertices (list): list of numpy arrays of the vertices on the heart surface
        videos (list): list of numpy arrays of the voltage data

    Returns:
        list: list of numpy arrays of the signals from the vertices on the heart surface
    """
    all_signals = []
    for v in range(len(videos)):
        signals = videos[v][:, vertices[v][:, 0], vertices[v][:, 1]]  # (L, N, f)
        all_signals.append(signals)
    return all_signals

def adjacency_matrix_from_positions(vertices, dist_threshold, exp_scaler=1):
    """form a weighted adjacency matrix from vertex coordinates by thresholding expoential
    of the euclidean distance

    Args:
        vertices (_type_): _description_
        dist_threshold (_type_): _description_
        exp_scaler (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    pw_euclidean_distances = squareform(pdist(vertices))  # pairwise euclidean distances
    satisfies_threshold = pw_euclidean_distances <= dist_threshold
    pw_euclidean_distances[~satisfies_threshold] = 0
    pw_euclidean_distances[satisfies_threshold] = np.exp((-1/exp_scaler) * pw_euclidean_distances[satisfies_threshold]**2)
    return pw_euclidean_distances

def get_connectivity(vertices, dist_thresh=1.42, exp_scaler=1):
    """get the connectivity structure for each heart

    Args:
        vertices (list): list of numpy arrays of the vertices on the heart surface
        dist_thresh (float, optional): threshold for the distance between nodes. Defaults to 1.42.
        exp_scaler (int, optional): exponential scaler for the edge weights. Defaults to 1.
        multi_mesh_order (int, optional): order of the multi-mesh. Defaults to 0.
        use_weighted_adjacency (bool, optional): whether to use the edge weights in the adjacency matrix. Defaults to True.

    Returns:
        list: list of numpy arrays of the connectivity structure for each heart
    """
    connectivity = []
    for v in vertices:
        # get the connectivity structure for each heart
        # nodes are connected to their nearest pixels (up, down and diagonal)
        A = adjacency_matrix_from_positions(v, dist_thresh, exp_scaler=exp_scaler)
        A[A > 0] = 1
        connectivity.append(A)
    return connectivity