import numpy as np

def rotate_point(xy: np.array, angle, center):
    rot_array = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_xy =  np.dot(xy-center, rot_array) + center
    
    return rotated_xy

def img_scaler(img_shape, min_x, max_x, min_y, max_y, scale=0.5):
    w, h = abs(min_x-max_x), abs(min_y-max_y)
    w_scale, h_scale = w*scale/2, h*scale/2
    
    return max(0, min_x-w_scale), min(max_x+w_scale, img_shape[1]), max(0, min_y-h_scale), min(max_y+h_scale, img_shape[0])