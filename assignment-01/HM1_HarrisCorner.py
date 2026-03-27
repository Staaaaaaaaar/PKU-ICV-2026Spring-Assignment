import numpy as np
from utils import  read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x,Sobel_filter_y,padding



def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: array
    """
    # Gradient of image.
    i_x = Sobel_filter_x(input_img)
    i_y = Sobel_filter_y(input_img)

    # Second-moment matrix entries.
    i_xx = i_x * i_x
    i_yy = i_y * i_y
    i_xy = i_x * i_y

    # Rectangular window aggregation.
    pad = window_size // 2
    window = np.ones((window_size, window_size), dtype=input_img.dtype)

    s_xx = convolve(padding(i_xx, pad, "zeroPadding"), window)
    s_yy = convolve(padding(i_yy, pad, "zeroPadding"), window)
    s_xy = convolve(padding(i_xy, pad, "zeroPadding"), window)

    # Harris response: theta = det(M) - alpha * trace(M)^2
    det_m = s_xx * s_yy - s_xy * s_xy
    trace_m = s_xx + s_yy
    theta = det_m - alpha * trace_m * trace_m

    mask = theta > threshold
    corner_rc = np.argwhere(mask)
    corner_theta = theta[mask].reshape(-1, 1)

    if corner_rc.shape[0] == 0:
        return []

    corner_list = np.concatenate((corner_rc.astype(np.float64), corner_theta), axis=1)
    return corner_list # array, each row contains information about one corner, namely (index of row, index of col, theta)



if __name__=="__main__":

    #Load the input images
    input_img = read_img("hand_writting.png")/255.

    #you can adjust the parameters to fit your own implementation 
    window_size = 5
    alpha = 0.04
    threshold = 10

    corner_list = corner_response_function(input_img,window_size,alpha,threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key = lambda x: x[2], reverse = True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted :
        for j in NML_selected :
            if(abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis) :
                break
        else :
            NML_selected.append(i[:-1])


    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
