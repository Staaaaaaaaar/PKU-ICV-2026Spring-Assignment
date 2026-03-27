import numpy as np
from utils import draw_save_plane_with_points, normalize


if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")


    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     

    # For a 3-point plane hypothesis, all 3 sampled points must be inliers.
    # Use exact probability without replacement: (100/130)*(99/129)*(98/128).
    success_prob_per_hyp = (100.0 / 130.0) * (99.0 / 129.0) * (98.0 / 128.0)
    sample_time = int(np.ceil(np.log(1.0 - 0.999) / np.log(1.0 - success_prob_per_hyp)))
    distance_threshold = 0.05

    # sample points group
    num_points = noise_points.shape[0]
    rand_key = np.random.rand(sample_time, num_points)
    sample_idx = np.argpartition(rand_key, kth=2, axis=1)[:, :3]
    sample_points = noise_points[sample_idx]  # (sample_time, 3, 3)




    # estimate the plane with sampled points group
    p1 = sample_points[:, 0, :]
    p2 = sample_points[:, 1, :]
    p3 = sample_points[:, 2, :]

    v1 = p2 - p1
    v2 = p3 - p1
    normal_all = np.cross(v1, v2)  # (sample_time, 3)
    d_all = -np.sum(normal_all * p1, axis=1)  # (sample_time,)




    #evaluate inliers (with point-to-plance distance < distance_threshold)
    normal_norm = np.linalg.norm(normal_all, axis=1)
    valid_hypothesis = normal_norm > 1e-12

    # Point-to-plane distance for all points and all hypotheses in parallel.
    numerator = np.abs(noise_points @ normal_all.T + d_all.reshape(1, -1))
    distance = numerator / (normal_norm.reshape(1, -1) + 1e-12)
    inlier_mask_all = distance < distance_threshold
    inlier_count = np.sum(inlier_mask_all, axis=0)
    inlier_count = np.where(valid_hypothesis, inlier_count, -1)

    best_idx = int(np.argmax(inlier_count))
    best_inliers = noise_points[inlier_mask_all[:, best_idx]]



    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    if best_inliers.shape[0] < 3:
        best_inliers = sample_points[best_idx]

    centroid = np.mean(best_inliers, axis=0)
    centered = best_inliers - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    d = -np.dot(normal, centroid)
    pf = np.concatenate((normal, np.array([d])))



    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt('result/HM1_RANSAC_sample_time.txt', np.array([sample_time]))
