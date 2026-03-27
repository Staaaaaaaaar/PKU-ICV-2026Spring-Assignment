import argparse
import numpy as np

from task1_convolution import conv2d_6x6_via_dbt, conv2d_im2col, pad2d
from task2_canny import canny_pipeline
from task3_harris import harris_response
from task4_plane_ransac import fit_plane_ransac_parallel


def run_task1_demo():
    print("[Task 1] Convolution demos")

    x6 = np.arange(36, dtype=np.float64).reshape(6, 6)
    k3 = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float64)

    y_dbt = conv2d_6x6_via_dbt(x6, k3)
    print("  DBT output shape:", y_dbt.shape)
    print("  DBT output sum:", float(np.sum(y_dbt)))

    img = np.arange(64, dtype=np.float64).reshape(8, 8)
    y_im2col = conv2d_im2col(img, k3, stride=1, padding=1, pad_mode="replicate")
    print("  im2col output shape:", y_im2col.shape)
    print("  im2col output mean:", float(np.mean(y_im2col)))

    padded = pad2d(img, pad=2, mode="replicate")
    print("  pad2d(replicate) shape:", padded.shape)


def run_task2_demo():
    print("[Task 2] Canny demos")

    img = np.zeros((32, 32), dtype=np.float64)
    img[8:24, 10:22] = 255.0

    m, d, nms, edges = canny_pipeline(img, low=40.0, high=80.0)
    print("  M shape:", m.shape, "max:", float(np.max(m)))
    print("  D shape:", d.shape)
    print("  NMS nonzero:", int(np.count_nonzero(nms)))
    print("  Edges nonzero:", int(np.count_nonzero(edges)))


def run_task3_demo():
    print("[Task 3] Harris demos")

    img = np.zeros((32, 32), dtype=np.float64)
    img[6:26, 6:26] = 200.0

    r = harris_response(img, window_size=3, k=0.04)
    print("  R shape:", r.shape)
    print("  R min/max:", float(np.min(r)), float(np.max(r)))


def run_task4_demo():
    print("[Task 4] Plane RANSAC demos")

    rng = np.random.default_rng(42)

    n_in = 300
    x = rng.uniform(-3.0, 3.0, size=n_in)
    y = rng.uniform(-3.0, 3.0, size=n_in)
    z = 0.5 * x - 0.2 * y + 1.0 + rng.normal(0.0, 0.03, size=n_in)
    inliers = np.stack([x, y, z], axis=1)

    n_out = 80
    outliers = rng.uniform(-3.0, 3.0, size=(n_out, 3))

    points = np.concatenate([inliers, outliers], axis=0)

    plane, inlier_mask, n_iters = fit_plane_ransac_parallel(
        points,
        dist_thresh=0.08,
        w_est=0.6,
        success_prob=0.999,
        rng_seed=7,
    )

    print("  Iterations:", n_iters)
    print("  Plane [a, b, c, d]:", plane)
    print("  Inlier count:", int(np.sum(inlier_mask)), "/", points.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Unified test entry for assignment-01")
    parser.add_argument(
        "--task",
        choices=["all", "1", "2", "3", "4"],
        default="all",
        help="Select which task demo to run",
    )
    args = parser.parse_args()

    if args.task in ("all", "1"):
        run_task1_demo()
        print()
    if args.task in ("all", "2"):
        run_task2_demo()
        print()
    if args.task in ("all", "3"):
        run_task3_demo()
        print()
    if args.task in ("all", "4"):
        run_task4_demo()


if __name__ == "__main__":
    main()
