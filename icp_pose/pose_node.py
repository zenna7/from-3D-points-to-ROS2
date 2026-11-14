import threading
import time

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2

CONFIG = {
    "pre_voxel": 0.01,    # m, first downsampling of every incoming cloud
    "max_depth": 4.0,     # m, farther points are too noisy
    "min_points": 1000,   # clouds with fewer points are skipped
    "min_fitness": 0.3,   # results below this are skipped
    "tolerance": 1e-6,    # ICP stops when the improvement is smaller than this
    # voxel, max_dist, iterations (coarse to fine, big max_dist first to catch big jumps)
    "scales": [(0.08, 0.30, 60), (0.04, 0.12, 50), (0.02, 0.05, 40)],
}
reg = o3d.pipelines.registration


def to_o3d(msg):
    xyz = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)
    xyz = xyz[xyz[:, 2] < CONFIG["max_depth"]]
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    p = p.voxel_down_sample(CONFIG["pre_voxel"])
    return p.remove_statistical_outlier(20, 2.0)[0]


def register(src, tgt, init):
    tol = CONFIG["tolerance"]
    T = init
    for voxel, dist, iters in CONFIG["scales"]:
        s = src.voxel_down_sample(voxel)
        t = tgt.voxel_down_sample(voxel)
        t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
        method = reg.TransformationEstimationPointToPlane(reg.HuberLoss(voxel))
        criteria = reg.ICPConvergenceCriteria(relative_fitness=tol, relative_rmse=tol,
                                              max_iteration=iters)
        res = reg.registration_icp(s, t, dist, T, method, criteria)
        T = res.transformation
    return res


class PoseNode(Node):
    def __init__(self):
        super().__init__("pose_node")
        self.sub = self.create_subscription(PointCloud2, "points", self.cb, 1)
        self.pose = np.eye(4)   # camera pose relative to the reference frame
        self.step = np.eye(4)   # last frame-to-frame motion, used as starting guess
        self.tgt = None
        self.started = False
        threading.Thread(target=self.wait_for_key, daemon=True).start()

    def wait_for_key(self):
        input("Waiting for key: place the camera, then press Enter to start\n")
        self.started = True
        self.get_logger().info("tracking started")

    def cb(self, msg):
        if not self.started:
            return

        t0 = time.perf_counter()
        src = to_o3d(msg)
        if len(src.points) < CONFIG["min_points"]:
            self.get_logger().info("skipped (too few points)")
            return

        if self.tgt is None:    # first frame after the key press is the reference
            self.tgt = src
            return

        try:
            res = register(src, self.tgt, self.step)
        except RuntimeError:
            self.get_logger().info("skipped (no correspondences)")
            return

        if res.fitness < CONFIG["min_fitness"]:
            self.get_logger().info(f"skipped (fitness {res.fitness:.3f})")
            return

        self.step = res.transformation
        self.pose = self.pose @ self.step
        self.tgt = src

        x, y, z = self.pose[:3, 3]
        a1, a2, a3 = Rotation.from_matrix(self.pose[:3, :3]).as_euler("xyz", degrees=True)
        ms = (time.perf_counter() - t0) * 1000
        self.get_logger().info(
            f"fitness {res.fitness:.3f} rmse {res.inlier_rmse:.4f} | "
            f"T = ({x:.3f}, {y:.3f}, {z:.3f}) m | "
            f"theta = ({a1:.1f}, {a2:.1f}, {a3:.1f}) deg | {ms:.0f} ms")


def main():
    rclpy.init()
    try:
        rclpy.spin(PoseNode())
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()