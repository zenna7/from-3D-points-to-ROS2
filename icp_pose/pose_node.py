import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2

CONFIG = {
    "min_fitness": 0.3,
    "scales": [(0.04, 0.12, 30), (0.02, 0.06, 20), (0.01, 0.03, 15)],  # voxel, max_dist, iterations
}
reg = o3d.pipelines.registration


def prep(p, voxel):
    p = p.voxel_down_sample(voxel)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return p


def register(src, tgt):
    T = np.eye(4)
    for voxel, dist, iters in CONFIG["scales"]:
        res = reg.registration_colored_icp(prep(src, voxel), prep(tgt, voxel), dist, T,
                                           reg.TransformationEstimationForColoredICP(),
                                           reg.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                      relative_rmse=1e-6,
                                                                      max_iteration=iters))
        T = res.transformation
    return res


def to_o3d(msg):
    data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
    rgb = data[:, 3].copy().view(np.uint32)
    colors = np.stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255], axis=1) / 255.0
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(data[:, :3].astype(np.float64))
    p.colors = o3d.utility.Vector3dVector(colors)
    return p.remove_statistical_outlier(20, 2.0)[0]


class PoseNode(Node):
    def __init__(self):
        super().__init__("pose_node")
        self.sub = self.create_subscription(PointCloud2, "points", self.cb, 1)
        self.pose = np.eye(4)
        self.tgt = None

    def cb(self, msg):
        src = to_o3d(msg)
        if self.tgt is None:
            self.tgt = src
            return

        try:
            res = register(src, self.tgt)
        except RuntimeError:
            self.get_logger().info("skipped (no correspondences)")
            return
        
        if res.fitness < CONFIG["min_fitness"]:
            self.get_logger().info(f"skipped (fitness {res.fitness:.3f})")
            return

        self.pose = self.pose @ res.transformation
        self.tgt = src

        x, y, z = self.pose[:3, 3]
        t1, t2, t3 = Rotation.from_matrix(self.pose[:3, :3]).as_euler("xyz", degrees=True)
        self.get_logger().info(
            f"fitness {res.fitness:.3f} rmse {res.inlier_rmse:.4f} | "
            f"T = ({x:.3f}, {y:.3f}, {z:.3f}) m | "
            f"theta = ({t1:.1f}, {t2:.1f}, {t3:.1f}) deg")


def main():
    rclpy.init()
    rclpy.spin(PoseNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()