# Node 2: receives the point clouds, tracks the camera motion with ICP,
# prints position and rotation, and plots the path live
import threading
import time

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


CONFIG = {
    "pre_voxel": 0.01,    # m, first downsampling of every incoming cloud
    "max_depth": 4.0,     # m, farther points are too noisy
    "min_points": 1000,   # clouds with fewer points are skipped
    "min_fitness": 0.3,   # results below this are skipped
    "tolerance": 1e-6,    # ICP stops when the improvement is smaller than this
    # voxel, max_dist, iterations (coarse to fine, big max_dist first to catch big jumps)
    "scales": [(0.08, 0.30, 60), (0.04, 0.12, 50), (0.02, 0.05, 40)],
}
reg = o3d.pipelines.registration    # shortcut to Open3D's registration module

# tip first, in camera axes (x right, y down, z forward); isosceles, 1 long and 0.4 wide
TRIANGLE = np.array([[0, 0, 2 / 3], [-0.2, 0, -1 / 3], [0.2, 0, -1 / 3]])

# key -> (elevation, azimuth, axis to hide, title); add a line to add a view
VIEWS = {
    "0": (20, -70, None, "3D view"),
    "1": (90, -90, "z", "Top view"),
}


def to_plot(a):
    """camera axes (x right, y down, z forward) -> plot axes (x right, y forward, z up)"""
    return a[..., [0, 2, 1]] * [1, 1, -1]


def to_o3d(msg):
    xyz = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)    # raw bytes -> (N, 3)
    xyz = xyz[xyz[:, 2] < CONFIG["max_depth"]]    # drop far points
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    p = p.voxel_down_sample(CONFIG["pre_voxel"])
    return p.remove_statistical_outlier(20, 2.0)[0]    # remove isolated noise points


def register(src, tgt, init):
    """Finds the motion that aligns src onto tgt, starting from the guess init."""
    tol = CONFIG["tolerance"]
    T = init
    for voxel, dist, iters in CONFIG["scales"]:    # each scale starts from the previous result
        s = src.voxel_down_sample(voxel)
        t = tgt.voxel_down_sample(voxel)
        t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))    # only the target needs normals
        method = reg.TransformationEstimationPointToPlane(reg.HuberLoss(voxel))    # Huber: limits the weight of outliers
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
        self.tgt = None         # previous cloud, new clouds are matched against it
        self.started = False    # False while waiting for Enter or paused
        self.path = [np.zeros(3)]    # camera positions, starting at the origin

        # plot window
        plt.ion()    # non-blocking mode
        fig = plt.figure("camera path")
        self.ax = fig.add_subplot(projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)    # no empty margins
        fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.ax.grid(False)    # comment this line out to bring the grid back
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))    # transparent: removes the light gray planes
        self.ax.set_title("3D view", y=0.95)
        self.ax.set_xlabel("x l/r")
        self.ax.set_ylabel("z f/b")
        self.ax.set_zlabel("y up/down")
        self.ax.view_init(elev=20, azim=-70)    # fixed starting view, forward goes into the screen
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.scatter(0, 0, 0, color="black")    # starting position

        self.line, = self.ax.plot([], [], [], color="tab:blue")    # the path
        self.tri = Poly3DCollection([TRIANGLE * 0.12], color="tab:red")    # current pose
        self.ax.add_collection3d(self.tri)
        self.create_timer(0.1, self.draw)    # redraw 10 times a second
        threading.Thread(target=self.wait_for_key, daemon=True).start()    # reads Enter without blocking ROS


    def wait_for_key(self):
        input("Waiting for key: place the camera, then press Enter to start\n")
        self.started = True
        self.get_logger().info("tracking started (Enter = pause/resume)")
        while True:
            input()
            self.started = not self.started
            if self.started:
                self.tgt = None            # next frame becomes the new reference
                self.step = np.eye(4)
                self.get_logger().info("tracking resumed")
            else:
                self.get_logger().info("tracking paused")


    def on_key(self, event):
        # key presses in the plot window switch view (see VIEWS)
        if event.key in VIEWS:
            elev, azim, hide, title = VIEWS[event.key]
            self.ax.view_init(elev, azim)
            self.ax.set_title(title, y=0.95)
            for name in "xyz":
                shown = name != hide
                getattr(self.ax, f"{name}axis").label.set_visible(shown)    # axis name
                self.ax.tick_params(axis=name, labelleft=shown)             # axis numbers
            self.ax.figure.canvas.draw_idle()


    def draw(self):
        p = np.array(self.path)
        q = to_plot(p)
        self.line.set_data_3d(q[:, 0], q[:, 1], q[:, 2])

        lim = max(0.5, np.abs(p).max() * 1.1)    # grows when the path leaves the box
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)

        size = lim / 4    # the triangle scales with the box
        tri = (TRIANGLE * size) @ self.pose[:3, :3].T + p[-1]    # rotate with the camera, move to its position
        self.tri.set_verts([to_plot(tri)])
        plt.pause(0.001)    # lets the window process events


    def cb(self, msg):
        if not self.started:    # waiting for Enter or paused
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
        except RuntimeError:    # Open3D raises this when no points can be paired
            self.get_logger().info("skipped (no correspondences)")
            return

        if res.fitness < CONFIG["min_fitness"]:
            self.get_logger().info(f"skipped (fitness {res.fitness:.3f})")
            return

        self.step = res.transformation
        self.pose = self.pose @ self.step    # chain the step onto the total pose
        self.path.append(self.pose[:3, 3].copy())
        self.tgt = src

        x, y, z = self.pose[:3, 3]
        a1, a2, a3 = Rotation.from_matrix(self.pose[:3, :3]).as_euler("xyz", degrees=True)
        ms = (time.perf_counter() - t0) * 1000    # processing time of this frame
        self.get_logger().info(
            f"fitness {res.fitness:.3f} rmse {res.inlier_rmse:.4f} | "
            f"T = ({x:.3f}, {y:.3f}, {z:.3f}) m | "
            f"theta = ({a1:.1f}, {a2:.1f}, {a3:.1f}) deg | {ms:.0f} ms")


def main():
    rclpy.init()
    try:
        rclpy.spin(PoseNode())    # runs until Ctrl+C
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
