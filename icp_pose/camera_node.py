import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

CONFIG = {"width": 640, "height": 480, "fps": 30, "publish_every": 6}  # 30 fps / 6 = 5 Hz


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.pub = self.create_publisher(PointCloud2, "points", 1)

        w, h, fps = CONFIG["width"], CONFIG["height"], CONFIG["fps"]
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        self.pipe = rs.pipeline()
        self.pipe.start(cfg)
        self.pc = rs.pointcloud()
        self.count = 0

        self.create_timer(1 / fps, self.tick)

    def tick(self):
        try:
            frames = self.pipe.wait_for_frames(1000)
        except RuntimeError:
            self.get_logger().warning("no frames from camera (disconnected?)")
            return

        color = frames.get_color_frame()
        if not color:
            return

        # 1. video feed
        img = np.asanyarray(color.get_data())
        cv2.imshow("camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        # 2. 3D points, at a lower rate
        self.count += 1
        dots = "." * (self.count // 20 % 4)
        print(f"Streaming camera feed{dots:<3}", end="\r", flush=True)
        if self.count % CONFIG["publish_every"] == 0:
            self.publish(frames)

    def publish(self, frames):
        depth = frames.get_depth_frame()
        if not depth:
            return

        pts = self.pc.calculate(depth)
        xyz = np.asanyarray(pts.get_vertices()).view(np.float32).reshape(-1, 3)
        xyz = xyz[xyz[:, 2] > 0]

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height, msg.width = 1, len(xyz)
        msg.fields = [PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(["x", "y", "z"])]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(xyz)
        msg.is_dense = True
        msg.data = xyz.tobytes()
        self.pub.publish(msg)



def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.pipe.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()