import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.pub = self.create_publisher(PointCloud2, "points", 1)

        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

        self.create_timer(0.2, self.tick)  # 5 Hz

    def tick(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        depth, color = frames.get_depth_frame(), frames.get_color_frame()
        if not depth or not color:
            return

        self.pc.map_to(color)
        pts = self.pc.calculate(depth)
        xyz = np.asanyarray(pts.get_vertices()).view(np.float32).reshape(-1, 3)
        uv = np.asanyarray(pts.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        img = np.asanyarray(color.get_data())
        h, w, _ = img.shape

        px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        py = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
        rgb = img[py, px].astype(np.uint32)
        packed = ((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).view(np.float32)

        valid = xyz[:, 2] > 0
        data = np.hstack([xyz[valid], packed[valid, None]]).astype(np.float32)

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height, msg.width = 1, len(data)
        msg.fields = [PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(["x", "y", "z", "rgb"])]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(data)
        msg.is_dense = True
        msg.data = data.tobytes()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.pipe.stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()