import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from pathlib import Path

CONFIG = {"out_dir": "ICP_pipe/frames", "width": 848, "height": 480, "fps": 30}

out = Path(CONFIG["out_dir"])
for f in out.glob("frame_*.ply"):
    f.unlink()
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.depth, CONFIG["width"], CONFIG["height"], rs.format.z16, CONFIG["fps"])
cfg.enable_stream(rs.stream.color, CONFIG["width"], CONFIG["height"], rs.format.bgr8, CONFIG["fps"])
profile = pipe.start(cfg)

sensor = profile.get_device().first_depth_sensor()
depth_scale = sensor.get_depth_scale()
i = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
intrinsics = o3d.camera.PinholeCameraIntrinsic(i.width, i.height, i.fx, i.fy, i.ppx, i.ppy)
align = rs.align(rs.stream.depth)
spatial, temporal = rs.spatial_filter(), rs.temporal_filter()
laser = sensor.get_option_range(rs.option.laser_power)
n = len(list(out.glob("frame_*.ply")))

WIN = "s = save, q = quit"
cv2.namedWindow(WIN)
nop = lambda v: None
cv2.createTrackbar("min d(cm)", WIN, 40, 500, nop)
cv2.createTrackbar("max d(cm)", WIN, 300, 1000, nop)
cv2.createTrackbar("laser power", WIN, int(laser.default), int(laser.max), nop)
cv2.createTrackbar("spatial filter", WIN, 0, 5, nop)
cv2.createTrackbar("temporal filter", WIN, 0, 1, nop)

last_power = None
try:
    while True:
        frames = align.process(pipe.wait_for_frames())
        depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()

        lo = cv2.getTrackbarPos("min d(cm)", WIN) / 100
        hi = cv2.getTrackbarPos("max d(cm)", WIN) / 100
        power = int(round(cv2.getTrackbarPos("laser power", WIN) / laser.step) * laser.step)
        smooth = cv2.getTrackbarPos("spatial filter", WIN)

        if power != last_power:
            sensor.set_option(rs.option.laser_power, power)
            last_power = power
        if smooth:
            spatial.set_option(rs.option.filter_magnitude, smooth)
            depth_frame = spatial.process(depth_frame)
        if cv2.getTrackbarPos("temporal filter", WIN):
            depth_frame = temporal.process(depth_frame)

        depth = np.asanyarray(depth_frame.get_data())
        meters = depth * depth_scale
        mask = (depth > 0) & (meters >= lo) & (meters <= hi)

        color = np.asanyarray(color_frame.get_data())
        tint = np.zeros_like(color)
        tint[:] = (0, 255, 0)
        blended = cv2.addWeighted(color, 0.6, tint, 0.4, 0)
        preview = np.where(mask[..., None], blended, color)
        cv2.putText(preview, f"valid {mask.mean() * 100:.0f}%", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WIN, preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            masked = np.where(mask, depth, 0).astype(np.uint16)
            rgb = np.ascontiguousarray(color[..., ::-1])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb), o3d.geometry.Image(masked),
                depth_scale=1 / depth_scale, depth_trunc=hi, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
            o3d.io.write_point_cloud(str(out / f"frame_{n:04d}.ply"), pcd)
            print(f"saved frame_{n:04d}.ply ({len(pcd.points)} points)")
            n += 1
        elif key == ord("q"):
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()