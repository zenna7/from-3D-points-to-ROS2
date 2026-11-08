import numpy as np
import open3d as o3d
from pathlib import Path

CONFIG = {
    "in_dir": "ICP_pipe/frames", "out_file": "ICP_pipe/reconstruction.ply", "voxel": 0.01, "min_fitness": 0.3,
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


files = sorted(Path(CONFIG["in_dir"]).glob("frame_*.ply"))
clouds = [o3d.io.read_point_cloud(str(f)).remove_statistical_outlier(20, 2.0)[0] for f in files]

pose = np.eye(4)
tgt = clouds[0]
merged = o3d.geometry.PointCloud(tgt)
for f, src in zip(files[1:], clouds[1:]):
    res = register(src, tgt)
    print(f"{f.name}: fitness {res.fitness:.3f}  rmse {res.inlier_rmse:.4f}")
    if res.fitness < CONFIG["min_fitness"]:
        print("  skipped")
        continue
    pose = pose @ res.transformation
    merged += o3d.geometry.PointCloud(src).transform(pose)
    tgt = src

merged = merged.voxel_down_sample(CONFIG["voxel"])
o3d.io.write_point_cloud(CONFIG["out_file"], merged)
o3d.visualization.draw_geometries([merged])