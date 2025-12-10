import os
import glob
from vggt.wrapper import VGGTWrapper


paths = {
    "eth3d_local": {
        "base_path": "/home/mattia/Desktop/datasets/eth3d",
        "images_path": "images_by_k",
        "output_path": "/home/mattia/Desktop/Repos/batchsfm/benchmark",
    },
    "eth3d_orochi": {
        "base_path": "/data/mdurso/eth3d",
        "images_path": "images_by_k",
        "output_path": "/data/mdurso/eth3d/results",
    },
    "synth_local": {
        "base_path": "/home/mattia/HDD_Fast/Synth_poses",
        "images_path": "images",
        "output_path": "/home/mattia/Desktop/Repos/posebench/benchmarks_3D/results",
    },
    "7scenes_local": {
        "base_path": "/home/mattia/Desktop/datasets/7scenes",
        "images_path": "images_150",
        "output_path": "/home/mattia/Desktop/Repos/batchsfm/benchmark",
    },
    "mipnerf360_local": {
        "base_path": "/home/mattia/Desktop/datasets/mipnerf360",
        "images_path": "images_4_150",
        "output_path": "/home/mattia/Desktop/Repos/batchsfm/benchmark",
    },
    "terrasky3D_local": {
        "base_path": "/home/mattia/Desktop/datasets/terrasky3D",
        "images_path": "images_150",
        "output_path": "/home/mattia/Desktop/Repos/batchsfm/benchmark",
    },
    "terrasky3D_orochi": {
        "base_path": "/data/mdurso/terrasky3d",
        "images_path": "frames",
        "output_path": "/data/mdurso/terrasky3d/results",
    },
    "tt_local": {
        "base_path": "/home/mattia/Desktop/datasets/tt",
        "images_path": "images_150",
        "output_path": "/home/mattia/Desktop/Repos/batchsfm/benchmark",
    },
}

use_ba = False
cuda_id = 7

for dataset in [
    # "terrasky3D_local",
    "terrasky3D_orochi",
    # "terrasky3D_local",
    # "7scenes_local",
    # "eth3d_local",
    # "synth_local",
]:

    _ba = "_ba" if use_ba else ""
    base_path = paths[dataset]["base_path"]
    images_path = paths[dataset]["images_path"]  # I might have files in that path
    output_folder = "/home/mattia/Desktop/Repos/batchsfm/benchmarks"  # paths[dataset]["output_path"]
    scenes = sorted(os.listdir(f"{base_path}"))

    if not use_ba:
        vggt = VGGTWrapper(cuda_id=cuda_id)

    for scene in scenes:
        input_path = f"{base_path}/{scene}/{images_path}"
        output_path = (
            f"{output_folder}/vggt{_ba}/{dataset.split('_')[0]}/{scene}/sparse"
        )

        if not os.path.exists(input_path):
            print(f"Input path {input_path} does not exist, skipping...")
            continue

        if os.path.exists(output_path + "/cameras.txt"):
            print(f"Output path {output_path} already exists, skipping...")
            continue

        print(f"Processing {dataset} - {scene}...")

        num_images = len(
            glob.glob(f"{input_path}/*.*")
            + glob.glob(f"{input_path}/*/*.*", recursive=True)
        )

        if use_ba:
            vggt = VGGTWrapper(cuda_id=cuda_id, oom_safe=use_ba)

        _ = vggt.forward(
            input_path,
            output_path,
            max_images=-1,
            use_ba=use_ba,
            query_frame_num=10,  # max 10
            max_query_pts=2048,
            shared_camera=True,  # per folder, in my datasets usually true
        )
