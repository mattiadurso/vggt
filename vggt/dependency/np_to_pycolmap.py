# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pycolmap
from .projection import project_3D_points_np


def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
    image_names=None,
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    3. Supports grouping shared cameras by folder name if image_names is provided.
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    reproj_mask = None

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(
            points3d, extrinsics, intrinsics
        )
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    ## Let it run with whatever is found, do not skip BA any more
    # if masks.sum(1).min() < min_inlier_per_frame:
    #     print(f"Not enough inliers per frame, skip BA.")
    #     return None, None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Track the mapping from valid_idx to point3D_id
    vidx_to_point3D_id = {}

    # Only add 3D points that have sufficient 2D points
    for vidx in valid_idx:
        # Use RGB colors if provided, otherwise use zeros
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
        point3D_id = reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)
        vidx_to_point3D_id[vidx] = point3D_id

    # Pre-compute cameras if shared_camera is True
    # Maps frame index `fidx` to `camera_id`
    frame_to_camera_id = {}

    if shared_camera:
        if image_names is not None:
            # Group frames by their parent folder (camera_name)
            # Format: camera_name/image_name
            print("Grouping frames by camera name")
            camera_groups = {}
            for idx, name in enumerate(image_names):
                # Extract camera name (folder name)
                camera_name = name.split("/")[0] if "/" in name else "default"
                if camera_name not in camera_groups:
                    camera_groups[camera_name] = []
                camera_groups[camera_name].append(idx)

            # Create one camera per group by averaging intrinsics
            for i, (cam_name, indices) in enumerate(camera_groups.items()):
                # Collect params for all frames in this group
                params_list = []
                for idx in indices:
                    params_list.append(
                        _build_pycolmap_intri(
                            idx, intrinsics, camera_type, extra_params
                        )
                    )

                # Average parameters
                avg_params = np.mean(params_list, axis=0)

                cam_id = i + 1
                camera = pycolmap.Camera(
                    camera_id=cam_id,
                    model=camera_type,
                    width=int(image_size[0]),
                    height=int(image_size[1]),
                    params=avg_params,
                )
                reconstruction.add_camera(camera)

                # Map frames to this camera
                for idx in indices:
                    frame_to_camera_id[idx] = cam_id
        else:
            # Fallback: Average all frames into a single global camera
            params_list = []
            for idx in range(N):
                params_list.append(
                    _build_pycolmap_intri(idx, intrinsics, camera_type, extra_params)
                )
            avg_params = np.mean(params_list, axis=0)

            cam_id = 1
            camera = pycolmap.Camera(
                camera_id=cam_id,
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=avg_params,
            )
            reconstruction.add_camera(camera)

            for idx in range(N):
                frame_to_camera_id[idx] = cam_id

    # Iterate over frames to add images and cameras (if not shared/pre-computed)
    for fidx in range(N):
        if shared_camera:
            cam_id = frame_to_camera_id[fidx]
        else:
            # Create a unique camera for this frame
            pycolmap_intri = _build_pycolmap_intri(
                fidx, intrinsics, camera_type, extra_params=None
            )
            cam_id = fidx + 1
            camera = pycolmap.Camera(
                camera_id=cam_id,
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=pycolmap_intri,
            )
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image_name = (
            image_names[fidx] if image_names is not None else f"image_{fidx + 1}"
        )

        # Create image WITHOUT cam_from_world in constructor
        image = pycolmap.Image(
            image_id=fidx + 1,
            name=image_name,
            camera_id=cam_id,
        )

        # Set cam_from_world AFTER creating the image
        image.cam_from_world = cam_from_world

        points2D_list = []
        point2D_idx = 0

        # Iterate through valid 3D points and check if they're visible in this frame
        for vidx in valid_idx:
            if masks[fidx, vidx]:  # Check if this point is visible in this frame
                point3D_id = vidx_to_point3D_id[vidx]  # Use the actual assigned ID
                point2D_xy = tracks[fidx, vidx]
                points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

                # add element
                track = reconstruction.points3D[point3D_id].track
                track.add_element(fidx + 1, point2D_idx)
                point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
        except:
            image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction, valid_mask  # Return valid_mask as well


def pycolmap_to_batch_np_matrix(
    reconstruction, device="cpu", camera_type="SIMPLE_PINHOLE"
):
    """
    Convert a PyCOLMAP Reconstruction Object to batched NumPy arrays.

    Args:
        reconstruction (pycolmap.Reconstruction): The reconstruction object from PyCOLMAP.
        device (str): Ignored in NumPy version (kept for API compatibility).
        camera_type (str): The type of camera model used (default: "SIMPLE_PINHOLE").

    Returns:
        tuple: A tuple containing points3D, extrinsics, intrinsics, and optionally extra_params.
    """

    num_images = len(reconstruction.images)
    max_points3D_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_points3D_id, 3))

    for point3D_id in reconstruction.points3D:
        points3D[point3D_id - 1] = reconstruction.points3D[point3D_id].xyz

    extrinsics = []
    intrinsics = []

    extra_params = [] if camera_type == "SIMPLE_RADIAL" else None

    for i in range(num_images):
        # Extract and append extrinsics
        pyimg = reconstruction.images[i + 1]
        pycam = reconstruction.cameras[pyimg.camera_id]
        matrix = pyimg.cam_from_world.matrix()
        extrinsics.append(matrix)

        # Extract and append intrinsics
        calibration_matrix = pycam.calibration_matrix()
        intrinsics.append(calibration_matrix)

        if camera_type == "SIMPLE_RADIAL":
            extra_params.append(pycam.params[-1])

    # Convert lists to NumPy arrays instead of torch tensors
    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)

    if camera_type == "SIMPLE_RADIAL":
        extra_params = np.stack(extra_params)
        extra_params = extra_params[:, None]

    return points3D, extrinsics, intrinsics, extra_params


########################################################


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=pycolmap_intri,
                camera_id=fidx + 1,
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            image_id=fidx + 1,
            name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # add element
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)

        except:
            print(f"frame {fidx + 1} does not have any points")

        # add image
        reconstruction.add_image(image)

    return reconstruction


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [
                intrinsics[fidx][0, 0],
                intrinsics[fidx][1, 1],
                intrinsics[fidx][0, 2],
                intrinsics[fidx][1, 2],
            ]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [
                focal,
                intrinsics[fidx][0, 2],
                intrinsics[fidx][1, 2],
                extra_params[fidx][0],
            ]
        )
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri
