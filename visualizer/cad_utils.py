"""CAD mesh conversion helpers for the Streamlit dashboard."""

from io import BytesIO
from itertools import permutations, product
from typing import Dict, List, Sequence, Tuple
import warnings

import numpy as np
import trimesh
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SUPPORTED_MESH_FORMATS = {"stl", "obj", "ply", "glb", "gltf"}


def _meshes_from_upload(filename: str, data: bytes) -> List[trimesh.Trimesh]:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in SUPPORTED_MESH_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_MESH_FORMATS))
        raise ValueError(
            f"Unsupported CAD mesh format '.{extension}'. Supported formats: {supported}. "
            "Export native CAD files such as STEP/IGES to a tessellated mesh first."
        )

    try:
        loaded = trimesh.load(
            BytesIO(data), file_type=extension, force="scene")
    except Exception as exc:
        raise ValueError(f"Could not load '{filename}' as a {extension.upper()} mesh: {exc}") from exc

    if isinstance(loaded, trimesh.Trimesh):
        candidates = [loaded]
    elif isinstance(loaded, trimesh.Scene):
        candidates = []
        for node in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph.get(node)
            mesh = loaded.geometry[geometry_name].copy()
            mesh.apply_transform(transform)
            candidates.append(mesh)
    else:
        candidates = []

    meshes = []
    for mesh in candidates:
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            continue
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(f"'{filename}' contains non-finite vertex coordinates.")

        valid_faces = mesh.area_faces > np.finfo(np.float64).eps
        if not valid_faces.any():
            continue
        if not valid_faces.all():
            mesh = mesh.copy()
            mesh.update_faces(valid_faces)
        meshes.append(mesh)

    if not meshes:
        raise ValueError(f"'{filename}' contains no usable triangle surfaces.")
    return meshes


def _allocate_surface_samples(areas: np.ndarray, total_points: int) -> np.ndarray:
    if len(areas) > total_points:
        raise ValueError(
            f"The CAD assembly has {len(areas)} separate mesh parts, "
            f"more than the requested {total_points} points."
        )

    remaining = total_points - len(areas)
    exact_extra = remaining * areas / areas.sum()
    extra = np.floor(exact_extra).astype(np.int64)
    counts = extra + 1
    remainder = total_points - int(counts.sum())
    if remainder:
        order = np.argsort(-(exact_extra - extra))
        counts[order[:remainder]] += 1
    return counts


def convert_mesh_files_to_point_cloud(
    files: Sequence[Tuple[str, bytes]], num_points: int
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Sample mesh surfaces into [xyz, normals, part_id] with exactly N points."""
    if num_points not in (2048, 4096):
        raise ValueError("Point count must be either 2048 or 4096.")
    if not files:
        raise ValueError("Upload at least one CAD mesh file.")

    meshes = []
    source_files = []
    for filename, data in files:
        file_meshes = _meshes_from_upload(filename, data)
        meshes.extend(file_meshes)
        source_files.append(filename)

    areas = np.asarray([mesh.area for mesh in meshes], dtype=np.float64)
    valid_areas = np.isfinite(areas) & (areas > 0)
    meshes = [mesh for mesh, valid in zip(meshes, valid_areas) if valid]
    areas = areas[valid_areas]
    if not len(meshes):
        raise ValueError("The uploaded CAD files have no finite, positive-area surfaces.")

    sample_counts = _allocate_surface_samples(areas, num_points)
    point_clouds = []
    for part_id, (mesh, count) in enumerate(zip(meshes, sample_counts)):
        points, face_indices = trimesh.sample.sample_surface(mesh, int(count))
        normals = mesh.face_normals[face_indices]
        part_ids = np.full((int(count), 1), part_id, dtype=np.float32)
        point_clouds.append(np.concatenate(
            [points.astype(np.float32), normals.astype(np.float32), part_ids], axis=1))

    point_cloud = np.concatenate(point_clouds, axis=0)
    info = {
        "num_points": int(len(point_cloud)),
        "num_parts": len(meshes),
        "source_files": source_files,
        "surface_area": float(areas.sum()),
    }
    return point_cloud, info


def _combine_uploaded_part(filename: str, data: bytes) -> trimesh.Trimesh:
    meshes = _meshes_from_upload(filename, data)
    mesh = trimesh.util.concatenate(meshes)
    if not mesh.is_watertight:
        raise ValueError(
            f"'{filename}' is not a watertight solid mesh. Repair or re-export it "
            "as a closed mesh before automatic fitting."
        )
    return mesh


def _filled_voxel_centers(mesh: trimesh.Trimesh, pitch: float) -> np.ndarray:
    voxel_grid = mesh.voxelized(pitch=pitch).fill()
    return voxel_grid.points.astype(np.float64)


def _rigid_fit_candidates(
    part_a: trimesh.Trimesh,
    part_b: trimesh.Trimesh,
    candidate_count: int = 3,
) -> Tuple[List[Dict[str, object]], float]:
    """Search rigid poses that favor opposing surface contact without overlap."""
    center_a = part_a.centroid
    center_b = part_b.centroid
    scale = max(
        float(np.linalg.norm(part_a.extents)),
        float(np.linalg.norm(part_b.extents)),
        1e-6,
    )
    voxel_pitch = scale / 32.0
    voxels_a = _filled_voxel_centers(part_a, voxel_pitch) - center_a
    voxels_b = _filled_voxel_centers(part_b, voxel_pitch) - center_b

    while max(len(voxels_a), len(voxels_b)) > 30000:
        voxel_pitch *= 1.25
        voxels_a = _filled_voxel_centers(part_a, voxel_pitch) - center_a
        voxels_b = _filled_voxel_centers(part_b, voxel_pitch) - center_b

    surface_a, faces_a = trimesh.sample.sample_surface(part_a, 768, seed=17)
    surface_b, faces_b = trimesh.sample.sample_surface(part_b, 768, seed=23)
    surface_a -= center_a
    surface_b -= center_b
    normals_a = part_a.face_normals[faces_a]
    normals_b = part_b.face_normals[faces_b]
    surface_tree_a = cKDTree(surface_a)
    voxel_tree_a = cKDTree(voxels_a)
    voxel_tree_b = cKDTree(voxels_b)
    contact_tolerance = max(2.0 * voxel_pitch, 0.02 * scale)

    def evaluate(parameters: np.ndarray, include_metrics: bool = False):
        rotation = Rotation.from_euler("xyz", parameters[:3]).as_matrix()
        translation = parameters[3:]

        moved_surface_b = surface_b @ rotation.T + translation
        distances, nearest_a = surface_tree_a.query(moved_surface_b, workers=-1)
        normal_alignment = np.einsum(
            "ij,ij->i", normals_b @ rotation.T, normals_a[nearest_a])
        contact_strength = (
            np.clip(1.0 - distances / contact_tolerance, 0.0, 1.0)
            * np.clip(-normal_alignment, 0.0, 1.0)
        )

        moved_voxels_b = voxels_b @ rotation.T + translation
        overlap_ab = voxel_tree_a.query(moved_voxels_b, workers=-1)[0]
        voxels_a_in_b = (voxels_a - translation) @ rotation
        overlap_ba = voxel_tree_b.query(voxels_a_in_b, workers=-1)[0]
        overlap_fraction = max(
            float(np.mean(overlap_ab < voxel_pitch * 0.55)),
            float(np.mean(overlap_ba < voxel_pitch * 0.55)),
        )
        contact_score = float(contact_strength.mean())
        score = 100.0 * overlap_fraction - 2.0 * contact_score

        if not include_metrics:
            return score

        opposing = normal_alignment < -0.3
        gap_samples = distances[opposing] if opposing.any() else distances
        contact = (distances < contact_tolerance) & (normal_alignment < -0.5)
        return score, {
            "voxel_overlap_percent": 100.0 * overlap_fraction,
            "opposing_contact_percent": 100.0 * float(contact.mean()),
            "surface_gap_p05": float(np.percentile(gap_samples, 5)),
            "voxel_resolution": float(voxel_pitch),
            "contact_tolerance": float(contact_tolerance),
        }

    bounds = [(-np.pi, np.pi)] * 3 + [(-scale, scale)] * 3
    covariance_a = np.cov((part_a.vertices - center_a).T)
    covariance_b = np.cov((part_b.vertices - center_b).T)
    basis_a = np.linalg.eigh(covariance_a)[1][:, ::-1]
    basis_b = np.linalg.eigh(covariance_b)[1][:, ::-1]
    if np.linalg.det(basis_a) < 0:
        basis_a[:, -1] *= -1
    if np.linalg.det(basis_b) < 0:
        basis_b[:, -1] *= -1

    seed_candidates = []
    target_gap = voxel_pitch * 0.75
    for axis_permutation in permutations(range(3)):
        for axis_signs in product((-1.0, 1.0), repeat=3):
            signed_permutation = np.zeros((3, 3), dtype=np.float64)
            signed_permutation[np.arange(3), axis_permutation] = axis_signs
            if np.linalg.det(signed_permutation) < 0:
                continue

            rotation = basis_a @ signed_permutation @ basis_b.T
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Gimbal lock detected")
                angles = Rotation.from_matrix(rotation).as_euler("xyz")
            rotated_b = surface_b @ rotation.T
            for contact_axis in range(3):
                for contact_sign in (-1.0, 1.0):
                    translation_components = np.empty(3, dtype=np.float64)
                    for axis_index, axis in enumerate(basis_a.T):
                        projected_a = surface_a @ axis
                        projected_b = rotated_b @ axis
                        if axis_index == contact_axis and contact_sign > 0:
                            translation_components[axis_index] = (
                                projected_a.max() - projected_b.min() + target_gap)
                        elif axis_index == contact_axis:
                            translation_components[axis_index] = (
                                projected_a.min() - projected_b.max() - target_gap)
                        else:
                            translation_components[axis_index] = (
                                projected_a.mean() - projected_b.mean())
                    translation = basis_a @ translation_components
                    parameters = np.concatenate([angles, translation])
                    seed_candidates.append((evaluate(parameters), parameters))

    seed_candidates.sort(key=lambda candidate: candidate[0])
    population_size = 48
    initial_population = [
        parameters for _, parameters in seed_candidates[:population_size]
    ]
    rng = np.random.default_rng(17)
    while len(initial_population) < population_size:
        initial_population.append(np.asarray([
            rng.uniform(low, high) for low, high in bounds
        ]))

    result = differential_evolution(
        evaluate,
        bounds,
        maxiter=8,
        popsize=8,
        tol=0.02,
        seed=17,
        init=np.asarray(initial_population),
        polish=False,
        workers=1,
        updating="immediate",
    )

    ordered = [int(np.argmin(result.population_energies))]
    ordered.extend(np.argsort(result.population_energies).tolist())
    candidates = []
    for index in ordered:
        parameters = result.population[index]
        if any(
            np.linalg.norm(parameters[3:] - existing[3:]) < scale * 0.08
            and np.linalg.norm(parameters[:3] - existing[:3]) < 0.15
            for existing in [candidate["parameters"] for candidate in candidates]
        ):
            continue

        score, metrics = evaluate(parameters, include_metrics=True)
        rotation = Rotation.from_euler("xyz", parameters[:3]).as_matrix()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = center_a + parameters[3:] - rotation @ center_b

        candidates.append({
            "parameters": parameters.copy(),
            "score": float(score),
            "transform": transform,
            "metrics": metrics,
        })
        if len(candidates) >= candidate_count:
            break

    return candidates, voxel_pitch


def auto_fit_mesh_pair(
    filename_a: str,
    data_a: bytes,
    filename_b: str,
    data_b: bytes,
    num_points: int = 2048,
) -> Dict[str, object]:
    """Suggest rigid placements for two closed mesh parts and sample the result."""
    if num_points not in (2048, 4096):
        raise ValueError("Point count must be either 2048 or 4096.")

    part_a = _combine_uploaded_part(filename_a, data_a)
    part_b = _combine_uploaded_part(filename_b, data_b)
    candidates, voxel_pitch = _rigid_fit_candidates(part_a, part_b)

    for candidate in candidates:
        fitted_b = part_b.copy()
        fitted_b.apply_transform(candidate["transform"])
        counts = _allocate_surface_samples(
            np.asarray([part_a.area, fitted_b.area]), num_points)
        cloud_parts = []
        for part_id, (mesh, count) in enumerate(zip((part_a, fitted_b), counts)):
            points, face_indices = trimesh.sample.sample_surface(mesh, int(count))
            normals = mesh.face_normals[face_indices]
            ids = np.full((int(count), 1), part_id, dtype=np.float32)
            cloud_parts.append(np.concatenate(
                [points.astype(np.float32), normals.astype(np.float32), ids], axis=1))
        candidate["point_cloud"] = np.concatenate(cloud_parts, axis=0)

    return {
        "candidates": candidates,
        "num_points": num_points,
        "source_files": [filename_a, filename_b],
        "voxel_resolution": voxel_pitch,
    }
