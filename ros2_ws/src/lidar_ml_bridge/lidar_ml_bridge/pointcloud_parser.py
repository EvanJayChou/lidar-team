"""PointCloud2 parsing utilities separated from ROS 2 node logic.

Provides:
  - ParsedCloud dataclass
  - parse_pointcloud() orchestrating selection of numpy vs iterative path
  - extract_with_numpy() and extract_iterative() helpers
  - pointfield_to_dtype() mapping

All functions are pure (no ROS dependencies) to enable standalone testing and reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from sensor_msgs.msg import PointCloud2, PointField

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None

XYZ_FIELDS = ("x", "y", "z")
OPTIONAL_FIELDS = ("intensity",)


@dataclass
class ParsedCloud:
    xyz: Sequence[Tuple[float, float, float]]  # or numpy array shape (N,3)
    intensity: Optional[Sequence[float]] = None
    frame_id: str = ""
    stamp_sec: int = 0
    stamp_nanosec: int = 0


def parse_pointcloud(msg: PointCloud2, keep_fields: Optional[List[str]] = None, use_numpy: bool = True) -> ParsedCloud:
    field_names = [f.name for f in msg.fields]
    if keep_fields:
        active_xyz_fields = [f for f in XYZ_FIELDS if f in keep_fields and f in field_names]
    else:
        active_xyz_fields = [f for f in XYZ_FIELDS if f in field_names]
    has_intensity = ("intensity" in field_names) and (not keep_fields or "intensity" in keep_fields)

    if not active_xyz_fields:
        return ParsedCloud([], None, msg.header.frame_id, msg.header.stamp.sec, msg.header.stamp.nanosec)

    if use_numpy and np is not None:
        points_xyz, intensities = extract_with_numpy(msg, active_xyz_fields, has_intensity)
    else:
        points_xyz, intensities = extract_iterative(msg, active_xyz_fields, has_intensity)

    return ParsedCloud(
        xyz=points_xyz,
        intensity=intensities,
        frame_id=msg.header.frame_id,
        stamp_sec=msg.header.stamp.sec,
        stamp_nanosec=msg.header.stamp.nanosec,
    )


def extract_with_numpy(msg: PointCloud2, xyz_fields: List[str], has_intensity: bool):  # type: ignore[override]
    dtype_fields = []
    for f in msg.fields:
        if f.name in xyz_fields or (has_intensity and f.name == "intensity"):
            np_dtype = pointfield_to_dtype(f)
            dtype_fields.append((f.name, np_dtype))
    if not dtype_fields:
        return [], None
    buf = bytes(msg.data)
    arr = np.frombuffer(buf, dtype=dtype_fields)
    points_xyz = np.vstack([arr[f] for f in xyz_fields]).T  # shape (N,3)
    intensities = arr["intensity"] if has_intensity and "intensity" in arr.dtype.names else None
    return points_xyz, intensities


def extract_iterative(msg: PointCloud2, xyz_fields: List[str], has_intensity: bool):
    from struct import unpack
    fmt = "<" + "f" * (len(xyz_fields) + (1 if has_intensity else 0))
    point_step = msg.point_step
    data = msg.data
    n = msg.width * msg.height
    points_xyz = []
    intensities = [] if has_intensity else None
    for i in range(n):
        start = i * point_step
        chunk = data[start:start + point_step]
        values = unpack(fmt, chunk[:4 * (len(xyz_fields) + (1 if has_intensity else 0))])
        xyz_vals = values[:len(xyz_fields)]
        points_xyz.append(tuple(xyz_vals))
        if has_intensity and intensities is not None:
            intensities.append(values[len(xyz_fields)])
    return points_xyz, intensities


def pointfield_to_dtype(field: PointField):
    if field.datatype in (PointField.FLOAT32,):
        return "<f4"
    if field.datatype in (PointField.UINT16,):
        return "<u2"
    if field.datatype in (PointField.INT16,):
        return "<i2"
    if field.datatype in (PointField.UINT32,):
        return "<u4"
    if field.datatype in (PointField.INT32,):
        return "<i4"
    return f"V{field.count}"  # opaque bytes fallback
