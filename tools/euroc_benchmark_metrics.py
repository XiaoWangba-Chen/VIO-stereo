#!/usr/bin/env python3
"""Pure metric helpers shared by the EuRoC benchmark and its tests."""

import csv
import math
from pathlib import Path

import numpy as np


def percentile(values, quantile):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, quantile)) if values.size else float("nan")


def rmse(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else float("nan")


def vector_rmse(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(vectors * vectors, axis=1)))) if vectors.size else float("nan")


def umeyama(source, target, estimate_scale=False):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1] = -1.0
    rotation = vt.T @ np.diag(correction) @ u.T
    scale = 1.0
    if estimate_scale:
        variance = np.mean(np.sum(source_centered * source_centered, axis=1))
        scale = float(np.sum(singular_values * correction) / variance)
    translation = target_mean - scale * rotation @ source_mean
    aligned = scale * (rotation @ source.T).T + translation
    return aligned, rotation, translation, scale


def quat_normalize(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion / norm


def quat_multiply(left, right):
    lw, lx, ly, lz = np.moveaxis(np.asarray(left), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack((lw*rw-lx*rx-ly*ry-lz*rz,
                     lw*rx+lx*rw+ly*rz-lz*ry,
                     lw*ry-lx*rz+ly*rw+lz*rx,
                     lw*rz+lx*ry-ly*rx+lz*rw), axis=-1)


def quat_inverse(quaternion):
    result = np.array(quaternion, dtype=np.float64, copy=True)
    result[..., 1:] *= -1.0
    return quat_normalize(result)


def quat_rotate(quaternion, vector):
    quaternion = quat_normalize(quaternion)
    vector_quaternion = np.concatenate(([0.0], np.asarray(vector, dtype=np.float64)))
    return quat_multiply(quat_multiply(quaternion, vector_quaternion),
                         quat_inverse(quaternion))[1:]


def rotation_matrix_to_quat(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    w = math.sqrt(max(0.0, 1.0 + np.trace(matrix))) / 2.0
    if w > 1e-12:
        x = (matrix[2, 1] - matrix[1, 2]) / (4.0 * w)
        y = (matrix[0, 2] - matrix[2, 0]) / (4.0 * w)
        z = (matrix[1, 0] - matrix[0, 1]) / (4.0 * w)
        return quat_normalize([w, x, y, z])
    values, vectors = np.linalg.eigh(matrix)
    axis = vectors[:, np.argmax(values)]
    return quat_normalize([0.0, *axis])


def interpolate_quaternions(source_times, quaternions, target_times):
    source_times = np.asarray(source_times)
    quaternions = quat_normalize(quaternions)
    result = []
    for stamp in target_times:
        upper = int(np.searchsorted(source_times, stamp))
        if upper <= 0:
            result.append(quaternions[0])
            continue
        if upper >= len(source_times):
            result.append(quaternions[-1])
            continue
        lower = upper - 1
        amount = (stamp - source_times[lower]) / (source_times[upper] - source_times[lower])
        first, second = quaternions[lower], quaternions[upper]
        dot = float(np.dot(first, second))
        if dot < 0.0:
            second, dot = -second, -dot
        if dot > 0.9995:
            value = first + amount * (second - first)
        else:
            angle = math.acos(max(-1.0, min(1.0, dot)))
            value = (math.sin((1.0 - amount) * angle) * first +
                     math.sin(amount * angle) * second) / math.sin(angle)
        result.append(quat_normalize(value))
    return np.asarray(result)


def relative_pose_errors(timestamps, estimate_positions, estimate_quaternions,
                         truth_positions, truth_quaternions, delta=1.0, tolerance=0.06):
    translation_errors, rotation_errors = [], []
    for index, stamp in enumerate(timestamps):
        other = int(np.searchsorted(timestamps, stamp + delta))
        if other >= len(timestamps) or abs(timestamps[other] - stamp - delta) > tolerance:
            continue
        est_delta_p = quat_rotate(quat_inverse(estimate_quaternions[index]),
                                  estimate_positions[other] - estimate_positions[index])
        gt_delta_p = quat_rotate(quat_inverse(truth_quaternions[index]),
                                 truth_positions[other] - truth_positions[index])
        translation_errors.append(np.linalg.norm(est_delta_p - gt_delta_p))
        est_delta_q = quat_multiply(quat_inverse(estimate_quaternions[index]),
                                    estimate_quaternions[other])
        gt_delta_q = quat_multiply(quat_inverse(truth_quaternions[index]),
                                   truth_quaternions[other])
        error_q = quat_multiply(quat_inverse(gt_delta_q), est_delta_q)
        rotation_errors.append(2.0 * math.acos(min(1.0, abs(float(error_q[0])))))
    return np.asarray(translation_errors), np.asarray(rotation_errors)


def read_numeric_csv(path, columns):
    rows = []
    with Path(path).open() as stream:
        for row in csv.reader(stream):
            if row and not row[0].startswith("#"):
                rows.append([float(row[index]) for index in columns])
    return np.asarray(rows, dtype=np.float64)


def dropped_frames(expected, processed):
    return max(0, int(expected) - int(processed))
