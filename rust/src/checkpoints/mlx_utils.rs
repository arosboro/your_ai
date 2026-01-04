// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Your AI Project
//
// Utility functions for converting between MLX Array and flat representations

use mlx_rs::Array;

/// Converts an MLX Array to a flat representation (data + shape)
/// This is useful for serialization and checkpointing
pub fn to_flat(array: &Array) -> (Vec<f32>, Vec<i32>) {
    let data = array.as_slice::<f32>().to_vec();
    let shape = array.shape().to_vec();
    (data, shape)
}

/// Converts flat data and shape back to an MLX Array
pub fn from_flat(data: &[f32], shape: &[i32]) -> Array {
    // Create a new array from the flat data with the specified shape
    mlx_rs::Array::from_slice(data, shape)
}
