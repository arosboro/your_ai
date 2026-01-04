use mlx_rs::Array;

fn main() {
    let a = Array::from_slice(&[1, 2, 3, 4], &[4]);
    let _s = mlx_rs::ops::slice(&a, &[0], &[2], &[1]);
}
