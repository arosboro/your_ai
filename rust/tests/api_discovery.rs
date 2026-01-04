
#[test]
fn test_linear_api_discovery() {
    use mlx_rs::nn::Linear;
    let mut l = Linear::new(10, 10).unwrap();

    // Verify we can disable bias
    // Logic: Param<Option<Array>> implements DerefMut<Target=Option<Array>>
    // So checking *l.bias = None

    // We need to dereference the Param wrapper to set the inner Option to None
    *l.bias = None;
}
