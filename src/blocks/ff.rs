use crate::layers::{
        layer::Layer,
        linear::Linear,
        act::GeLU
};
use ndarray;

pub struct FeedForward {
    pub operations: Vec<Box<dyn Layer>>,
}
impl FeedForward {
    pub fn new (
        weights_1: ndarray::Array4<f32>,
        bias_1: ndarray::Array4<f32>,
        is_bias_1 : bool,
        weights_2: ndarray::Array4<f32>,
        bias_2: ndarray::Array4<f32>,
        is_bias_2 : bool,
    ) -> Self {
        let mut vec: Vec<Box<dyn Layer>> = Vec::new();
        vec.push(Box::new(Linear {weights: weights_1, bias: bias_1, is_bias: is_bias_1}));
        vec.push(Box::new(GeLU));
        vec.push(Box::new(Linear {weights: weights_2, bias: bias_2, is_bias: is_bias_2}));
        Self { operations: vec }
    }
}
impl Layer for FeedForward {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let _ = &self.operations[0].operation(args)?;
        let shape_after_lin = args.shape();
        let limit = shape_after_lin[3] / 2;

        let mut part_1 = args
        .slice_mut(ndarray::s![.., .., .., ..limit])
        .to_owned();

        let mut part_2= args
        .slice_mut(ndarray::s![.., .., .., limit..])
        .to_owned();

        let _ = &self.operations[1].operation(&mut part_2).unwrap();
        part_1 *= &part_2;
        let _ = &self.operations[2].operation(&mut part_1).unwrap();
        *args = part_1;
        Ok(())
    }
}

#[test]
fn ff_test_2x2x1280x1280() {
    use crate::func::functions::input;
    let mut tensor= input(r"C:\study\coursework\src\trash\ff_input.safetensors".to_string()).unwrap();
    let weights_1 = input(r"C:\study\coursework\src\trash\ff_lin1.safetensors".to_string()).unwrap();
    let bias_1 = input(r"C:\study\coursework\src\trash\ff_lin1_bias.safetensors".to_string()).unwrap();
    let bias_2 = input(r"C:\study\coursework\src\trash\ff_lin2_bias.safetensors".to_string()).unwrap();
    let weights_2 = input(r"C:\study\coursework\src\trash\ff_lin2.safetensors".to_string()).unwrap();
    let ff = FeedForward::new(weights_1, bias_1,  true, weights_2, bias_2, true);
    let _ = ff.operation(&mut tensor).unwrap();

    let py_tensor = input(r"C:\study\coursework\src\trash\ff_output.safetensors".to_string()).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-06);
                }
            }
        }
    }

}