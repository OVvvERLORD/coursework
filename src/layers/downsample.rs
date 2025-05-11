use crate::layers::{
        layer::Layer,
        conv::Conv2d
        };
use crate::func::functions::input;

pub struct DownSample2D {
    pub operations : Vec<Box<dyn Layer>>,
}

impl DownSample2D {
    pub fn new(
        in_channels : usize,
        out_channels : usize,
        padding : i32,
        stride : i32,
        kernel_size : usize,
        kernel_weights : Vec<f32>,
        bias: ndarray::Array4<f32>,
        is_bias: bool
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(Conv2d{
            in_channels : in_channels, 
            out_channels : out_channels, 
            padding : padding, 
            stride : stride, 
            kernel_size : kernel_size, 
            kernel_weights : kernel_weights, 
            bias: bias, 
            is_bias: is_bias}));
        Self { operations: vec }
    }
}

impl Layer for DownSample2D {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        for layer in operations {
            let _ = layer.operation(args)?;
        } 
        Ok(())
    }
}

// #[test]
// fn test_downsample_unbiased() {
    // let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_upsample_inp.safetensors".to_string()).unwrap();
    // let (kernel_weights, shape) = input(r"C:\study\coursework\src\trash\test_downsample_conv.safetensors".to_string()).unwrap();
    // let downsample2d = DownSample2D::DownSample2D_constr(640, 640, 1, 2, 3, kernel_weights.to_vec());
    // let (res_vec, res_vec_shape) = downsample2d.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    // let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_downsample_outp.safetensors".to_string()).unwrap();
//     assert!(res_vec_shape == py_vec_shape.to_vec());
//     for i in 0..res_vec.len() {
//         assert!((res_vec[i] - py_vec[i]).abs() <= 1e-04);
//     }
// }

#[test]
fn test_downsample_bias() {
    let mut tensor = input(r"C:\study\coursework\src\trash\test_downsample_inp.safetensors".to_string()).unwrap();
    let kernel = input(r"C:\study\coursework\src\trash\test_downsample_conv.safetensors".to_string()).unwrap();
    let bias = input(r"C:\study\coursework\src\trash\test_downsample_conv_b.safetensors".to_string()).unwrap();
    let downsample2d = DownSample2D::new(640, 640, 1, 2, 3, kernel.into_raw_vec_and_offset().0, bias, true);
    let _ = downsample2d.operation(&mut tensor);
    let py_tensor = input(r"C:\study\coursework\src\trash\test_downsample_outp.safetensors".to_string()).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-05);
                }
            }
        }
    }
}