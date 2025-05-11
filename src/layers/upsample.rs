use crate::{
    layers::{
        layer::Layer,
        conv::Conv2d
    },
    func::functions::{nearest_neighbour_interpolation, input}
};

pub struct Upsample2D {
    pub operations: Vec<Box<dyn Layer>>,
}

impl Upsample2D {
    pub fn new(
        in_channels : usize,
        out_channels : usize,
        padding : i32,
        stride: i32,
        kernel_size : usize,
        kernel_weights : Vec<f32>,
        bias: ndarray::Array4<f32>,
        is_bias: bool
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(
            Conv2d{
                in_channels : in_channels, 
                out_channels : out_channels, 
                padding : padding, 
                stride:stride,  
                kernel_size : kernel_size, 
                kernel_weights : kernel_weights,
                bias: bias,
                is_bias: is_bias
            }));
        Self { operations: vec }
    }
}

impl Layer for Upsample2D {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let _ = nearest_neighbour_interpolation(args)?;
        let _ = &self.operations[0].operation(args)?;
        Ok(())
    }
}

#[test]
fn test_upsample() {
    let kernel = input(r"C:\study\coursework\src\trash\test_upsample_conv.safetensors".to_string()).unwrap();
    let mut tensor= input(r"C:\study\coursework\src\trash\test_upsample_inp.safetensors".to_string()).unwrap();
    let upsample = Upsample2D::new(640, 640, 1, 1, 3, kernel.clone().into_raw_vec_and_offset().0, kernel, false);
    let _ = upsample.operation(&mut tensor).unwrap();

    let py_tensor = input(r"C:\study\coursework\src\trash\test_upsample_outp.safetensors".to_string()).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-04);
                }
            }
        }
    }
}

fn test_upsample_bias() {
    let mut tensor = input(r"C:\study\coursework\src\trash\test_upsample_inp.safetensors".to_string()).unwrap();
    let kernel = input(r"C:\study\coursework\src\trash\test_upsample_conv.safetensors".to_string()).unwrap();
    let bias = input(format!(r"C:\study\coursework\src\trash\test_upsample_b_conv.safetensors")).unwrap();
    let upsample = Upsample2D::new(640, 640, 1, 1, 3, kernel.into_raw_vec_and_offset().0, bias, true);
    let _ = upsample.operation(&mut tensor).unwrap();

    let py_tensor = input(r"C:\study\coursework\src\trash\test_upsample_outp.safetensors".to_string()).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-04);
                }
            }
        }
    }
}