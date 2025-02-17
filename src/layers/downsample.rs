use crate::layers::{
        layer::Layer,
        conv::Conv2d
        };
use crate::func::functions::input;

pub struct DownSample2D {
    pub operations : Vec<Box<dyn Layer>>,
}

impl DownSample2D {
    pub fn DownSample2D_constr(
        in_channels : usize,
        out_channels : usize,
        padding : i32,
        stride : i32,
        kernel_size : usize,
        kernel_weights : Vec<f32>,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(Conv2d{in_channels : in_channels, out_channels : out_channels, padding : padding, stride : stride, kernel_size : kernel_size, kernel_weights : kernel_weights}));
        Self { operations: vec }
    }
}

impl Layer for DownSample2D {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut res_vec = args.0;
        let mut res_vec_shape = args.1;
        for layer in operations {
            let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
        } 
        Ok((res_vec, res_vec_shape))
    }
}

#[test]
fn test_downsample_unbiased() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_upsample_inp.safetensors".to_string()).unwrap();
    let (kernel_weights, shape) = input(r"C:\study\coursework\src\trash\test_downsample_conv.safetensors".to_string()).unwrap();
    let downsample2d = DownSample2D::DownSample2D_constr(640, 640, 1, 2, 3, kernel_weights.to_vec());
    let (res_vec, res_vec_shape) = downsample2d.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_downsample_outp.safetensors".to_string()).unwrap();
    assert!(res_vec_shape == py_vec_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!((res_vec[i] - py_vec[i]).abs() <= 1e-04);
    }
}