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
    pub fn Upsample2D_constr(
        in_channels : usize,
        out_channels : usize,
        padding : i32,
        stride: i32,
        kernel_size : usize,
        kernel_weights : Vec<f32>
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(Conv2d{in_channels : in_channels, out_channels : out_channels, padding : padding, stride:stride,  kernel_size : kernel_size, kernel_weights : kernel_weights}));
        Self { operations: vec }
    }
}

impl Layer for Upsample2D {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let (input_tensor, input_tensor_shape) = args;
        let (near_vec, near_vec_shape) = nearest_neighbour_interpolation((input_tensor, input_tensor_shape))?;
        let (res_vec, res_vec_shape) = &self.operations[0].operation((near_vec, near_vec_shape))?;
        Ok((res_vec.to_vec(), res_vec_shape.to_vec()))
    }
}

#[test]
fn test_upsample() {
    let (kernel_weights, _) = input(r"C:\study\coursework\src\trash\test_upsample_conv.safetensors".to_string()).unwrap();
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_upsample_inp.safetensors".to_string()).unwrap();
    let upsample = Upsample2D::Upsample2D_constr(640, 640, 1, 1, 3, kernel_weights.to_vec());
    let (res_vec, res_vec_shape) = upsample.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_upsample_outp.safetensors".to_string()).unwrap();
    assert! (res_vec_shape == py_vec_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-03)
    }
}