use crate::{
    layers::{
        layer::Layer,
        conv::Conv2d
    },
    func::functions::nearest_neighbour_interpolation
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