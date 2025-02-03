use crate::layers::{
        layer::Layer,
        conv::Conv2d
        };

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