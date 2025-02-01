use crate::{
    layers::{
        layer::Layer,
        params::Resnet2d_params,
        downsample::DownSample2D
    },
    blocks::{
        resnet::Resnet2d,
    }
};

pub struct DownBlock2D {
    pub operations : Vec<Box<dyn Layer>>,
}

impl DownBlock2D {
    pub fn DownBlock2D_constr(
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        in_channels : usize, out_channels : usize, padding : i32, stride : i32, kernel_size : usize, kernel_weights : Vec<f32>,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::Resnet2d_constr(params_for_resnet1);
        vec.push(Box::new(resnet1));
        let resnet2 = Resnet2d::Resnet2d_constr(params_for_resnet2);
        vec.push(Box::new(resnet2));
        let downsample = DownSample2D::DownSample2D_constr(in_channels, out_channels, padding, stride, kernel_size, kernel_weights);
        vec.push(Box::new(downsample));
        Self { operations: vec }
    }
}

impl Layer for DownBlock2D {
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