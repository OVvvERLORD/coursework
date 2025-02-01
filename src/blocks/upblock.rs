use crate::{
    layers::{
        params::Resnet2d_params,
        layer::Layer
    },
    blocks::resnet::Resnet2d
};

pub struct UpBlock2d {
    pub operations : Vec<Box<dyn Layer>>,
}

impl UpBlock2d {
    pub fn UpBlock2d_constr(
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        params_for_resnet3 : Resnet2d_params,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::Resnet2d_constr(params_for_resnet1);
        vec.push(Box::new(resnet1));
        let resnet2 = Resnet2d::Resnet2d_constr(params_for_resnet2);
        vec.push(Box::new(resnet2));
        let resnet3 = Resnet2d::Resnet2d_constr(params_for_resnet3);
        vec.push(Box::new(resnet3));
        Self { operations: vec }
    }
}

impl Layer for UpBlock2d {
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