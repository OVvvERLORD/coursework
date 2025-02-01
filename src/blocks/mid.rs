use crate::{
    layers::{
        params::{
            Transformer2D_params,
            Resnet2d_params
        },
        layer::Layer,
    },
    blocks::{
        resnet::Resnet2d,
        trans::Transformer2D
    }
};

pub struct mid_block {
    pub operations : Vec<Box<dyn Layer>>,
}

impl mid_block {
    pub fn mid_block_constr (
        params_for_transformer2d :Transformer2D_params,
        params_for_resnet_1: Resnet2d_params,
        params_for_resnet_2: Resnet2d_params
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::Resnet2d_constr(params_for_resnet_1);
        vec.push(Box::new(resnet1));
        let transformer = Transformer2D::Transformer2D_constr(
            params_for_transformer2d);
        vec.push(Box::new(transformer));
        let resnet2 = Resnet2d::Resnet2d_constr(params_for_resnet_2);
        vec.push(Box::new(resnet2));
        Self { operations: vec }
    }
}
impl Layer for mid_block {
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