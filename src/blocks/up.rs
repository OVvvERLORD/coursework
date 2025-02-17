use crate::{
    layers::{
        layer::Layer,
        params::{
            CrossAttnUpBlock2D_params,
            Resnet2d_params
        }
    },
    blocks::{
        attn::CrossAttnUpBlock2D,
        upblock::UpBlock2d
    }
};

pub struct Up_blocks {
    pub operations : Vec<Box<dyn Layer>>,
}

impl Up_blocks {
    pub fn Up_block_constr (
        params_for_crossblock1 : CrossAttnUpBlock2D_params,
        params_for_crossblock2 : CrossAttnUpBlock2D_params,
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        params_for_resnet3 : Resnet2d_params,
        hidden_states : Box<Vec::<(Vec<f32>, Vec<usize>)>>
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let crossAttnUpBlock1 = CrossAttnUpBlock2D::CrossAttnUpBlock2D_constr(params_for_crossblock1);
        vec.push(Box::new(crossAttnUpBlock1));
        let crossAttnUpBlock2 = CrossAttnUpBlock2D::CrossAttnUpBlock2D_constr(params_for_crossblock2);
        vec.push(Box::new(crossAttnUpBlock2));
        let upblock2d = UpBlock2d::UpBlock2d_constr(params_for_resnet1, params_for_resnet2, params_for_resnet3, hidden_states);
        vec.push(Box::new(upblock2d));
        Self { operations: vec }
    }
}

impl Layer for Up_blocks {
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