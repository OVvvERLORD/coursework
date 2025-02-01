use crate::{
    layers::{
        layer::Layer,
        params::{
            Resnet2d_params,
            CrossAttnDownBlock2D_params
        },
    },
    
    blocks::{
        downblock::DownBlock2D,
        attn::CrossAttnDownBlock2D
    }
};

pub struct Down_blocks {
    pub operations : Vec<Box<dyn Layer>>,
}

impl Down_blocks {
    pub fn Down_blocks_constr (
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        in_channels : usize, out_channels : usize, padding : i32, stride : i32, kernel_size : usize, kernel_weights : Vec<f32>,
        params_for_crattbl1 : CrossAttnDownBlock2D_params,
        params_for_crattbl2 : CrossAttnDownBlock2D_params,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let downblock = DownBlock2D::DownBlock2D_constr(params_for_resnet1, params_for_resnet2, in_channels, out_channels, padding, stride, kernel_size, kernel_weights);
        vec.push(Box::new(downblock));
        let crossattnblock1 = CrossAttnDownBlock2D::CrossAttnDownBlock2D_constr(params_for_crattbl1);
        vec.push(Box::new(crossattnblock1));
        let crossattnblock2 = CrossAttnDownBlock2D::CrossAttnDownBlock2D_constr(params_for_crattbl2);
        vec.push(Box::new(crossattnblock2));
        Self { operations: vec }
    }
}

impl Layer for Down_blocks {
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