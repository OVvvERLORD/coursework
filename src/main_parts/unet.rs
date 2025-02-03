use crate::{
    layers::{
        layer::Layer,
        params::{
            Resnet2d_params,
            CrossAttnDownBlock2D_params,
            CrossAttnUpBlock2D_params,
            Transformer2D_params
        },
        linear::Linear,
        conv::Conv2d,
        norm::GroupNorm,
        act::SiLU,
    },
    blocks::{
        down::Down_blocks,
        up::Up_blocks,
        mid::mid_block
    }
};

pub struct Unet2dConditionModel {
    pub operations : Vec<Box<dyn Layer>>,
    pub time_emb : Vec<f32>,
    pub time_emb_shape : Vec<usize>,
}

impl Unet2dConditionModel {
    pub fn Unet2dConditionModel_constr(
        time_emb : Vec<f32>, time_emb_shape : Vec<usize>, // time
        weigths1: Vec<f32>, weights_shape1 : Vec<usize>, bias1: Vec<f32>, bias_shape1 : Vec<usize>, is_bias1 : bool,
        weigths2 : Vec<f32>, weights_shape2 : Vec<usize>, bias2: Vec<f32>, bias_shape2 : Vec<usize>, is_bias2 : bool,

        in_channels_in : usize, out_channels_in: usize, padding_in : i32, stride_in : i32, kernel_size_in : usize, kernel_weights_in : Vec<f32>, //in

        params_for_resnet1_down : Resnet2d_params, // down
        params_for_resnet2_down : Resnet2d_params,
        in_channels_down : usize, out_channels_down : usize, padding_down : i32, stride_down : i32, kernel_size_down : usize, kernel_weights_down : Vec<f32>,
        params_for_crattbl1 : CrossAttnDownBlock2D_params,
        params_for_crattbl2 : CrossAttnDownBlock2D_params,

        params_for_crossblock1 : CrossAttnUpBlock2D_params, // up
        params_for_crossblock2 : CrossAttnUpBlock2D_params,
        params_for_resnet1_up : Resnet2d_params,
        params_for_resnet2_up : Resnet2d_params,
        params_for_resnet3_up : Resnet2d_params,

        params_for_transformer2d :Transformer2D_params, // mid
        params_for_resnet_1_mid: Resnet2d_params,
        params_for_resnet_2_mid: Resnet2d_params,

        number_of_groups_out: usize, eps_out: f32, gamma_out: f32, beta_out: f32, //out
        in_channels_out : usize, out_channels_out: usize, padding_out : i32, stride_out : i32, kernel_size_out : usize, kernel_weights_out : Vec<f32>,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(Linear {weigths : weigths1, weights_shape : weights_shape1, bias : bias1, bias_shape : bias_shape1, is_bias : is_bias1}));
        vec.push(Box::new(Linear {weigths : weigths2, weights_shape : weights_shape2, bias : bias2, bias_shape : bias_shape2, is_bias : is_bias2}));
        vec.push(Box::new(Conv2d{in_channels : in_channels_in, out_channels : out_channels_in, padding : padding_in, stride : stride_in, kernel_size : kernel_size_in, kernel_weights : kernel_weights_in}));
        let down = Down_blocks::Down_blocks_constr(params_for_resnet1_down, params_for_resnet2_down, in_channels_down, out_channels_down, padding_down, stride_down, kernel_size_down, kernel_weights_down, params_for_crattbl1, params_for_crattbl2);
        vec.push(Box::new(down));
        let mid = mid_block::mid_block_constr(params_for_transformer2d, params_for_resnet_1_mid, params_for_resnet_2_mid);
        vec.push(Box::new(mid));
        let up = Up_blocks::Up_block_constr(params_for_crossblock1, params_for_crossblock2, params_for_resnet1_up, params_for_resnet2_up, params_for_resnet3_up);
        vec.push(Box::new(up));
        vec.push(Box::new(GroupNorm{number_of_groups : number_of_groups_out, eps : eps_out, gamma : gamma_out, beta: beta_out}));
        vec.push(Box::new(SiLU));
        vec.push(Box::new(Conv2d{in_channels: in_channels_out, out_channels : out_channels_out, padding : padding_out, stride: stride_out, kernel_size : kernel_size_out, kernel_weights : kernel_weights_out}));
        Self { operations: vec, time_emb: time_emb, time_emb_shape: time_emb_shape }
    }
}

impl Layer for Unet2dConditionModel {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let (temp_emb_vec, temp_emb_shape) = operations[0].operation((self.time_emb.clone(), self.time_emb_shape.clone()))?;
        let (time_emb_vec, time_emb_shape) = operations[1].operation((temp_emb_vec.clone(), temp_emb_shape.clone()))?;
        let mut res_vec = args.0;
        let mut res_vec_shape = args.1;
        for i in 2..operations.len() {
            let (temp_vec, temp_vec_shape) = operations[i].operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
        } 
        Ok((res_vec, res_vec_shape))
    }
}
