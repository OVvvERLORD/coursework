use std::rc::Rc;
use std::cell::RefCell;

pub struct Transformer2D_params {
    pub number_of_groups: usize, pub eps: f32, pub gamma: f32, pub beta: f32,
    pub weigths_in: Vec<f32>, pub weights_shape_in: Vec<usize>, pub bias_in: Vec<f32>, pub bias_shape_in : Vec<usize>, pub is_bias_in : bool,
    pub weigths_out: Vec<f32>, pub weights_shape_out: Vec<usize>, pub bias_out: Vec<f32>, pub bias_shape_out : Vec<usize>, pub is_bias_out : bool,
    pub params_for_basics_vec : Vec<BasicTransofmerBlock_params>,
}

pub struct BasicTransofmerBlock_params {
    pub eps_1 : f32, pub gamma_1 : f32, pub beta_1 : f32, pub number_1 : usize, // LayerNorm 
    pub eps_2 : f32, pub gamma_2 : f32, pub beta_2 : f32, pub number_2 : usize, // LayerNorm 
    pub eps_3 : f32, pub gamma_3 : f32, pub beta_3 : f32, pub number_3 : usize, // LayerNorm 
    pub weigths_1: Vec<f32>, pub weights_shape_1 : Vec<usize>, pub bias_1: Vec<f32>, pub bias_shape_1 : Vec<usize>, pub is_bias_1 : bool,  // Attn1
    pub weigths_2: Vec<f32>, pub weights_shape_2 : Vec<usize>, pub bias_2: Vec<f32>, pub bias_shape_2 : Vec<usize>, pub is_bias_2 : bool,
    pub weigths_3: Vec<f32>, pub weights_shape_3 : Vec<usize>, pub bias_3: Vec<f32>, pub bias_shape_3 : Vec<usize>, pub is_bias_3 : bool,
    pub weigths_4: Vec<f32>, pub weights_shape_4 : Vec<usize>, pub bias_4: Vec<f32>, pub bias_shape_4 : Vec<usize>, pub is_bias_4 : bool,
    pub encoder_hidden_tensor_1 : Rc<RefCell<(Vec<f32>, Vec<usize>)>>, pub if_encoder_tensor_1 : bool, pub number_of_heads_1: usize, 

    pub weigths_5: Vec<f32>, pub weights_shape_5 : Vec<usize>, pub bias_5: Vec<f32>, pub bias_shape_5 : Vec<usize>, pub is_bias_5 : bool, // Attn2
    pub weigths_6: Vec<f32>, pub weights_shape_6 : Vec<usize>, pub bias_6: Vec<f32>, pub bias_shape_6 : Vec<usize>, pub is_bias_6 : bool,
    pub weigths_7: Vec<f32>, pub weights_shape_7 : Vec<usize>, pub bias_7: Vec<f32>, pub bias_shape_7 : Vec<usize>, pub is_bias_7 : bool,
    pub weigths_8: Vec<f32>, pub weights_shape_8 : Vec<usize>, pub bias_8: Vec<f32>, pub bias_shape_8 : Vec<usize>, pub is_bias_8 : bool,
    pub encoder_hidden_tensor_2 : Rc<RefCell<(Vec<f32>, Vec<usize>)>>, pub if_encoder_tensor_2 : bool, pub number_of_heads_2: usize, 

    pub weigths_ff1: Vec<f32>, pub weights_shape_ff1 : Vec<usize>, pub bias_ff1: Vec<f32>, pub bias_shape_ff1 : Vec<usize>, pub is_bias_ff1 : bool, // FeedForward
    pub weigths_ff2: Vec<f32>, pub weights_shape_ff2 : Vec<usize>, pub bias_ff2: Vec<f32>, pub bias_shape_ff2 : Vec<usize>, pub is_bias_ff2 : bool,
}

pub struct Resnet2d_params {
    pub number_of_groups_1 : usize, pub eps_1: f32, pub gamma_1: f32, pub beta_1: f32,
    pub in_channels_1 : usize, pub out_channels_1 : usize, pub padding_1 : i32, pub stride_1 : i32, pub kernel_size_1 : usize, pub kernel_weights_1 : Vec<f32>,
    pub weigths: Vec<f32>, pub weights_shape : Vec<usize>, pub bias: Vec<f32>, pub bias_shape : Vec<usize>, pub is_bias : bool,
    pub number_of_groups_2 : usize, pub eps_2: f32, pub gamma_2: f32, pub beta_2: f32,
    pub in_channels_2 : usize, pub out_channels_2 : usize, pub padding_2 : i32,  pub stride_2 : i32, pub kernel_size_2 : usize, pub kernel_weights_2 : Vec<f32>,
    pub is_shortcut : bool,
    pub in_channels_short : usize, pub out_channels_short : usize, pub padding_short : i32,  pub stride_short : i32, pub kernel_size_short : usize, pub kernel_weights_short: Vec<f32>,
    // pub time_emb : Vec<f32>, pub time_emb_shape : Vec<usize>,
    pub time_emb : Rc<RefCell<(Vec<f32>, Vec<usize>)>>
}

pub struct CrossAttnUpBlock2D_params {
    pub params_for_transformer1 : Transformer2D_params,
    pub params_for_transformer2 : Transformer2D_params,
    pub params_for_transformer3 : Transformer2D_params,
    pub params_for_resnet1 : Resnet2d_params,
    pub params_for_resnet2 : Resnet2d_params,
    pub params_for_resnet3 : Resnet2d_params,
    pub in_channels: usize, pub out_channels: usize, pub padding: i32, pub stride : i32, pub kernel_size: usize, pub kernel_weights: Vec<f32>,
    pub hidden_states : Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>
}

pub struct CrossAttnDownBlock2D_params {
    pub is_downsample2d: bool,
    pub params_for_transformer1 : Transformer2D_params,
    pub params_for_transformer2 : Transformer2D_params,
    pub params_for_resnet1 : Resnet2d_params,
    pub params_for_resnet2: Resnet2d_params,
    pub in_channels : usize, pub out_channels : usize, pub padding : i32, pub stride : i32, pub kernel_size : usize, pub kernel_weights : Vec<f32>,
    pub hidden_states : Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>
}