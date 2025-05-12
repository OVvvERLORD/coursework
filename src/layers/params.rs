use std::rc::Rc;
use std::cell::RefCell;

pub struct Transformer2D_params {
    pub number_of_groups: usize, pub eps: f32, pub gamma: ndarray::Array4<f32>, pub beta: ndarray::Array4<f32>,
    pub weigths_in: ndarray::Array4<f32>, pub bias_in: ndarray::Array4<f32>, pub is_bias_in : bool,
    pub weigths_out: ndarray::Array4<f32>, pub bias_out: ndarray::Array4<f32>, pub is_bias_out : bool,
    pub params_for_basics_vec : Vec<BasicTransofmerBlock_params>,
}

pub struct BasicTransofmerBlock_params {
    pub eps_1 : f32, pub gamma_1 : ndarray::Array4<f32>, pub beta_1 : ndarray::Array4<f32>, pub number_1 : usize, // LayerNorm 
    pub eps_2 : f32, pub gamma_2 : ndarray::Array4<f32>, pub beta_2 : ndarray::Array4<f32>, pub number_2 : usize, // LayerNorm 
    pub eps_3 : f32, pub gamma_3 : ndarray::Array4<f32>, pub beta_3 : ndarray::Array4<f32>, pub number_3 : usize, // LayerNorm 
    pub weights_1: ndarray::Array4<f32>, pub bias_1: ndarray::Array4<f32>, pub is_bias_1 : bool,  // Attn1
    pub weights_2: ndarray::Array4<f32>, pub bias_2: ndarray::Array4<f32>, pub is_bias_2 : bool,
    pub weights_3: ndarray::Array4<f32>, pub bias_3: ndarray::Array4<f32>, pub is_bias_3 : bool,
    pub weights_4: ndarray::Array4<f32>, pub bias_4: ndarray::Array4<f32>, pub is_bias_4 : bool,
    pub encoder_hidden_tensor_1 : Rc<RefCell<ndarray::Array3<f32>>>, pub if_encoder_tensor_1 : bool, pub number_of_heads_1: usize, 

    pub weights_5: ndarray::Array4<f32>, pub bias_5: ndarray::Array4<f32>, pub is_bias_5 : bool, // Attn2
    pub weights_6: ndarray::Array4<f32>, pub bias_6: ndarray::Array4<f32>, pub is_bias_6 : bool,
    pub weights_7: ndarray::Array4<f32>, pub bias_7: ndarray::Array4<f32>, pub is_bias_7 : bool,
    pub weights_8: ndarray::Array4<f32>, pub bias_8: ndarray::Array4<f32>, pub is_bias_8 : bool,
    pub encoder_hidden_tensor_2 : Rc<RefCell<ndarray::Array3<f32>>>, pub if_encoder_tensor_2 : bool, pub number_of_heads_2: usize, 

    pub weights_ff1: ndarray::Array4<f32>, pub bias_ff1: ndarray::Array4<f32>, pub is_bias_ff1 : bool, // FeedForward
    pub weights_ff2: ndarray::Array4<f32>, pub bias_ff2: ndarray::Array4<f32>, pub is_bias_ff2 : bool,
}

pub struct Resnet2d_params {
    pub number_of_groups_1 : usize, pub eps_1: f32, pub gamma_1: ndarray::Array4<f32>, pub beta_1: ndarray::Array4<f32>,

    pub in_channels_1 : usize, pub out_channels_1 : usize, pub padding_1 : i32, pub stride_1 : i32, pub kernel_size_1 : usize, pub kernel_weights_1 : Vec<f32>, 
    pub bias_c1: ndarray::Array4<f32>, pub is_bias_c1:bool,

    pub weights: ndarray::Array4<f32>, pub bias: ndarray::Array4<f32>, pub is_bias : bool,
    pub number_of_groups_2 : usize, pub eps_2: f32, pub gamma_2: ndarray::Array4<f32>, pub beta_2: ndarray::Array4<f32>,
    pub in_channels_2 : usize, pub out_channels_2 : usize, pub padding_2 : i32,  pub stride_2 : i32, pub kernel_size_2 : usize, pub kernel_weights_2 : Vec<f32>,
    pub bias_c2: ndarray::Array4<f32>, pub is_bias_c2:bool,
    pub is_shortcut : bool,
    pub in_channels_short : usize, pub out_channels_short : usize, pub padding_short : i32,  pub stride_short : i32, pub kernel_size_short : usize, pub kernel_weights_short: Vec<f32>,
    pub bias_s: ndarray::Array4<f32>, pub is_bias_s:bool,
    // pub time_emb : Vec<f32>, pub time_emb_shape : Vec<usize>,
    pub time_emb : Rc<RefCell<ndarray::Array4<f32>>>
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