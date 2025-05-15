use crate::{
    blocks::{
        down::Down_blocks, mid::mid_block, resnet::Resnet2d, trans::Transformer2D, up::Up_blocks
    }, func::functions::{aug_emb, input, scalar_timestep_embedding}, layers::{
        act::SiLU, conv::Conv2d, layer::Layer, linear::Linear, norm::GroupNorm, params::{
            BasicTransofmerBlock_params, CrossAttnDownBlock2D_params, CrossAttnUpBlock2D_params, Resnet2d_params, Transformer2D_params
        }
    }
};

// use core::time;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Unet2dConditionModel {
    pub operations: Vec<Box<dyn Layer>>,
    pub time_emb: Rc<RefCell<ndarray::Array4<f32>>>,
    pub timestep: f32,
    pub hidden_states: Rc<RefCell<Vec<ndarray::Array4<f32>>>>
}
impl Unet2dConditionModel {
    pub fn new_weights_bias(
        encoder_cross: Rc<RefCell<ndarray::Array3<f32>>>,
        time_emb: Rc<RefCell<ndarray::Array4<f32>>>,
        res_hidden: Rc<RefCell<Vec<ndarray::Array4<f32>>>>
    ) -> Self {
    let kernel1 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight= input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_linear_bias.safetensors".to_string()).unwrap();
    let kernels = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv_short_weight_b.safetensors".to_string()).unwrap();

    let res1_params_up= Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: 960, 
            out_channels_1: 320, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 320, 
            out_channels_2: 320, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
    let kernel1 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_linear_bias.safetensors".to_string()).unwrap();
    let kernels = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv_short_weight_b.safetensors".to_string()).unwrap();
    let res2_params_up = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: 640, 
            out_channels_1: 320, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 320, 
            out_channels_2: 320, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };

    let kernel1 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv2_weight_b.safetensors".to_string()).unwrap();
    let norm1_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_norm2_weight_b.safetensors".to_string()).unwrap();
    let linear_weight = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_linear_bias.safetensors".to_string()).unwrap();
    let kernels = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv_short_weight_b.safetensors".to_string()).unwrap();
    let res3_params_up = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: 640, 
            out_channels_1: 320, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 320, 
            out_channels_2: 320, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };

    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut trans2_params : Option<Transformer2D_params> = None;
    let mut trans3_params : Option<Transformer2D_params> = None;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2_params : Option<Resnet2d_params> = None;
    let mut resnet3_params : Option<Resnet2d_params> = None;

    for i in 0..3 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv1.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv2.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv1_b.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv2_b.safetensors", i)).unwrap();
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_norm1.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_norm2.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_norm2_b.safetensors", i)).unwrap();

        let kernels = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv_short.safetensors", i)).unwrap();
        let cs_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv_short_b.safetensors", i)).unwrap();
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 || i == 1 {2560} else {1920};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 1280, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 1280, 
            out_channels_2: 1280, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet1_params = Some(resnet_par);
        } else if i == 1 {
            resnet2_params = Some(resnet_par);
        } else {
            resnet3_params = Some(resnet_par);
        }
    }
    for j in 0..3 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..10 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();



            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 20, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 20, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans1_params = Some(params);
        } else if j == 1 {
            trans2_params = Some(params);
        } else {
            trans3_params = Some(params);
        }
    }
    let kernel_upsample1 = input(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_upsample.safetensors".to_string()).unwrap();
    let cupsample1_b = input(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_upsample_b.safetensors".to_string()).unwrap();
    let crossattnupblock_1_params = CrossAttnUpBlock2D_params {
        params_for_transformer1: trans1_params.unwrap(), 
        params_for_transformer2: trans2_params.unwrap(),
        params_for_transformer3: trans3_params.unwrap(),
        params_for_resnet1: resnet1_params.unwrap(),
        params_for_resnet2: resnet2_params.unwrap(),
        params_for_resnet3: resnet3_params.unwrap(),
        in_channels: 1280, out_channels: 1280, padding: 1, stride: 1, kernel_size: 3, 
        kernel_weights: kernel_upsample1.into_raw_vec_and_offset().0, bias: cupsample1_b, is_bias: true,
        hidden_states: Rc::clone(&res_hidden)
    };

    let mut trans12_params : Option<Transformer2D_params> = None;
    let mut trans22_params : Option<Transformer2D_params> = None;
    let mut trans32_params : Option<Transformer2D_params> = None;
    let mut resnet12_params : Option<Resnet2d_params> = None;
    let mut resnet22_params : Option<Resnet2d_params> = None;
    let mut resnet32_params : Option<Resnet2d_params> = None;

    for i in 0..3 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv1.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv2.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv1_b.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv2_b.safetensors", i)).unwrap();
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_norm1.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_norm2.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_norm2_b.safetensors", i)).unwrap();
        let kernels = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv_short.safetensors", i)).unwrap();
        let cs_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv_short_b.safetensors", i)).unwrap();
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {1920} else if i == 1 {1280} else {960};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 640, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 640, 
            out_channels_2: 640, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet12_params = Some(resnet_par);
        } else if i == 1 {
            resnet22_params = Some(resnet_par);
        } else {
            resnet32_params = Some(resnet_par);
        }
    }
    for j in 0..3 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..2 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        

            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();

            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 10, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 10, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in= input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let gamma_in= input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans12_params = Some(params);
        } else if j == 1 {
            trans22_params = Some(params);
        } else {
            trans32_params = Some(params);
        }
    }
    let kernelupsample2 = input(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_upsample.safetensors".to_string()).unwrap();
    let cupsample2_b = input(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_upsample_b.safetensors".to_string()).unwrap();
    let crossattnupblock_2_params = CrossAttnUpBlock2D_params {
        params_for_transformer1: trans12_params.unwrap(), 
        params_for_transformer2: trans22_params.unwrap(),
        params_for_transformer3: trans32_params.unwrap(),
        params_for_resnet1: resnet12_params.unwrap(),
        params_for_resnet2: resnet22_params.unwrap(),
        params_for_resnet3: resnet32_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 1, kernel_size: 3, kernel_weights: kernelupsample2.into_raw_vec_and_offset().0,
        bias: cupsample2_b, is_bias: true,
        hidden_states: Rc::clone(&res_hidden)
    };

   let mut trans1:Transformer2D;
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut resnet1:Resnet2d;
    let mut resnet1_params_mid : Option<Resnet2d_params> = None;
    let mut resnet2:Resnet2d;
    let mut resnet2_params_mid : Option<Resnet2d_params> = None;
    for i in 0..2 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv1.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv1_b.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv2.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv2_b.safetensors", i)).unwrap();
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_norm1.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_norm2.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_norm2_b.safetensors", i)).unwrap();
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = 1280;
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 1280, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 1280, 
            out_channels_2: 1280, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: false,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: vec![1.],
            bias_s: ndarray::Array4::from_shape_vec([1, 1, 1, 1], vec![1.]).unwrap(), is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet1_params_mid = Some(resnet_par);
        } else {
            resnet2_params_mid = Some(resnet_par);
        }
    }
    let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
    for i in 0..10 {
        let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_q_test.safetensors", i)).unwrap();
        let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_k_test.safetensors", i)).unwrap();
        let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_v_test.safetensors", i)).unwrap();
        let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_out_w_test.safetensors", i)).unwrap(); 
        let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_out_b_test.safetensors", i)).unwrap(); 
    
        let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_q_test.safetensors", i)).unwrap();
        let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_k_test.safetensors", i)).unwrap();
        let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_v_test.safetensors", i)).unwrap();
        let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_out_w_test.safetensors", i)).unwrap(); 
        let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_out_b_test.safetensors", i)).unwrap(); 
    
        let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_geglu_w_test.safetensors", i)).unwrap();
        let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_geglu_b_test.safetensors", i)).unwrap();
    
        let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_ff_w_test.safetensors", i)).unwrap();
        let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_ff_b_test.safetensors", i)).unwrap();
    
        let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm1_w_test.safetensors", i)).unwrap();
        let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm2_w_test.safetensors", i)).unwrap();
        let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm3_w_test.safetensors", i)).unwrap();

        let beta1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm1_b_test.safetensors", i)).unwrap();
        let beta2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm2_b_test.safetensors", i)).unwrap();
        let beta3 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm3_b_test.safetensors", i)).unwrap();



        let btb1_params = BasicTransofmerBlock_params {
        eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
        eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
        eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
        weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
        weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
        weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
        weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
        encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 20, 

        weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
        weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
        weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
        weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
        encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 20, 

        weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
        weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
        };
        param_vec.push(btb1_params);
    }
    let weights_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_w_test.safetensors".to_string()).unwrap(); 
    let bias_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_b_test.safetensors".to_string()).unwrap(); 

    let weights_out = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_w_test.safetensors".to_string()).unwrap(); 
    let bias_out = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_b_test.safetensors".to_string()).unwrap(); 

    let gamma_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_norm_w_test.safetensors".to_string()).unwrap(); 
    let beta_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_norm_b_test.safetensors".to_string()).unwrap(); 


    let params_trans_mid = Transformer2D_params{
        number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
    weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
    weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
    params_for_basics_vec : param_vec
    };
    // let mid = mid_block::new(
    //     params, 
    //     resnet1_params.unwrap(), 
    //     resnet2_params.unwrap());



    let kernel1 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_linear_bias.safetensors".to_string()).unwrap();
    let kernel_down = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample.safetensors".to_string()).unwrap();
    let cdown_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample_b.safetensors".to_string()).unwrap();
    let res1_params_down = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: 320, 
        out_channels_1: 320, 
        padding_1: 1, 
        stride_1 : 1, 
        kernel_size_1 : 3, 
        kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
        bias_c1: c1_b, is_bias_c1: true,
        weights: linear_weight, bias : linear_bias, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
        in_channels_2: 320, 
        out_channels_2: 320, 
        padding_2: 1, stride_2 : 1, 
        kernel_size_2 : 3, 
        kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
        bias_c2: c2_b, is_bias_c2: true,
        is_shortcut: false,
        in_channels_short: 320, out_channels_short: 320, padding_short: 1, stride_short : 1, kernel_size_short : 3, kernel_weights_short: vec![1.],
        bias_s: ndarray::Array4::from_elem([1, 1, 1, 1], 1.), is_bias_s: true,
        time_emb: Rc::clone(&time_emb)
    };

    let kernel1 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight= input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_linear_bias.safetensors".to_string()).unwrap();

    let res2_params_down = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: 320, 
        out_channels_1: 320, 
        padding_1: 1, 
        stride_1 : 1, 
        kernel_size_1 : 3, 
        kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
        bias_c1: c1_b, is_bias_c1: true,
        weights: linear_weight, bias : linear_bias, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
        in_channels_2: 320, 
        out_channels_2: 320, 
        padding_2: 1, stride_2 : 1, 
        kernel_size_2 : 3, 
        kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
        bias_c2: c2_b, is_bias_c2: true,
        is_shortcut: false,
        in_channels_short: 320, out_channels_short: 320, padding_short: 1, stride_short : 1, kernel_size_short : 3, kernel_weights_short: vec![1.],
        bias_s: ndarray::Array4::from_elem([1, 1, 1, 1], 1.), is_bias_s: true,
        time_emb: Rc::clone(&time_emb)
    };

    let kernel_down = input(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_downsample.safetensors".to_string()).unwrap();
    let cdown_b = input(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_downsample_b.safetensors".to_string()).unwrap();
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut trans2_params : Option<Transformer2D_params> = None;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2_params : Option<Resnet2d_params> = None;

    for i in 0..2 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv1.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv2.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv1_b.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv2_b.safetensors", i)).unwrap();
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm1.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm2.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm2_b.safetensors", i)).unwrap();
        let kernels = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv_short.safetensors", i)).unwrap()}
        else
        {kernel1.clone()};
        let cs_b = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv_short_b.safetensors", i)).unwrap()}
        else
        {c1_b.clone()};
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {320} else {640};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 640, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 640, 
            out_channels_2: 640, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: shortcut_flag,
            in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet1_params = Some(resnet_par);
        } else {
            resnet2_params = Some(resnet_par);
        } 
    }
    for j in 0..2 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..2 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();



            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 640, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 640, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 640, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 10, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 10, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out= input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans1_params = Some(params);
        } else{
            trans2_params = Some(params);
        }
    }
    let crossattndownblock2d_1_params = CrossAttnDownBlock2D_params {
        is_downsample2d: true,
        params_for_transformer1: trans1_params.unwrap(),
        params_for_transformer2: trans2_params.unwrap(),
        params_for_resnet1: resnet1_params.unwrap(),
        params_for_resnet2: resnet2_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 3, kernel_weights: kernel_down.into_raw_vec_and_offset().0,
        is_bias: true, bias: cdown_b,
        hidden_states: Rc::clone(&res_hidden)
    };
    let mut trans12_params : Option<Transformer2D_params> = None;
    let mut trans22_params : Option<Transformer2D_params> = None;
    let mut resnet12_params : Option<Resnet2d_params> = None;
    let mut resnet22_params : Option<Resnet2d_params> = None;

    for i in 0..2 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv1.safetensors", i)).unwrap();

        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv2.safetensors", i)).unwrap();

        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv1_b.safetensors", i)).unwrap();

        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv2_b.safetensors", i)).unwrap();

        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm1.safetensors", i)).unwrap();

        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm2.safetensors", i)).unwrap();

        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm1_b.safetensors", i)).unwrap();

        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm2_b.safetensors", i)).unwrap();

        let kernels = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv_short.safetensors", i)).unwrap()}
        else
        {kernel1.clone()};
        let cs_b = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv_short_b.safetensors", i)).unwrap()}
        else
        {c1_b.clone()};

        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {640} else {1280};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 1280, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 1280, 
            out_channels_2: 1280, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: shortcut_flag,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet12_params = Some(resnet_par);
        } else {
            resnet22_params = Some(resnet_par);
        } 
    }
    for j in 0..2 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..10 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();


            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 20, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 20, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projout_b_test.safetensors", j)).unwrap(); 

        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans12_params = Some(params);
        } else{
            trans22_params = Some(params);
        }
    }
    let crossattndownblock2d_2_params = CrossAttnDownBlock2D_params {
        is_downsample2d: false,
        params_for_transformer1: trans12_params.unwrap(),
        params_for_transformer2: trans22_params.unwrap(),
        params_for_resnet1: resnet12_params.unwrap(),
        params_for_resnet2: resnet22_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 13123124, kernel_weights: vec![1.],
        bias: ndarray::Array4::from_shape_vec([1, 1, 1, 1], vec![1.]).unwrap(), is_bias: false,
        hidden_states: Rc::clone(&res_hidden)
    };
    let kernel_downblock2d_downsample = input(format!(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample.safetensors")).unwrap();
    let c_downblock2d_downsample_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample_b.safetensors")).unwrap();

    let weights_temb1 = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_w.safetensors")).unwrap();
    let bias_temb1 = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_b.safetensors")).unwrap();
    let weights_temb2 = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_w.safetensors")).unwrap();
    let bias_temb2 = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_b.safetensors")).unwrap();

    let weights_aug1 = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l1.safetensors")).unwrap();
    let weights_aug2 = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l2.safetensors")).unwrap();
    let bias_aug1 = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l1_b.safetensors")).unwrap();
    let bias_aug2 = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l2_b.safetensors")).unwrap();

    let kernels_in = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_in.safetensors")).unwrap();
    let cin_b = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_in_b.safetensors")).unwrap();

    let kernel_out = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_out.safetensors")).unwrap();
    let cout_b = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_out_b.safetensors")).unwrap();

    let gamma_out = input(format!(r"C:\study\coursework\src\trash\test_unet_norm_out.safetensors")).unwrap();
    let beta_out = input(format!(r"C:\study\coursework\src\trash\test_unet_norm_out_b.safetensors")).unwrap();
    

    let mut unet = Unet2dConditionModel::new(
        time_emb,
        weights_temb1,
        bias_temb1,
        true,
        weights_temb2,
        bias_temb2,
        true,
        weights_aug1, 
        bias_aug1,
        true,
        weights_aug2,
        bias_aug2,
        true,
        4,
        320,
        1, 1, 3,
        kernels_in.into_raw_vec_and_offset().0,
        cin_b,
        true,
        res1_params_down,
        res2_params_down,
        320, 320, 1, 2, 3,
        kernel_downblock2d_downsample.into_raw_vec_and_offset().0,
        c_downblock2d_downsample_b,
        true,
        crossattndownblock2d_1_params,
        crossattndownblock2d_2_params,
        crossattnupblock_1_params,
        crossattnupblock_2_params,
        res1_params_up,
        res2_params_up,
        res3_params_up,
        res_hidden,
        params_trans_mid,
        resnet1_params_mid.unwrap(),
        resnet2_params_mid.unwrap(),
        32, 1e-05, 
        gamma_out, 
        beta_out,
        320, 4, 1, 1, 3,
        kernel_out.into_raw_vec_and_offset().0,
        cout_b, true
    );
        return unet
    }

    pub fn new(
        // timestep : f32, // time
        time_emb : Rc<RefCell<ndarray::Array4<f32>>>,

        weights_temb1: ndarray::Array4<f32>, bias_temb1: ndarray::Array4<f32>, is_bias_temb1: bool, // temb
        weights_temb2: ndarray::Array4<f32>, bias_temb2: ndarray::Array4<f32>, is_bias_temb2: bool,

        weights_aug1: ndarray::Array4<f32>, bias_aug1: ndarray::Array4<f32>, is_bias_aug1: bool, // temb
        weights_aug2: ndarray::Array4<f32>, bias_aug2: ndarray::Array4<f32>, is_bias_aug2: bool,

        in_channels_in : usize, 
        out_channels_in: usize, 
        padding_in : i32, 
        stride_in : i32, 
        kernel_size_in : usize, 
        kernel_weights_in : Vec<f32>, // conv_in
        bias_in: ndarray::Array4<f32>,
        is_bias_in: bool,

        params_for_resnet1_down : Resnet2d_params, // down
        params_for_resnet2_down : Resnet2d_params,
        in_channels_down : usize, 
        out_channels_down : usize,
        padding_down : i32, 
        stride_down : i32, 
        kernel_size_down : usize, 
        kernel_weights_down : Vec<f32>,
        bias_down: ndarray::Array4<f32>,
        is_bias_down: bool,
        params_for_crattbl1 : CrossAttnDownBlock2D_params,
        params_for_crattbl2 : CrossAttnDownBlock2D_params,

        params_for_crossblock1 : CrossAttnUpBlock2D_params, // up
        params_for_crossblock2 : CrossAttnUpBlock2D_params,
        params_for_resnet1_up : Resnet2d_params,
        params_for_resnet2_up : Resnet2d_params,
        params_for_resnet3_up : Resnet2d_params,
        hidden_states : Rc<RefCell<Vec<ndarray::Array4<f32>>>>,

        params_for_transformer2d :Transformer2D_params, // mid
        params_for_resnet_1_mid: Resnet2d_params,
        params_for_resnet_2_mid: Resnet2d_params,

        number_of_groups_out: usize, eps_out: f32, gamma_out: ndarray::Array4<f32>, beta_out: ndarray::Array4<f32>, //out
        in_channels_out : usize, 
        out_channels_out: usize, 
        padding_out : i32, 
        stride_out : i32, 
        kernel_size_out : usize, 
        kernel_weights_out : Vec<f32>,
        bias_out: ndarray::Array4<f32>,
        is_bias_out: bool
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(Linear { // time_emb
            weights : weights_temb1, 
            bias : bias_temb1, 
            is_bias : is_bias_temb1
        })); 
        vec.push(Box::new(SiLU));
        vec.push(Box::new(Linear {
            weights : weights_temb2, 
            bias : bias_temb2, 
            is_bias : is_bias_temb2
        }));



        vec.push(Box::new(Linear { // add_emb
            weights : weights_aug1, 
            bias : bias_aug1, 
            is_bias : is_bias_aug1
        })); 
        vec.push(Box::new(SiLU));
        vec.push(Box::new(Linear {
            weights : weights_aug2, 
            bias : bias_aug2, 
            is_bias : is_bias_aug2
        }));

        vec.push(Box::new(Conv2d{
            in_channels : in_channels_in, 
            out_channels : out_channels_in, 
            padding : padding_in, 
            stride : stride_in, 
            kernel_size : kernel_size_in, 
            kernel_weights : kernel_weights_in,
            bias: bias_in,
            is_bias: is_bias_in
        }));
        let down = Down_blocks::new(
            params_for_resnet1_down,
            params_for_resnet2_down,
            in_channels_down,
            out_channels_down,
            padding_down,
            stride_down,
            kernel_size_down,
            kernel_weights_down, 
            bias_down,
            is_bias_down,
            params_for_crattbl1, params_for_crattbl2, Rc::clone(&hidden_states));
        vec.push(Box::new(down));

        let mid = mid_block::new(
            params_for_transformer2d, 
            params_for_resnet_1_mid, 
            params_for_resnet_2_mid
        );
        vec.push(Box::new(mid));
        let up = Up_blocks::new(
        params_for_crossblock1, 
        params_for_crossblock2, 
        params_for_resnet1_up, 
        params_for_resnet2_up, 
        params_for_resnet3_up, 
        Rc::clone(&hidden_states));
        vec.push(Box::new(up));

        vec.push(Box::new(GroupNorm{
        number_of_groups : number_of_groups_out, 
        eps : eps_out, 
        gamma : gamma_out, 
        beta: beta_out
        }));
        vec.push(Box::new(SiLU));
        vec.push(Box::new(Conv2d{
        in_channels: in_channels_out, 
        out_channels : out_channels_out, 
        padding : padding_out, 
        stride: stride_out, 
        kernel_size : kernel_size_out, 
        kernel_weights : kernel_weights_out,
        bias: bias_out,
        is_bias: is_bias_out
    }));
        Self { operations: vec, time_emb: Rc::clone(&time_emb), timestep: -1., hidden_states: hidden_states}
    }
}

impl Unet2dConditionModel { 
    pub fn operation(&mut self, 
        args: &mut ndarray::Array4<f32>, 
        timestep:f32, 
        kwargs : Rc<RefCell<(ndarray::Array4<f32>, ndarray::Array4<f32>)>>
    ) -> Result<(), Box<dyn std::error::Error>> {
        // let operations = &self.operations;
        // let mut temb = self.time_emb.borrow_mut();
        // let mut in_temb = temb.0.clone();
        // let mut in_temb_shape = temb.1.clone();
        self.timestep = timestep;
        let timestep = ndarray::Array1::from_elem([1], timestep);
        let mut in_temb = scalar_timestep_embedding(timestep, args.shape()[0], 320).unwrap();
        let _ = &self.operations[0].operation(&mut in_temb)?;
        let _ = &self.operations[1].operation(&mut in_temb)?;
        let _ = &self.operations[2].operation(&mut in_temb)?;
        let mut add_emb = aug_emb(kwargs).unwrap();
        let _ = &self.operations[3].operation(&mut add_emb);
        let _ = &self.operations[4].operation(&mut add_emb);
        let _ = &self.operations[5].operation(&mut add_emb);
        in_temb += &add_emb;

        {*self.time_emb.borrow_mut() = in_temb;}

        for i in 6..*&self.operations.len(){
            let _ = &self.operations[i].operation(args)?;
        } 
        Ok(())
    }
}

// #[test]
// fn test_timeemb() {
//     let (temb, temb_shape) = scalar_timestep_embedding(0., 2, 320).unwrap();
//     let (l1w, l1ws) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_w.safetensors")).unwrap();
//     let (l1b, l1bs) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_b.safetensors")).unwrap();
//     let linear1 = Linear{weigths:l1w.to_vec(), weights_shape: l1ws.to_vec(), bias: l1b.to_vec(), bias_shape: l1bs.to_vec(), is_bias: true};
//     let (l2w, l2ws) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_w.safetensors")).unwrap();
//     let (l2b, l2bs) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_b.safetensors")).unwrap();
//     let linear2 = Linear{weigths:l2w.to_vec(), weights_shape: l2ws.to_vec(), bias: l2b.to_vec(), bias_shape: l2bs.to_vec(), is_bias: true};
//     let silu = SiLU;
//     let (temb, temb_shape) = linear1.operation((temb.clone(), temb_shape.clone())).unwrap();
//     let (temb, temb_shape) = silu.operation((temb.clone(), temb_shape.clone())).unwrap();
//     let (temb, temb_shape) = linear2.operation((temb.clone(), temb_shape.clone())).unwrap();
//     let (py_temb, py_temb_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_output.safetensors")).unwrap();
//     assert!(py_temb_shape.to_vec() == temb_shape);
//     for i in 0..py_temb.len() {
//         let d = (temb[i] - py_temb[i]).abs();
//         assert!(!d.is_nan());
//         assert!(d <= 1e-05);
//     }
// }

#[test]
fn test_unet_simple() {
    let encoder_cross = input(format!(r"C:\study\coursework\src\trash\test_unet_encoder.safetensors")).unwrap();
    let encoder_cross = Rc::new(RefCell::new(encoder_cross.remove_axis(ndarray::Axis(0))));
    let time_emb = Rc::new(RefCell::new(ndarray::Array4::from_elem([1, 1, 1, 1], 1.)));

    let res_hidden = Rc::new(RefCell::new(Vec::<ndarray::Array4<f32>>::new()));




    let kernel1 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight= input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_linear_bias.safetensors".to_string()).unwrap();
    let kernels = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res0_conv_short_weight_b.safetensors".to_string()).unwrap();

    let res1_params_up= Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: 960, 
            out_channels_1: 320, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 320, 
            out_channels_2: 320, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
    let kernel1 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_linear_bias.safetensors".to_string()).unwrap();
    let kernels = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res1_conv_short_weight_b.safetensors".to_string()).unwrap();
    let res2_params_up = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: 640, 
            out_channels_1: 320, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 320, 
            out_channels_2: 320, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };

    let kernel1 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv2_weight_b.safetensors".to_string()).unwrap();
    let norm1_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_norm2_weight_b.safetensors".to_string()).unwrap();
    let linear_weight = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_linear_bias.safetensors".to_string()).unwrap();
    let kernels = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_upblocks_upblock2d_res2_conv_short_weight_b.safetensors".to_string()).unwrap();
    let res3_params_up = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: 640, 
            out_channels_1: 320, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 320, 
            out_channels_2: 320, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };

    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut trans2_params : Option<Transformer2D_params> = None;
    let mut trans3_params : Option<Transformer2D_params> = None;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2_params : Option<Resnet2d_params> = None;
    let mut resnet3_params : Option<Resnet2d_params> = None;

    for i in 0..3 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv1.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv2.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv1_b.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv2_b.safetensors", i)).unwrap();
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_norm1.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_norm2.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_norm2_b.safetensors", i)).unwrap();

        let kernels = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv_short.safetensors", i)).unwrap();
        let cs_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_conv_short_b.safetensors", i)).unwrap();
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 || i == 1 {2560} else {1920};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 1280, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 1280, 
            out_channels_2: 1280, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet1_params = Some(resnet_par);
        } else if i == 1 {
            resnet2_params = Some(resnet_par);
        } else {
            resnet3_params = Some(resnet_par);
        }
    }
    for j in 0..3 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..10 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();



            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 20, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 20, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans1_params = Some(params);
        } else if j == 1 {
            trans2_params = Some(params);
        } else {
            trans3_params = Some(params);
        }
    }
    let kernel_upsample1 = input(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_upsample.safetensors".to_string()).unwrap();
    let cupsample1_b = input(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock0_upsample_b.safetensors".to_string()).unwrap();
    let crossattnupblock_1_params = CrossAttnUpBlock2D_params {
        params_for_transformer1: trans1_params.unwrap(), 
        params_for_transformer2: trans2_params.unwrap(),
        params_for_transformer3: trans3_params.unwrap(),
        params_for_resnet1: resnet1_params.unwrap(),
        params_for_resnet2: resnet2_params.unwrap(),
        params_for_resnet3: resnet3_params.unwrap(),
        in_channels: 1280, out_channels: 1280, padding: 1, stride: 1, kernel_size: 3, 
        kernel_weights: kernel_upsample1.into_raw_vec_and_offset().0, bias: cupsample1_b, is_bias: true,
        hidden_states: Rc::clone(&res_hidden)
    };

    let mut trans12_params : Option<Transformer2D_params> = None;
    let mut trans22_params : Option<Transformer2D_params> = None;
    let mut trans32_params : Option<Transformer2D_params> = None;
    let mut resnet12_params : Option<Resnet2d_params> = None;
    let mut resnet22_params : Option<Resnet2d_params> = None;
    let mut resnet32_params : Option<Resnet2d_params> = None;

    for i in 0..3 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv1.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv2.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv1_b.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv2_b.safetensors", i)).unwrap();
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_norm1.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_norm2.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_norm2_b.safetensors", i)).unwrap();
        let kernels = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv_short.safetensors", i)).unwrap();
        let cs_b = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_conv_short_b.safetensors", i)).unwrap();
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {1920} else if i == 1 {1280} else {960};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 640, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 640, 
            out_channels_2: 640, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: true,
            in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet12_params = Some(resnet_par);
        } else if i == 1 {
            resnet22_params = Some(resnet_par);
        } else {
            resnet32_params = Some(resnet_par);
        }
    }
    for j in 0..3 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..2 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        

            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();

            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 10, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 10, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in= input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let gamma_in= input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out = input(format!(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans12_params = Some(params);
        } else if j == 1 {
            trans22_params = Some(params);
        } else {
            trans32_params = Some(params);
        }
    }
    let kernelupsample2 = input(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_upsample.safetensors".to_string()).unwrap();
    let cupsample2_b = input(r"C:\study\coursework\src\trash\test_upblocks_crossattnupblock1_upsample_b.safetensors".to_string()).unwrap();
    let crossattnupblock_2_params = CrossAttnUpBlock2D_params {
        params_for_transformer1: trans12_params.unwrap(), 
        params_for_transformer2: trans22_params.unwrap(),
        params_for_transformer3: trans32_params.unwrap(),
        params_for_resnet1: resnet12_params.unwrap(),
        params_for_resnet2: resnet22_params.unwrap(),
        params_for_resnet3: resnet32_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 1, kernel_size: 3, kernel_weights: kernelupsample2.into_raw_vec_and_offset().0,
        bias: cupsample2_b, is_bias: true,
        hidden_states: Rc::clone(&res_hidden)
    };


    // let upblocks = Up_blocks::new(crossattnupblock_1_params, 
    //     crossattnupblock_2_params, 
    //     res1_params, 
    //     res2_params, 
    //     res3_params, 
    //     Rc::clone(&res_hidden_states));



   let mut trans1:Transformer2D;
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut resnet1:Resnet2d;
    let mut resnet1_params_mid : Option<Resnet2d_params> = None;
    let mut resnet2:Resnet2d;
    let mut resnet2_params_mid : Option<Resnet2d_params> = None;
    for i in 0..2 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv1.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv1_b.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv2.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv2_b.safetensors", i)).unwrap();
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_norm1.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_norm2.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_norm2_b.safetensors", i)).unwrap();
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = 1280;
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 1280, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 1280, 
            out_channels_2: 1280, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: false,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: vec![1.],
            bias_s: ndarray::Array4::from_shape_vec([1, 1, 1, 1], vec![1.]).unwrap(), is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet1_params_mid = Some(resnet_par);
        } else {
            resnet2_params_mid = Some(resnet_par);
        }
    }
    let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
    for i in 0..10 {
        let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_q_test.safetensors", i)).unwrap();
        let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_k_test.safetensors", i)).unwrap();
        let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_v_test.safetensors", i)).unwrap();
        let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_out_w_test.safetensors", i)).unwrap(); 
        let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_out_b_test.safetensors", i)).unwrap(); 
    
        let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_q_test.safetensors", i)).unwrap();
        let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_k_test.safetensors", i)).unwrap();
        let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_v_test.safetensors", i)).unwrap();
        let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_out_w_test.safetensors", i)).unwrap(); 
        let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_out_b_test.safetensors", i)).unwrap(); 
    
        let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_geglu_w_test.safetensors", i)).unwrap();
        let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_geglu_b_test.safetensors", i)).unwrap();
    
        let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_ff_w_test.safetensors", i)).unwrap();
        let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_ff_b_test.safetensors", i)).unwrap();
    
        let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm1_w_test.safetensors", i)).unwrap();
        let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm2_w_test.safetensors", i)).unwrap();
        let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm3_w_test.safetensors", i)).unwrap();

        let beta1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm1_b_test.safetensors", i)).unwrap();
        let beta2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm2_b_test.safetensors", i)).unwrap();
        let beta3 = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_norm3_b_test.safetensors", i)).unwrap();



        let btb1_params = BasicTransofmerBlock_params {
        eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
        eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
        eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
        weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
        weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
        weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
        weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
        encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 20, 

        weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
        weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
        weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
        weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
        encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 20, 

        weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
        weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
        };
        param_vec.push(btb1_params);
    }
    let weights_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_w_test.safetensors".to_string()).unwrap(); 
    let bias_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_b_test.safetensors".to_string()).unwrap(); 

    let weights_out = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_w_test.safetensors".to_string()).unwrap(); 
    let bias_out = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_b_test.safetensors".to_string()).unwrap(); 

    let gamma_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_norm_w_test.safetensors".to_string()).unwrap(); 
    let beta_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_norm_b_test.safetensors".to_string()).unwrap(); 


    let params_trans_mid = Transformer2D_params{
        number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
    weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
    weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
    params_for_basics_vec : param_vec
    };
    // let mid = mid_block::new(
    //     params, 
    //     resnet1_params.unwrap(), 
    //     resnet2_params.unwrap());



    let kernel1 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_linear_bias.safetensors".to_string()).unwrap();
    let kernel_down = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample.safetensors".to_string()).unwrap();
    let cdown_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample_b.safetensors".to_string()).unwrap();
    let res1_params_down = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: 320, 
        out_channels_1: 320, 
        padding_1: 1, 
        stride_1 : 1, 
        kernel_size_1 : 3, 
        kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
        bias_c1: c1_b, is_bias_c1: true,
        weights: linear_weight, bias : linear_bias, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
        in_channels_2: 320, 
        out_channels_2: 320, 
        padding_2: 1, stride_2 : 1, 
        kernel_size_2 : 3, 
        kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
        bias_c2: c2_b, is_bias_c2: true,
        is_shortcut: false,
        in_channels_short: 320, out_channels_short: 320, padding_short: 1, stride_short : 1, kernel_size_short : 3, kernel_weights_short: vec![1.],
        bias_s: ndarray::Array4::from_elem([1, 1, 1, 1], 1.), is_bias_s: true,
        time_emb: Rc::clone(&time_emb)
    };

    let kernel1 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight= input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_linear_bias.safetensors".to_string()).unwrap();

    let res2_params_down = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: 320, 
        out_channels_1: 320, 
        padding_1: 1, 
        stride_1 : 1, 
        kernel_size_1 : 3, 
        kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
        bias_c1: c1_b, is_bias_c1: true,
        weights: linear_weight, bias : linear_bias, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
        in_channels_2: 320, 
        out_channels_2: 320, 
        padding_2: 1, stride_2 : 1, 
        kernel_size_2 : 3, 
        kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
        bias_c2: c2_b, is_bias_c2: true,
        is_shortcut: false,
        in_channels_short: 320, out_channels_short: 320, padding_short: 1, stride_short : 1, kernel_size_short : 3, kernel_weights_short: vec![1.],
        bias_s: ndarray::Array4::from_elem([1, 1, 1, 1], 1.), is_bias_s: true,
        time_emb: Rc::clone(&time_emb)
    };

    let kernel_down = input(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_downsample.safetensors".to_string()).unwrap();
    let cdown_b = input(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_downsample_b.safetensors".to_string()).unwrap();
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut trans2_params : Option<Transformer2D_params> = None;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2_params : Option<Resnet2d_params> = None;

    for i in 0..2 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv1.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv2.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv1_b.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv2_b.safetensors", i)).unwrap();
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm1.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm2.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm2_b.safetensors", i)).unwrap();
        let kernels = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv_short.safetensors", i)).unwrap()}
        else
        {kernel1.clone()};
        let cs_b = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv_short_b.safetensors", i)).unwrap()}
        else
        {c1_b.clone()};
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {320} else {640};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 640, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 640, 
            out_channels_2: 640, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: shortcut_flag,
            in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet1_params = Some(resnet_par);
        } else {
            resnet2_params = Some(resnet_par);
        } 
    }
    for j in 0..2 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..2 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();



            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 640, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 640, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 640, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 10, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 10, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out= input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans1_params = Some(params);
        } else{
            trans2_params = Some(params);
        }
    }
    let crossattndownblock2d_1_params = CrossAttnDownBlock2D_params {
        is_downsample2d: true,
        params_for_transformer1: trans1_params.unwrap(),
        params_for_transformer2: trans2_params.unwrap(),
        params_for_resnet1: resnet1_params.unwrap(),
        params_for_resnet2: resnet2_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 3, kernel_weights: kernel_down.into_raw_vec_and_offset().0,
        is_bias: true, bias: cdown_b,
        hidden_states: Rc::clone(&res_hidden)
    };
    let mut trans12_params : Option<Transformer2D_params> = None;
    let mut trans22_params : Option<Transformer2D_params> = None;
    let mut resnet12_params : Option<Resnet2d_params> = None;
    let mut resnet22_params : Option<Resnet2d_params> = None;

    for i in 0..2 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv1.safetensors", i)).unwrap();

        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv2.safetensors", i)).unwrap();

        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv1_b.safetensors", i)).unwrap();

        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv2_b.safetensors", i)).unwrap();

        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm1.safetensors", i)).unwrap();

        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm2.safetensors", i)).unwrap();

        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm1_b.safetensors", i)).unwrap();

        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm2_b.safetensors", i)).unwrap();

        let kernels = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv_short.safetensors", i)).unwrap()}
        else
        {kernel1.clone()};
        let cs_b = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv_short_b.safetensors", i)).unwrap()}
        else
        {c1_b.clone()};

        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {640} else {1280};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 1280, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 1280, 
            out_channels_2: 1280, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: shortcut_flag,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet12_params = Some(resnet_par);
        } else {
            resnet22_params = Some(resnet_par);
        } 
    }
    for j in 0..2 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..10 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();


            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder_cross), if_encoder_tensor_1 : false, number_of_heads_1: 20, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder_cross), if_encoder_tensor_2 : true, number_of_heads_2: 20, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projout_b_test.safetensors", j)).unwrap(); 

        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans12_params = Some(params);
        } else{
            trans22_params = Some(params);
        }
    }
    let crossattndownblock2d_2_params = CrossAttnDownBlock2D_params {
        is_downsample2d: false,
        params_for_transformer1: trans12_params.unwrap(),
        params_for_transformer2: trans22_params.unwrap(),
        params_for_resnet1: resnet12_params.unwrap(),
        params_for_resnet2: resnet22_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 13123124, kernel_weights: vec![1.],
        bias: ndarray::Array4::from_shape_vec([1, 1, 1, 1], vec![1.]).unwrap(), is_bias: false,
        hidden_states: Rc::clone(&res_hidden)
    };
    let kernel_downblock2d_downsample = input(format!(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample.safetensors")).unwrap();
    let c_downblock2d_downsample_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample_b.safetensors")).unwrap();

    let weights_temb1 = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_w.safetensors")).unwrap();
    let bias_temb1 = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_b.safetensors")).unwrap();
    let weights_temb2 = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_w.safetensors")).unwrap();
    let bias_temb2 = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_b.safetensors")).unwrap();

    let weights_aug1 = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l1.safetensors")).unwrap();
    let weights_aug2 = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l2.safetensors")).unwrap();
    let bias_aug1 = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l1_b.safetensors")).unwrap();
    let bias_aug2 = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l2_b.safetensors")).unwrap();

    let kernels_in = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_in.safetensors")).unwrap();
    let cin_b = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_in_b.safetensors")).unwrap();

    let kernel_out = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_out.safetensors")).unwrap();
    let cout_b = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_out_b.safetensors")).unwrap();

    let gamma_out = input(format!(r"C:\study\coursework\src\trash\test_unet_norm_out.safetensors")).unwrap();
    let beta_out = input(format!(r"C:\study\coursework\src\trash\test_unet_norm_out_b.safetensors")).unwrap();
    

    let mut unet = Unet2dConditionModel::new(
        time_emb,
        weights_temb1,
        bias_temb1,
        true,
        weights_temb2,
        bias_temb2,
        true,
        weights_aug1, 
        bias_aug1,
        true,
        weights_aug2,
        bias_aug2,
        true,
        4,
        320,
        1, 1, 3,
        kernels_in.into_raw_vec_and_offset().0,
        cin_b,
        true,
        res1_params_down,
        res2_params_down,
        320, 320, 1, 2, 3,
        kernel_downblock2d_downsample.into_raw_vec_and_offset().0,
        c_downblock2d_downsample_b,
        true,
        crossattndownblock2d_1_params,
        crossattndownblock2d_2_params,
        crossattnupblock_1_params,
        crossattnupblock_2_params,
        res1_params_up,
        res2_params_up,
        res3_params_up,
        res_hidden,
        params_trans_mid,
        resnet1_params_mid.unwrap(),
        resnet2_params_mid.unwrap(),
        32, 1e-05, 
        gamma_out, 
        beta_out,
        320, 4, 1, 1, 3,
        kernel_out.into_raw_vec_and_offset().0,
        cout_b, true
    );
    let add_time_ids = input(format!(r"C:\study\coursework\src\trash\test_unet_add_time_ids.safetensors")).unwrap();
    let add_text_embs = input(format!(r"C:\study\coursework\src\trash\test_unet_add_text_embeds.safetensors")).unwrap();
    let kwargs = Rc::new(RefCell::new((add_time_ids, add_text_embs)));
    let mut tensor = input(format!(r"C:\study\coursework\src\trash\test_unet_input.safetensors")).unwrap();
    let _ = unet.operation(&mut tensor, 1., kwargs).unwrap();
    print!("{:?}", tensor);
}

#[test]
fn test_time_emb () {
    let timestep: f32 = 10.;
    let timestep = ndarray::Array1::from_elem([1], timestep);
    let mut in_temb = scalar_timestep_embedding(timestep, 2, 320).unwrap();

    let l1w = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_w.safetensors")).unwrap();
    let l1b = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_b.safetensors")).unwrap();
    let l2w = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_w.safetensors")).unwrap();
    let l2b = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_b.safetensors")).unwrap();
    let lin1 = Linear{weights: l1w, bias: l1b, is_bias: true};
    let lin2 = Linear{weights: l2w, bias: l2b, is_bias: true};
    let silu = SiLU;
    let _ = lin1.operation(&mut in_temb).unwrap();
    let _ = silu.operation(&mut in_temb).unwrap();
    let _ = lin2.operation(&mut in_temb).unwrap();
    let shape = in_temb.shape();
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_output.safetensors")).unwrap();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((in_temb[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-6);
                }
            }
        }
    }

}

#[test]
fn test_aug_emb_v1() {
    let add_time_ids = input(format!(r"C:\study\coursework\src\trash\test_unet_add_time_ids.safetensors")).unwrap();
    let add_text_embs = input(format!(r"C:\study\coursework\src\trash\test_unet_add_text_embeds.safetensors")).unwrap();
    let shape = add_time_ids.dim();
    let add_time_ids_fl = add_time_ids.into_shape_with_order([shape.0 * shape.1 * shape.2 * shape.3]).unwrap();
    let test = scalar_timestep_embedding(add_time_ids_fl, shape.0 * shape.1 * shape.2 * shape.3, 256).unwrap();
    let shape = test.dim();
    let test = test.into_shape_with_order([1, 1, 2, shape.0 * shape.1 * shape.2 * shape.3 / 2]).unwrap();
    let mut test = ndarray::concatenate(ndarray::Axis(3), &[add_text_embs.view(), test.view()]).unwrap();
    let l1_w = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l1.safetensors")).unwrap();
    let l2_w = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l2.safetensors")).unwrap();
    let l1_b = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l1_b.safetensors")).unwrap();
    let l2_b = input(format!(r"C:\study\coursework\src\trash\test_unet_add_l2_b.safetensors")).unwrap();
    let lin1 = Linear{weights: l1_w, bias: l1_b, is_bias: true};
    let lin2 = Linear{weights: l2_w, bias: l2_b, is_bias: true};
    let silu = SiLU;
    let _ = lin1.operation(&mut test);
    let _ = silu.operation(&mut test);
    let _ = lin2.operation(&mut test);
    let shape = test.shape();
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_unet_aug_emb.safetensors")).unwrap();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((test[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-6);
                }
            }
        }
    }
}

fn test_unet_timestep_eq_1() {
        let encoder_cross = input(format!(r"C:\study\coursework\src\trash\test_unet_encoder.safetensors")).unwrap();
    let encoder_cross = Rc::new(RefCell::new(encoder_cross.remove_axis(ndarray::Axis(0))));
    let time_emb = Rc::new(RefCell::new(ndarray::Array4::from_elem([1, 1, 1, 1], 1.)));

    let res_hidden = Rc::new(RefCell::new(Vec::<ndarray::Array4<f32>>::new()));

    let mut unet = Unet2dConditionModel::new_weights_bias(encoder_cross, time_emb, res_hidden);
    let add_time_ids = input(format!(r"C:\study\coursework\src\trash\test_unet_add_time_ids.safetensors")).unwrap();
    let add_text_embs = input(format!(r"C:\study\coursework\src\trash\test_unet_add_text_embeds.safetensors")).unwrap();
    let kwargs = Rc::new(RefCell::new((add_time_ids, add_text_embs)));
    let mut tensor = input(format!(r"C:\study\coursework\src\trash\test_unet_input.safetensors")).unwrap();
    let _ = unet.operation(&mut tensor, 1., kwargs).unwrap();
    print!("{:?}", tensor);
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_unet_output.safetensors")).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-4);
                }
            }
        }
    }
}