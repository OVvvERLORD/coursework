use crate::{
    layers::{
        layer::Layer,
        params::{
            CrossAttnUpBlock2D_params,
            Resnet2d_params,
            BasicTransofmerBlock_params,
            Transformer2D_params
        }
    },
    blocks::{
        attn::CrossAttnUpBlock2D,
        upblock::UpBlock2d
    },
    func::functions::input
};

use std::rc::Rc;
use std::cell::RefCell;

pub struct Up_blocks {
    pub operations : Vec<Box<dyn Layer>>,
}

impl Up_blocks {
    pub fn new(
        params_for_crossblock1 : CrossAttnUpBlock2D_params,
        params_for_crossblock2 : CrossAttnUpBlock2D_params,
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        params_for_resnet3 : Resnet2d_params,
        hidden_states : Rc<RefCell<Vec<ndarray::Array4<f32>>>>
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let crossattnupblock1 = CrossAttnUpBlock2D::new(params_for_crossblock1);
        vec.push(Box::new(crossattnupblock1));
        let crossattnupblock2 = CrossAttnUpBlock2D::new(params_for_crossblock2);
        vec.push(Box::new(crossattnupblock2));
        let upblock2d = UpBlock2d::new(
            params_for_resnet1,
            params_for_resnet2,
            params_for_resnet3,
            hidden_states);
        vec.push(Box::new(upblock2d));
        Self { operations: vec }
    }
}

impl Layer for Up_blocks {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        for layer in operations {
            let _ = layer.operation(args).unwrap();
        } 
        Ok(())
    }
}

#[test]
fn test_up_blocks() {
    let temb = input(format!(r"C:\study\coursework\src\trash\test_upblocks_temb.safetensors")).unwrap();
    let time_emb = Rc::new(RefCell::new(temb));
    let encoder = input(format!(r"C:\study\coursework\src\trash\test_upblocks_encoder.safetensors")).unwrap();
    let encoder = Rc::new(RefCell::new(encoder.remove_axis(ndarray::Axis(0))));
    let res_hidden_states = Rc::new(RefCell::new(Vec::<ndarray::Array4<f32>>::new()));

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

    let res1_params = Resnet2d_params{
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
    let res2_params = Resnet2d_params{
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
    let res3_params = Resnet2d_params{
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
            encoder_hidden_tensor_1 : Rc::clone(&encoder), if_encoder_tensor_1 : false, number_of_heads_1: 20, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder), if_encoder_tensor_2 : true, number_of_heads_2: 20, 

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
        hidden_states: Rc::clone(&res_hidden_states)
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
            encoder_hidden_tensor_1 : Rc::clone(&encoder), if_encoder_tensor_1 : false, number_of_heads_1: 10, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder), if_encoder_tensor_2 : true, number_of_heads_2: 10, 

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
        hidden_states: Rc::clone(&res_hidden_states)
    };

    for i in 0..9 {
        let hidden = input(format!(r"C:\study\coursework\src\trash\test_upblocks_res_hidden{}.safetensors", i)).unwrap();
        res_hidden_states.borrow_mut().push(hidden);
    }

    let mut tensor = input(format!(r"C:\study\coursework\src\trash\test_upblocks_input.safetensors")).unwrap();

    let upblocks = Up_blocks::new(crossattnupblock_1_params, 
        crossattnupblock_2_params, 
        res1_params, 
        res2_params, 
        res3_params, 
        Rc::clone(&res_hidden_states));


    let _ = upblocks.operation(&mut tensor);
    let shape = tensor.shape();
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_upblocks_output.safetensors")).unwrap();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-4);
                    assert!(!(tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).is_nan());
                }
            }
        }
    }
}