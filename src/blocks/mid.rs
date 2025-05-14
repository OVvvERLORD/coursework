use std::rc::Rc;
use std::cell::RefCell;

use crate::{
    layers::{
        params::{
            Transformer2D_params,
            Resnet2d_params,
            BasicTransofmerBlock_params,
        },
        layer::Layer,
    },
    blocks::{
        resnet::Resnet2d,
        trans::Transformer2D
    },
    func::functions::input
};

pub struct mid_block {
    pub operations : Vec<Box<dyn Layer>>,
}

impl mid_block {
    pub fn new(
        params_for_transformer2d :Transformer2D_params,
        params_for_resnet_1: Resnet2d_params,
        params_for_resnet_2: Resnet2d_params
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::new(params_for_resnet_1);
        vec.push(Box::new(resnet1));
        let transformer = Transformer2D::new(
            params_for_transformer2d);
        vec.push(Box::new(transformer));
        let resnet2 = Resnet2d::new(params_for_resnet_2);
        vec.push(Box::new(resnet2));
        Self { operations: vec }
    }
}

impl Layer for mid_block {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        for layer in operations {
            let _= layer.operation(args)?;
        } 
        Ok(())
    }
}

#[test]
fn test_crossattnmidblock() {
    let mut trans1:Transformer2D;
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut resnet1:Resnet2d;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2:Resnet2d;
    let mut resnet2_params : Option<Resnet2d_params> = None;
    let mut tensor = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_input.safetensors".to_string()).unwrap();
    let encoder = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_encoder.safetensors".to_string()).unwrap();
    let temb = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_temb.safetensors".to_string()).unwrap();
    let time_emb = Rc::new(RefCell::new(temb));
    let encoder = Rc::new(RefCell::new(encoder.remove_axis(ndarray::Axis(0))));
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
            resnet1_params = Some(resnet_par);
        } else {
            resnet2_params = Some(resnet_par);
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
    let weights_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_w_test.safetensors".to_string()).unwrap(); 
    let bias_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_b_test.safetensors".to_string()).unwrap(); 

    let weights_out = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_w_test.safetensors".to_string()).unwrap(); 
    let bias_out = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_b_test.safetensors".to_string()).unwrap(); 

    let gamma_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_norm_w_test.safetensors".to_string()).unwrap(); 
    let beta_in = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_norm_b_test.safetensors".to_string()).unwrap(); 


    let params = Transformer2D_params{
        number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
    weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
    weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
    params_for_basics_vec : param_vec
    };
    let mid = mid_block::new(
        params, 
        resnet1_params.unwrap(), 
        resnet2_params.unwrap());
    let _ = mid.operation(&mut tensor);

    let shape = tensor.shape();
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_output.safetensors")).unwrap();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-3);
                }
            }
        }
    }
}