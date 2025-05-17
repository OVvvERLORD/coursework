use ndarray::Zip;

use crate::{
    func::functions::input,
    layers::{
        layer::Layer,
        norm::LayerNorm,
        params::BasicTransofmerBlock_params
    },
    blocks::{
        attn::Attention,
        ff::FeedForward
    },
};

use std::rc::Rc;
use std::cell::RefCell;
use rayon::prelude::*;

pub struct BasicTransofmerBlock {
    pub operations: Vec<Box<dyn Layer>>,
}

impl BasicTransofmerBlock {
    pub fn new(
        params : BasicTransofmerBlock_params
    ) -> Self {
        let mut vec: Vec<Box<dyn Layer>> = Vec::new();
        vec.push(Box::new(LayerNorm {
             eps : params.eps_1, gamma : params.gamma_1, beta : params.beta_1, number : params.number_1,
            }));
        let attn1 = Attention::new(
            params.weights_1, params.bias_1, params.is_bias_1, 
            params.weights_2, params.bias_2, params.is_bias_2, 
            params.weights_3, params.bias_3, params.is_bias_3, 
            params.weights_4, params.bias_4, params.is_bias_4, 
            Rc::clone(&params.encoder_hidden_tensor_1), params.if_encoder_tensor_1,
            params.number_of_heads_1);
        vec.push(Box::new(attn1));
        vec.push(Box::new(LayerNorm {
            eps : params.eps_2, gamma : params.gamma_2, beta : params.beta_2, number : params.number_2
        }));
        let attn2 = Attention::new(
            params.weights_5, params.bias_5, params.is_bias_5, 
            params.weights_6, params.bias_6, params.is_bias_6, 
            params.weights_7, params.bias_7, params.is_bias_7, 
            params.weights_8, params.bias_8, params.is_bias_8, 
            Rc::clone(&params.encoder_hidden_tensor_2), params.if_encoder_tensor_2,
            params.number_of_heads_2);
        vec.push(Box::new(attn2));
        vec.push(Box::new(LayerNorm {
            eps : params.eps_3, gamma : params.gamma_3, beta : params.beta_3, number : params.number_3
        }));
        let ff = FeedForward::new(
            params.weights_ff1, params.bias_ff1, params.is_bias_ff1, 
            params.weights_ff2, params.bias_ff2, params.is_bias_ff2);
        vec.push(Box::new(ff));
        Self { operations: vec }
    }
}

impl Layer for BasicTransofmerBlock {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut residual = args.clone();
        let _ = operations[0].operation(args);
        let _ = operations[1].operation(args);
        Zip::from(&mut residual).and(&*args).for_each(|x, y| *x += *y);
        *args = residual.clone();
        let _ = operations[2].operation(&mut residual)?;
        let _ = operations[3].operation(&mut residual)?;
        Zip::from(&mut *args).and(&residual).for_each(|x, y| *x += *y);
        residual = args.clone();
        let _ = operations[4].operation(args)?;
        let _ = operations[5].operation(args)?;
        Zip::from(args).and(&residual).for_each(|x, y| *x += *y);
        Ok(())
    }
}

#[test]
fn test_btb_bchw_biased() {
    let weights_1 = input(r"C:\study\coursework\src\trash\test_btb1_attn1_q_test.safetensors".to_string()).unwrap();
    let weights_2= input(r"C:\study\coursework\src\trash\test_btb1_attn1_k_test.safetensors".to_string()).unwrap();
    let weights_3 = input(r"C:\study\coursework\src\trash\test_btb1_attn1_v_test.safetensors".to_string()).unwrap();
    let weights_4 = input(r"C:\study\coursework\src\trash\test_btb1_attn1_out_w_test.safetensors".to_string()).unwrap(); 
    let bias_4 = input(r"C:\study\coursework\src\trash\test_btb1_attn1_out_b_test.safetensors".to_string()).unwrap(); 

    let weights_5 = input(r"C:\study\coursework\src\trash\test_btb1_attn2_q_test.safetensors".to_string()).unwrap();
    let weights_6 = input(r"C:\study\coursework\src\trash\test_btb1_attn2_k_test.safetensors".to_string()).unwrap();
    let weights_7 = input(r"C:\study\coursework\src\trash\test_btb1_attn2_v_test.safetensors".to_string()).unwrap();
    let weights_8 = input(r"C:\study\coursework\src\trash\test_btb1_attn2_out_w_test.safetensors".to_string()).unwrap(); 
    let bias_8 = input(r"C:\study\coursework\src\trash\test_btb1_attn2_out_b_test.safetensors".to_string()).unwrap(); 

    let weights_ff1 = input(r"C:\study\coursework\src\trash\test_btb1_geglu_w_test.safetensors".to_string()).unwrap();
    let bias_ff1 = input(r"C:\study\coursework\src\trash\test_btb1_geglu_b_test.safetensors".to_string()).unwrap();

    let weights_ff2 = input(r"C:\study\coursework\src\trash\test_btb1_ff_w_test.safetensors".to_string()).unwrap();
    let bias_ff2 = input(r"C:\study\coursework\src\trash\test_btb1_ff_b_test.safetensors".to_string()).unwrap();

    let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_btb1_norm1_w_test.safetensors")).unwrap();
    let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_btb1_norm2_w_test.safetensors")).unwrap();
    let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_btb1_norm3_w_test.safetensors")).unwrap();

    let beta1 = input(format!(r"C:\study\coursework\src\trash\test_btb1_norm1_b_test.safetensors")).unwrap();
    let beta2 = input(format!(r"C:\study\coursework\src\trash\test_btb1_norm2_b_test.safetensors")).unwrap();
    let beta3 = input(format!(r"C:\study\coursework\src\trash\test_btb1_norm3_b_test.safetensors")).unwrap();



    let mut tensor = input(r"C:\study\coursework\src\trash\test_btb1_bchw_test.safetensors".to_string()).unwrap();
    let encoder = input(r"C:\study\coursework\src\trash\test_btb1_encoder.safetensors".to_string()).unwrap();
    let encoder_hidden = Rc::new(RefCell::new(encoder.remove_axis(ndarray::Axis(0))));
    let params = BasicTransofmerBlock_params {
    eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
    eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
    eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
    weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
    weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
    weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
    weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
    encoder_hidden_tensor_1 : Rc::clone(&encoder_hidden), if_encoder_tensor_1 : false, number_of_heads_1: 20, 

    weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
    weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
    weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
    weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
    encoder_hidden_tensor_2 : Rc::clone(&encoder_hidden), if_encoder_tensor_2 : true, number_of_heads_2: 20, 

    weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
    weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
    };
    let btb1 = BasicTransofmerBlock::new(params);

    let _ = btb1.operation(&mut tensor);
    let shape = tensor.shape();
    let py_tensor = input(r"C:\study\coursework\src\trash\test_btb1_bchw_output.safetensors".to_string()).unwrap();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-05);
                }
            }
        }
    }
}