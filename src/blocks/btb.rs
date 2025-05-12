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

// #[test]
// fn test_btb_bse_unbiased() {
//     let (weigths_1, weights_shape_1) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_q_test.safetensors".to_string()).unwrap();
//     let (weigths_2, weights_shape_2) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_k_test.safetensors".to_string()).unwrap();
//     let (weigths_3, weights_shape_3) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_v_test.safetensors".to_string()).unwrap();
//     let (weigths_4, weights_shape_4) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_out_w_test.safetensors".to_string()).unwrap(); 
//     let (bias_4, bias_shape_4) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_out_b_test.safetensors".to_string()).unwrap(); 

//     let (weigths_5, weights_shape_5) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_q_test.safetensors".to_string()).unwrap();
//     let (weigths_6, weights_shape_6) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_k_test.safetensors".to_string()).unwrap();
//     let (weigths_7, weights_shape_7) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_v_test.safetensors".to_string()).unwrap();
//     let (weigths_8, weights_shape_8) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_out_w_test.safetensors".to_string()).unwrap(); 
//     let (bias_8, bias_shape_8) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_out_b_test.safetensors".to_string()).unwrap(); 

//     let (weigths_ff1, weights_shape_ff1) = input(r"C:\study\coursework\src\trash\test_btb1_geglu_w_test.safetensors".to_string()).unwrap();
//     let (bias_ff1, bias_shape_ff1) = input(r"C:\study\coursework\src\trash\test_btb1_geglu_b_test.safetensors".to_string()).unwrap();

//     let (weigths_ff2, weights_shape_ff2) = input(r"C:\study\coursework\src\trash\test_btb1_ff_w_test.safetensors".to_string()).unwrap();
//     let (bias_ff2, bias_shape_ff2) = input(r"C:\study\coursework\src\trash\test_btb1_ff_b_test.safetensors".to_string()).unwrap();

//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_btb1_test.safetensors".to_string()).unwrap();
//     let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_btb1_encoder.safetensors".to_string()).unwrap();
//     let encoder_hidden = Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec())));

//     let params = BasicTransofmerBlock_params{
//         eps_1 : 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280,
//         eps_2 : 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
//         eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,

//         weigths_1: weigths_1.to_vec(), weights_shape_1: weights_shape_1.to_vec(), bias_1: weigths_1.to_vec(), bias_shape_1: weights_shape_1.to_vec(), is_bias_1: false,
//         weigths_2: weigths_2.to_vec(), weights_shape_2: weights_shape_2.to_vec(), bias_2: weigths_2.to_vec(), bias_shape_2: weights_shape_2.to_vec(), is_bias_2: false,
//         weigths_3: weigths_3.to_vec(), weights_shape_3: weights_shape_3.to_vec(), bias_3: weigths_3.to_vec(), bias_shape_3: weights_shape_3.to_vec(), is_bias_3: false,
//         weigths_4: weigths_4.to_vec(), weights_shape_4: weights_shape_4.to_vec(), bias_4: bias_4.to_vec(), bias_shape_4: bias_shape_4.to_vec(), is_bias_4: true,
//         encoder_hidden_tensor_1: Rc::clone(&encoder_hidden), if_encoder_tensor_1 : false, number_of_heads_1: 20,
        
//         weigths_5: weigths_5.to_vec(), weights_shape_5: weights_shape_5.to_vec(), bias_5: weigths_5.to_vec(), bias_shape_5: weights_shape_5.to_vec(), is_bias_5: false,
//         weigths_6: weigths_6.to_vec(), weights_shape_6: weights_shape_6.to_vec(), bias_6: weigths_6.to_vec(), bias_shape_6: weights_shape_6.to_vec(), is_bias_6: false,
//         weigths_7: weigths_7.to_vec(), weights_shape_7: weights_shape_7.to_vec(), bias_7: weigths_7.to_vec(), bias_shape_7: weights_shape_7.to_vec(), is_bias_7: false,
//         weigths_8: weigths_8.to_vec(), weights_shape_8: weights_shape_8.to_vec(), bias_8: bias_8.to_vec(), bias_shape_8: bias_shape_8.to_vec(), is_bias_8: true,
//         encoder_hidden_tensor_2: Rc::clone(&encoder_hidden), if_encoder_tensor_2 : true, number_of_heads_2: 20,

//         weigths_ff1: weigths_ff1.to_vec(), weights_shape_ff1: weights_shape_ff1.to_vec(), bias_ff1: bias_ff1.to_vec(), bias_shape_ff1: bias_shape_ff1.to_vec(), is_bias_ff1: true,
//         weigths_ff2: weigths_ff2.to_vec(), weights_shape_ff2: weights_shape_ff2.to_vec(), bias_ff2: bias_ff2.to_vec(), bias_shape_ff2: bias_shape_ff2.to_vec(), is_bias_ff2: true,
//     };
//     let btb1 = BasicTransofmerBlock::BasicTransofmerBlock_constr(&params);
//     let (res_vec, res_vec_shape) = btb1.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_btb1_output_test.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..res_vec.len() {
//         assert!((res_vec[i] - py_vec[i]).abs() <= 1e-01);
//     }
// }