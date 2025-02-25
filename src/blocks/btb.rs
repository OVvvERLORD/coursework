use crate::{
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
    pub fn BasicTransofmerBlock_constr(
        params : &BasicTransofmerBlock_params
    ) -> Self {
        let mut vec: Vec<Box<dyn Layer>> = Vec::new();
        vec.push(Box::new(LayerNorm { eps : params.eps_1.clone(), gamma : params.gamma_1.clone(), beta : params.beta_1.clone(), number : params.number_1.clone()}));
        let attn1 = Attention::Attention_constr(
            params.weigths_1.clone(), params.weights_shape_1.clone(), params.bias_1.clone(), params.bias_shape_1.clone(), params.is_bias_1.clone(), 
            params.weigths_2.clone(), params.weights_shape_2.clone(), params.bias_2.clone(), params.bias_shape_2.clone(), params.is_bias_2.clone(), 
            params.weigths_3.clone(), params.weights_shape_3.clone(), params.bias_3.clone(), params.bias_shape_3.clone(), params.is_bias_3.clone(), 
            params.weigths_4.clone(), params.weights_shape_4.clone(), params.bias_4.clone(), params.bias_shape_4.clone(), params.is_bias_4.clone(), 
            Rc::clone(&params.encoder_hidden_tensor_1), params.if_encoder_tensor_1 , params.number_of_heads_1);
        vec.push(Box::new(attn1));
        vec.push(Box::new(LayerNorm { eps :params.eps_2.clone(), gamma : params.gamma_2.clone(), beta : params.beta_2.clone(), number : params.number_2.clone()}));
        let attn2 = Attention::Attention_constr(params.weigths_5.clone(), params.weights_shape_5.clone(), params.bias_5.clone(), params.bias_shape_5.clone(), params.is_bias_5.clone(), 
        params.weigths_6.clone(), params.weights_shape_6.clone(), params.bias_6.clone(), params.bias_shape_6.clone(), params.is_bias_6.clone(), 
        params.weigths_7.clone(), params.weights_shape_7.clone(), params.bias_7.clone(), params.bias_shape_7.clone(), params.is_bias_7.clone(), 
        params.weigths_8.clone(), params.weights_shape_8.clone(), params.bias_8.clone(), params.bias_shape_8.clone(), params.is_bias_8.clone(), 
        Rc::clone(&params.encoder_hidden_tensor_2), params.if_encoder_tensor_2 , params.number_of_heads_2);
        vec.push(Box::new(attn2));
        vec.push(Box::new(LayerNorm { eps : params.eps_3.clone(), gamma : params.gamma_3.clone(), beta : params.beta_3.clone(), number : params.number_3.clone()}));
        let ff = FeedForward::FeedForward_constr(params.weigths_ff1.clone(), params.weights_shape_ff1.clone(), params.bias_ff1.clone(), params.bias_shape_ff1.clone(), params.is_bias_ff1.clone(), params.weigths_ff2.clone(), params.weights_shape_ff2.clone(), params.bias_ff2.clone(), params.bias_shape_ff2.clone(), params.is_bias_ff2.clone());
        vec.push(Box::new(ff));
        Self { operations: vec }
    }
}
impl Layer for BasicTransofmerBlock {
    // fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    //     let operations = &self.operations;
        // let mut res_vec = args.0;
        // let mut res_vec_shape = args.1;
    //     for layer in operations {
    //         let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
    //         res_vec = temp_vec;
    //         res_vec_shape = temp_vec_shape;
    //     } 
    //     Ok((res_vec, res_vec_shape))
    // }
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut res_vec = args.0.clone();
        let mut res_vec_shape = args.1.clone();
        let (norm_vec, norm_vec_shape) = operations[0].operation(args)?;
        let (attn1_vec, attn1_vec_shape) = operations[1].operation((norm_vec, norm_vec_shape))?;
        let mut res_matr = ndarray::Array1::from_shape_vec((res_vec_shape[0] * res_vec_shape[1] * res_vec_shape[2] * res_vec_shape[3]), res_vec)?;
        let attn_matr = ndarray::Array1::from_shape_vec((attn1_vec_shape[0] * attn1_vec_shape[1] * attn1_vec_shape[2] * attn1_vec_shape[3]), attn1_vec)?;
        res_matr = res_matr.clone() + attn_matr;
        let temp_matr = res_matr.clone();
        res_vec = temp_matr.into_raw_vec_and_offset().0;
        let (norm_vec, norm_vec_shape) = operations[2].operation((res_vec, res_vec_shape.clone()))?;
        let (attn2_vec, attn2_vec_shape) = operations[3].operation((norm_vec, norm_vec_shape))?;
        let attn_matr = ndarray::Array1::from_shape_vec((attn2_vec_shape[0] * attn2_vec_shape[1] * attn2_vec_shape[2] * attn2_vec_shape[3]), attn2_vec)?;
        res_matr = res_matr.clone() + attn_matr;
        let temp_matr = res_matr.clone();
        res_vec = temp_matr.into_raw_vec_and_offset().0;
        let (norm_vec, norm_vec_shape) = operations[4].operation((res_vec, res_vec_shape.clone()))?;
        let (ff_vec, ff_vec_shape) = operations[5].operation((norm_vec, norm_vec_shape))?;
        let ff_matr = ndarray::Array1::from_shape_vec(ff_vec_shape[0] * ff_vec_shape[1] * ff_vec_shape[2] * ff_vec_shape[3], ff_vec)?;
        res_matr = res_matr.clone() + ff_matr;
        let temp_matr = res_matr.clone();
        res_vec = temp_matr.into_raw_vec_and_offset().0;
        Ok((res_vec, res_vec_shape))
    }
}