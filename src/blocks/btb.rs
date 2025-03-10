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
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut res_vec = args.0.clone();
        let res_vec_shape = args.1.clone();

        let (norm_vec, norm_vec_shape) = operations[0].operation(args)?;
        let (attn1_vec, attn1_vec_shape) = operations[1].operation((norm_vec, norm_vec_shape))?;

        let mut res_matr = if res_vec_shape.len() != 3 
        {ndarray::Array1::from_shape_vec(res_vec_shape[0] * res_vec_shape[1] * res_vec_shape[2] * res_vec_shape[3], res_vec)?}
        else
        {ndarray::Array1::from_shape_vec(res_vec_shape[0] * res_vec_shape[1] * res_vec_shape[2], res_vec)?};

        let attn_matr = if attn1_vec_shape.len() != 3 
        {ndarray::Array1::from_shape_vec(attn1_vec_shape[0] * attn1_vec_shape[1] * attn1_vec_shape[2] * attn1_vec_shape[3], attn1_vec)?}
        else
        {ndarray::Array1::from_shape_vec(attn1_vec_shape[0] * attn1_vec_shape[1] * attn1_vec_shape[2], attn1_vec)?};

        res_matr = res_matr + attn_matr;

        let temp_matr = res_matr.clone();
        res_vec = temp_matr.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        let (norm_vec, norm_vec_shape) = operations[2].operation((res_vec, res_vec_shape.clone()))?;
        let (attn2_vec, attn2_vec_shape) = operations[3].operation((norm_vec, norm_vec_shape))?;
        let attn_matr = if attn2_vec_shape.len() != 3 
        {ndarray::Array1::from_shape_vec(attn2_vec_shape[0] * attn2_vec_shape[1] * attn2_vec_shape[2] * attn2_vec_shape[3], attn2_vec)?}
        else
        {ndarray::Array1::from_shape_vec(attn2_vec_shape[0] * attn2_vec_shape[1] * attn2_vec_shape[2], attn2_vec)?};
        
        res_matr = res_matr + attn_matr;
        
        let temp_matr = res_matr.clone();
        res_vec = temp_matr.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        let (norm_vec, norm_vec_shape) = operations[4].operation((res_vec, res_vec_shape.clone()))?;
        let (ff_vec, ff_vec_shape) = operations[5].operation((norm_vec, norm_vec_shape))?;
        let ff_matr = ndarray::Array1::from_shape_vec(ff_vec_shape[0] * ff_vec_shape[1] * ff_vec_shape[2] * ff_vec_shape[3], ff_vec)?;
        res_matr = res_matr + ff_matr;
        let temp_matr = res_matr.clone();
        res_vec = temp_matr.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        Ok((res_vec, res_vec_shape))
    }
}

#[test]
fn test_btb_bse_unbiased() {
    let (weigths_1, weights_shape_1) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_q_test.safetensors".to_string()).unwrap();
    let (weigths_2, weights_shape_2) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_k_test.safetensors".to_string()).unwrap();
    let (weigths_3, weights_shape_3) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_v_test.safetensors".to_string()).unwrap();
    let (weigths_4, weights_shape_4) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_out_w_test.safetensors".to_string()).unwrap(); 
    let (bias_4, bias_shape_4) = input(r"C:\study\coursework\src\trash\test_btb1_attn1_out_b_test.safetensors".to_string()).unwrap(); 

    let (weigths_5, weights_shape_5) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_q_test.safetensors".to_string()).unwrap();
    let (weigths_6, weights_shape_6) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_k_test.safetensors".to_string()).unwrap();
    let (weigths_7, weights_shape_7) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_v_test.safetensors".to_string()).unwrap();
    let (weigths_8, weights_shape_8) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_out_w_test.safetensors".to_string()).unwrap(); 
    let (bias_8, bias_shape_8) = input(r"C:\study\coursework\src\trash\test_btb1_attn2_out_b_test.safetensors".to_string()).unwrap(); 

    let (weigths_ff1, weights_shape_ff1) = input(r"C:\study\coursework\src\trash\test_btb1_geglu_w_test.safetensors".to_string()).unwrap();
    let (bias_ff1, bias_shape_ff1) = input(r"C:\study\coursework\src\trash\test_btb1_geglu_b_test.safetensors".to_string()).unwrap();

    let (weigths_ff2, weights_shape_ff2) = input(r"C:\study\coursework\src\trash\test_btb1_ff_w_test.safetensors".to_string()).unwrap();
    let (bias_ff2, bias_shape_ff2) = input(r"C:\study\coursework\src\trash\test_btb1_ff_b_test.safetensors".to_string()).unwrap();

    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_btb1_test.safetensors".to_string()).unwrap();
    let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_btb1_encoder.safetensors".to_string()).unwrap();
    let encoder_hidden = Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec())));

    let params = BasicTransofmerBlock_params{
        eps_1 : 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280,
        eps_2 : 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
        eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,

        weigths_1: weigths_1.to_vec(), weights_shape_1: weights_shape_1.to_vec(), bias_1: weigths_1.to_vec(), bias_shape_1: weights_shape_1.to_vec(), is_bias_1: false,
        weigths_2: weigths_2.to_vec(), weights_shape_2: weights_shape_2.to_vec(), bias_2: weigths_2.to_vec(), bias_shape_2: weights_shape_2.to_vec(), is_bias_2: false,
        weigths_3: weigths_3.to_vec(), weights_shape_3: weights_shape_3.to_vec(), bias_3: weigths_3.to_vec(), bias_shape_3: weights_shape_3.to_vec(), is_bias_3: false,
        weigths_4: weigths_4.to_vec(), weights_shape_4: weights_shape_4.to_vec(), bias_4: bias_4.to_vec(), bias_shape_4: bias_shape_4.to_vec(), is_bias_4: true,
        encoder_hidden_tensor_1: Rc::clone(&encoder_hidden), if_encoder_tensor_1 : false, number_of_heads_1: 20,
        
        weigths_5: weigths_5.to_vec(), weights_shape_5: weights_shape_5.to_vec(), bias_5: weigths_5.to_vec(), bias_shape_5: weights_shape_5.to_vec(), is_bias_5: false,
        weigths_6: weigths_6.to_vec(), weights_shape_6: weights_shape_6.to_vec(), bias_6: weigths_6.to_vec(), bias_shape_6: weights_shape_6.to_vec(), is_bias_6: false,
        weigths_7: weigths_7.to_vec(), weights_shape_7: weights_shape_7.to_vec(), bias_7: weigths_7.to_vec(), bias_shape_7: weights_shape_7.to_vec(), is_bias_7: false,
        weigths_8: weigths_8.to_vec(), weights_shape_8: weights_shape_8.to_vec(), bias_8: bias_8.to_vec(), bias_shape_8: bias_shape_8.to_vec(), is_bias_8: true,
        encoder_hidden_tensor_2: Rc::clone(&encoder_hidden), if_encoder_tensor_2 : true, number_of_heads_2: 20,

        weigths_ff1: weigths_ff1.to_vec(), weights_shape_ff1: weights_shape_ff1.to_vec(), bias_ff1: bias_ff1.to_vec(), bias_shape_ff1: bias_shape_ff1.to_vec(), is_bias_ff1: true,
        weigths_ff2: weigths_ff2.to_vec(), weights_shape_ff2: weights_shape_ff2.to_vec(), bias_ff2: bias_ff2.to_vec(), bias_shape_ff2: bias_shape_ff2.to_vec(), is_bias_ff2: true,
    };
    let btb1 = BasicTransofmerBlock::BasicTransofmerBlock_constr(&params);
    let (res_vec, res_vec_shape) = btb1.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_btb1_output_test.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len() {
        assert!((res_vec[i] - py_vec[i]).abs() <= 1e-01);
    }
}