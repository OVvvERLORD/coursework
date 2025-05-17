use std::rc::Rc;
use std::cell::RefCell;

use crate::{
    layers::{
        layer::Layer,
        norm::GroupNorm,
        linear::Linear,
        params::{Transformer2D_params, BasicTransofmerBlock_params}
    },
    blocks::btb::BasicTransofmerBlock,
    func::functions::input
};
use rayon::prelude::*;
pub struct Transformer2D {
    pub operations : Vec<Box<dyn Layer>>,
    pub number_of_basic : usize,
}

impl Transformer2D {
    pub fn new(
        params : Transformer2D_params
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(
            GroupNorm {number_of_groups : params.number_of_groups, eps: params.eps, gamma : params.gamma, beta : params.beta}));
        vec.push(Box::new(Linear {
            weights: params.weigths_in, 
            bias: params.bias_in, 
            is_bias : params.is_bias_in
        }));
        let num = params.params_for_basics_vec.len();
        for param in params.params_for_basics_vec {
            let basic_ins = BasicTransofmerBlock::new(param);
            vec.push(Box::new(basic_ins));
        }
        vec.push(Box::new(Linear {
            weights: params.weigths_out, 
            bias: params.bias_out, 
            is_bias : params.is_bias_out
        }));
        Self { operations: vec ,  number_of_basic : num}
    }
}

impl Layer for Transformer2D {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let residual = args.view().to_owned();
        let initial_shape = residual.dim();
        for i in 0..operations.len() {
            if i == 1 {
                let shape = args.dim();
                *args = args.view()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .into_shape_with_order([1, shape.0, shape.3 * shape.2, shape.1])
                .unwrap()
                .to_owned();

            }
            let _ = operations[i].operation(args)?;
        }

        *args = args.view()
        .into_shape_with_order([initial_shape.0, initial_shape.2, initial_shape.3, initial_shape.1])
        .unwrap()
        .permuted_axes([0, 3, 1, 2])
        .to_owned();

        *args += &residual;
        if !args.is_standard_layout() {
            *args = args.as_standard_layout().to_owned();
        }
        Ok(())
    }
}

// #[test]
// fn test_transformer_unbiased() {
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_trans_test.safetensors".to_string()).unwrap();
//     let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_trans_encoder_test.safetensors".to_string()).unwrap();
//     let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
//     for i in 0..10 {
//         let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_q_test.safetensors", i)).unwrap();
//         let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_k_test.safetensors", i)).unwrap();
//         let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_v_test.safetensors", i)).unwrap();
//         let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_out_w_test.safetensors", i)).unwrap(); 
//         let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_out_b_test.safetensors", i)).unwrap(); 
    
//         let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_q_test.safetensors", i)).unwrap();
//         let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_k_test.safetensors", i)).unwrap();
//         let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_v_test.safetensors", i)).unwrap();
//         let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_out_w_test.safetensors", i)).unwrap(); 
//         let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_out_b_test.safetensors", i)).unwrap(); 
    
//         let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_geglu_w_test.safetensors", i)).unwrap();
//         let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_geglu_b_test.safetensors", i)).unwrap();
    
//         let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_ff_w_test.safetensors", i)).unwrap();
//         let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_ff_b_test.safetensors", i)).unwrap();
    
//         let btb1_params = BasicTransofmerBlock_params {
//         eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280, 
//         eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
//         eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,
//         weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
//         weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
//         weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
//         weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
//         encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 20,
    
//         weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
//         weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
//         weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
//         weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
//         encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 20,
    
//         weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
//         weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
//         };
//         param_vec.push(btb1_params);
//     }
//     let (weigths_in,  weights_shape_in) = input(r"C:\study\coursework\src\trash\test_trans_projin_w_test.safetensors".to_string()).unwrap(); 
//     let (bias_in, bias_shape_in) = input(r"C:\study\coursework\src\trash\test_trans_projin_b_test.safetensors".to_string()).unwrap(); 

//     let (weigths_out,  weights_shape_out) = input(r"C:\study\coursework\src\trash\test_trans_projout_w_test.safetensors".to_string()).unwrap(); 
//     let (bias_out, bias_shape_out) = input(r"C:\study\coursework\src\trash\test_trans_projout_b_test.safetensors".to_string()).unwrap(); 

//     let params = Transformer2D_params{
//         number_of_groups: 32, eps: 1e-05, gamma: 1., beta: 0.,
//         weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
//         weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
//         params_for_basics_vec: param_vec
//     };
//     let trans2d = Transformer2D::Transformer2D_constr(params);

//     let (res_vec, res_vec_shape) = trans2d.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_trans_output_test.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..py_vec_shape.len() {
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-01 );
//     }
// }

#[test]
fn test_transformer_large_biased() {
    let mut tensor = input(r"C:\study\coursework\src\trash\test_trans_large_input_test.safetensors".to_string()).unwrap();
    let encoder = input(r"C:\study\coursework\src\trash\test_trans_encoder_test.safetensors".to_string()).unwrap();
    let encoder = Rc::new(RefCell::new(encoder.remove_axis(ndarray::Axis(0))));
    let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
    for i in 0..10 {
        let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_q_test.safetensors", i)).unwrap();
        let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_k_test.safetensors", i)).unwrap();
        let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_v_test.safetensors", i)).unwrap();
        let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_out_w_test.safetensors", i)).unwrap(); 
        let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn1_out_b_test.safetensors", i)).unwrap(); 
    
        let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_q_test.safetensors", i)).unwrap();
        let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_k_test.safetensors", i)).unwrap();
        let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_v_test.safetensors", i)).unwrap();
        let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_out_w_test.safetensors", i)).unwrap(); 
        let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_attn2_out_b_test.safetensors", i)).unwrap(); 
    
        let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_geglu_w_test.safetensors", i)).unwrap();
        let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_geglu_b_test.safetensors", i)).unwrap();
    
        let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_ff_w_test.safetensors", i)).unwrap();
        let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_ff_b_test.safetensors", i)).unwrap();
    
        let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_norm1_w_test.safetensors", i)).unwrap();
        let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_norm2_w_test.safetensors", i)).unwrap();
        let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_norm3_w_test.safetensors", i)).unwrap();

        let beta1 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_norm1_b_test.safetensors", i)).unwrap();
        let beta2 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_norm2_b_test.safetensors", i)).unwrap();
        let beta3 = input(format!(r"C:\study\coursework\src\trash\test_trans_{}_norm3_b_test.safetensors", i)).unwrap();



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
    let weights_in = input(r"C:\study\coursework\src\trash\test_trans_projin_w_test.safetensors".to_string()).unwrap(); 
    let bias_in = input(r"C:\study\coursework\src\trash\test_trans_projin_b_test.safetensors".to_string()).unwrap(); 

    let weights_out = input(r"C:\study\coursework\src\trash\test_trans_projout_w_test.safetensors".to_string()).unwrap(); 
    let bias_out = input(r"C:\study\coursework\src\trash\test_trans_projout_b_test.safetensors".to_string()).unwrap(); 

    let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_trans_norm_w_test.safetensors")).unwrap();
    let beta_in = input(format!(r"C:\study\coursework\src\trash\test_trans_norm_b_test.safetensors")).unwrap();
    let params = Transformer2D_params{
        number_of_groups: 32, eps: 1e-05, gamma: gamma_in, beta: beta_in,
    weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
    weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
    params_for_basics_vec : param_vec
    };
    let trans2d = Transformer2D::new(params);

    let _ = trans2d.operation(&mut tensor).unwrap();
    let shape = tensor.shape();
    let py_tensor = input(r"C:\study\coursework\src\trash\test_trans_large_output_test.safetensors".to_string()).unwrap();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-02);
                }
            }
        }
    }
}