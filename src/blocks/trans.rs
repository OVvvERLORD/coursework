// use std::rc::Rc;
// use std::cell::RefCell;

// use crate::{
//     layers::{
//         layer::Layer,
//         norm::GroupNorm,
//         linear::Linear,
//         params::{Transformer2D_params, BasicTransofmerBlock_params}
//     },
//     blocks::btb::BasicTransofmerBlock,
//     func::functions::input
// };

// pub struct Transformer2D {
//     pub operations : Vec<Box<dyn Layer>>,
//     pub number_of_basic : usize,
// }

// impl Transformer2D {
//     pub fn Transformer2D_constr(
//         params : Transformer2D_params
//     ) -> Self {
//         let mut vec = Vec::<Box<dyn Layer>>::new();
//         vec.push(Box::new(GroupNorm {number_of_groups : params.number_of_groups, eps: params.eps, gamma : params.gamma, beta : params.beta}));
//         vec.push(Box::new(Linear {weigths: params.weigths_in, weights_shape: params.weights_shape_in, bias: params.bias_in, bias_shape : params.bias_shape_in, is_bias : params.is_bias_in}));
//         for param in &params.params_for_basics_vec {
//             let basic_ins = BasicTransofmerBlock::BasicTransofmerBlock_constr(param);
//             vec.push(Box::new(basic_ins));
//         }
//         vec.push(Box::new(Linear {weigths: params.weigths_out, weights_shape: params.weights_shape_out, bias: params.bias_out, bias_shape : params.bias_shape_out, is_bias : params.is_bias_out}));
//         Self { operations: vec ,  number_of_basic : params.params_for_basics_vec.len()}
//     }
// }
// impl Layer for Transformer2D {
//     fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
//         let operations = &self.operations;
//         let mut res_vec = args.0.clone();
//         let mut res_vec_shape = args.1.clone();

//         for i in 0..operations.len() {
//             if i == 1 {
//                 let input_tensor = 
//                 ndarray::Array4::from_shape_vec((res_vec_shape[0], res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]), res_vec)
//                 .unwrap()
//                 .permuted_axes([0, 2, 3, 1])
//                 .as_standard_layout()
//                 .into_shape_with_order([res_vec_shape[0], res_vec_shape[3] * res_vec_shape[2], res_vec_shape[1]])
//                 .unwrap()
//                 .to_owned();
//                 // inner_dim = res_vec_shape[1]
//                 let input_tensor = if input_tensor.is_standard_layout() 
//                 {input_tensor}
//                 else
//                 {input_tensor.as_standard_layout().to_owned()};
//                 res_vec = input_tensor
//                 .into_raw_vec_and_offset().0;
//                 res_vec_shape = vec![res_vec_shape[0], res_vec_shape[3] * res_vec_shape[2], res_vec_shape[1]].to_vec();
//             }
//             else if i == 2{
//                 if res_vec_shape.len() == 4 && res_vec_shape[0] == 1 {
//                     res_vec_shape = vec![res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]].to_vec();
//                 }
//             }
//             let (temp_vec, temp_vec_shape) = operations[i].operation((res_vec.clone(), res_vec_shape.clone()))?;
//             res_vec = temp_vec;
//             res_vec_shape = temp_vec_shape;
//         }

//         let mut output = ndarray::Array4::from_shape_vec((res_vec_shape[0], res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]), res_vec)
//         .unwrap()
//         .into_shape_with_order([args.1[0], args.1[2], args.1[3], args.1[1]]) // batches, height, weight, channels (innerd_dim)
//         .unwrap()
//         .permuted_axes([0, 3, 1, 2])
//         ;

//         let temp_shape = output.dim();
//         res_vec_shape = vec![temp_shape.0, temp_shape.1, temp_shape.2, temp_shape.3].to_vec();
        
//         let residual = ndarray::Array4::from_shape_vec([args.1[0], args.1[1], args.1[2], args.1[3]], args.0).unwrap();
//         output = output + residual;
//         let output = if output.is_standard_layout() 
//         {output}
//         else
//         {output.as_standard_layout().to_owned()};
//         res_vec = output.into_raw_vec_and_offset().0;

//         Ok((res_vec, res_vec_shape))
//     }
// }

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

// #[test]
// fn test_transformer_large_unbiased() {
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_trans_large_input_test.safetensors".to_string()).unwrap();
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
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_trans_large_output_test.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..py_vec_shape.len() {
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-01);
//     }
    
// }