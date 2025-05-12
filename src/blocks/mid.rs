// use std::rc::Rc;
// use std::cell::RefCell;

// use crate::{
//     layers::{
//         params::{
//             Transformer2D_params,
//             Resnet2d_params,
//             BasicTransofmerBlock_params,
//         },
//         layer::Layer,
//     },
//     blocks::{
//         resnet::Resnet2d,
//         trans::Transformer2D
//     },
//     func::functions::input
// };

// pub struct mid_block {
//     pub operations : Vec<Box<dyn Layer>>,
// }

// impl mid_block {
//     pub fn mid_block_constr (
//         params_for_transformer2d :Transformer2D_params,
//         params_for_resnet_1: Resnet2d_params,
//         params_for_resnet_2: Resnet2d_params
//     ) -> Self {
//         let mut vec = Vec::<Box<dyn Layer>>::new();
//         let resnet1 = Resnet2d::Resnet2d_constr(params_for_resnet_1);
//         vec.push(Box::new(resnet1));
//         let transformer = Transformer2D::Transformer2D_constr(
//             params_for_transformer2d);
//         vec.push(Box::new(transformer));
//         let resnet2 = Resnet2d::Resnet2d_constr(params_for_resnet_2);
//         vec.push(Box::new(resnet2));
//         Self { operations: vec }
//     }
// }
// impl Layer for mid_block {
//     fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
//         let operations = &self.operations;
//         let mut res_vec = args.0;
//         let mut res_vec_shape = args.1;
//         for layer in operations {
//             let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
//             res_vec = temp_vec;
//             res_vec_shape = temp_vec_shape;
//         } 
//         Ok((res_vec, res_vec_shape))
//     }
// }

// #[test]
// fn test_crossattnmidblock_small_unbiased() {
//     let mut trans1:Transformer2D;
//     let mut trans1_params : Option<Transformer2D_params> = None;
//     let mut resnet1:Resnet2d;
//     let mut resnet1_params : Option<Resnet2d_params> = None;
//     let mut resnet2:Resnet2d;
//     let mut resnet2_params : Option<Resnet2d_params> = None;
//     let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_input.safetensors".to_string()).unwrap();
//     let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_encoder.safetensors".to_string()).unwrap();
//     let (temb_vec, temb_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_temb.safetensors".to_string()).unwrap();
//     let time_emb = Rc::new(RefCell::new((temb_vec.to_vec(), temb_vec_shape.to_vec())));
//     for i in 0..2 {
//         let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv1.safetensors", i)).unwrap();
//         let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_conv2.safetensors", i)).unwrap();
//         let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_temb_w.safetensors", i)).unwrap();
//         let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{}_temb_b.safetensors", i)).unwrap();
//         let in_ch = 1280;
//         let resnet_par = Resnet2d_params{
//             number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
//             in_channels_1: in_ch, out_channels_1: 1280, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
//             weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
//             number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
//             in_channels_2: 1280, out_channels_2: 1280, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
//             is_shortcut: false,
//             in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_1.to_vec(),
//             time_emb: Rc::clone(&time_emb)
//         };
//         if i == 0 {
//             resnet1_params = Some(resnet_par);
//         } else {
//             resnet2_params = Some(resnet_par);
//         }
//     }
//     let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
//     for i in 0..10 {
//         let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_q_test.safetensors", i)).unwrap();
//         let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_k_test.safetensors", i)).unwrap();
//         let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_v_test.safetensors", i)).unwrap();
//         let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_out_w_test.safetensors", i)).unwrap(); 
//         let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn1_out_b_test.safetensors", i)).unwrap(); 
    
//         let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_q_test.safetensors", i)).unwrap();
//         let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_k_test.safetensors", i)).unwrap();
//         let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_v_test.safetensors", i)).unwrap();
//         let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_out_w_test.safetensors", i)).unwrap(); 
//         let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_attn2_out_b_test.safetensors", i)).unwrap(); 
    
//         let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_geglu_w_test.safetensors", i)).unwrap();
//         let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_geglu_b_test.safetensors", i)).unwrap();
    
//         let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_ff_w_test.safetensors", i)).unwrap();
//         let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{}_ff_b_test.safetensors", i)).unwrap();
    
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
//     let (weigths_in,  weights_shape_in) = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_w_test.safetensors".to_string()).unwrap(); 
//     let (bias_in, bias_shape_in) = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_b_test.safetensors".to_string()).unwrap(); 

//     let (weigths_out,  weights_shape_out) = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_w_test.safetensors".to_string()).unwrap(); 
//     let (bias_out, bias_shape_out) = input(r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_b_test.safetensors".to_string()).unwrap(); 

//     let params = Transformer2D_params{
//         number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
//         weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
//         weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
//         params_for_basics_vec: param_vec
//     };
//     let mid = mid_block::mid_block_constr(params, resnet1_params.unwrap(), resnet2_params.unwrap());
//     let (res_vec, res_vec_shape) = mid.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnmidblock_output.safetensors")).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..py_vec.len() {
//         let d = (res_vec[i] - py_vec[i]).abs();
//         assert!(d <= 1e-03);
//         assert!(!d.is_nan());
//     }
// }