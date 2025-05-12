// use crate::{
//     layers::{
//         layer::Layer,
//         params::{
//             Resnet2d_params,
//             CrossAttnDownBlock2D_params,
//             BasicTransofmerBlock_params,
//             Transformer2D_params
//         },
    
//     },
//     func::functions::input,
//     blocks::{
//         downblock::DownBlock2D,
//         attn::CrossAttnDownBlock2D
//     }
// };
// use std::rc::Rc;
// use std::cell::RefCell;

// pub struct Down_blocks {
//     pub operations : Vec<Box<dyn Layer>>,
//     pub hidden_states : Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>
// }

// impl Down_blocks {
//     pub fn Down_blocks_constr (
//         params_for_resnet1 : Resnet2d_params,
//         params_for_resnet2 : Resnet2d_params,
//         in_channels : usize, out_channels : usize, padding : i32, stride : i32, kernel_size : usize, kernel_weights : Vec<f32>,
//         params_for_crattbl1 : CrossAttnDownBlock2D_params,
//         params_for_crattbl2 : CrossAttnDownBlock2D_params,
//         hidden_states : Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>
//     ) -> Self {
//         let mut vec = Vec::<Box<dyn Layer>>::new();
//         let downblock = DownBlock2D::DownBlock2D_constr(params_for_resnet1, params_for_resnet2, in_channels, out_channels, padding, stride, kernel_size, kernel_weights, Rc::clone(&hidden_states));
//         vec.push(Box::new(downblock));
//         let crossattnblock1 = CrossAttnDownBlock2D::CrossAttnDownBlock2D_constr(params_for_crattbl1);
//         vec.push(Box::new(crossattnblock1));
//         let crossattnblock2 = CrossAttnDownBlock2D::CrossAttnDownBlock2D_constr(params_for_crattbl2);
//         vec.push(Box::new(crossattnblock2));
//         Self { operations: vec, hidden_states: Rc::clone(&hidden_states) }
//     }
// }

// impl Layer for Down_blocks {
//     fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
//         let operations = &self.operations;
//         let mut res_vec = args.0;
//         let mut res_vec_shape = args.1;
//         {
//             let mut hidden_states = self.hidden_states.borrow_mut();
//             hidden_states.push((res_vec.clone(), res_vec_shape.clone()));
//         }
//         for layer in operations {
//             let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
//             res_vec = temp_vec;
//             res_vec_shape = temp_vec_shape;
//         } 
//         Ok((res_vec, res_vec_shape))
//     }
// }

// #[test]
// fn test_downblocks_large_unbiased() {
//     let (temb, temb_shape) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_temb.safetensors")).unwrap();
//     let (encoder_vec, encoder_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_encoder.safetensors")).unwrap();
//     let mut res_hidden_states = Rc::new(RefCell::new(Vec::<(Vec::<f32>, Vec::<usize>)>::new()));
//     let time_emb = Rc::new(RefCell::new((temb.to_vec(), temb_shape.to_vec())));

//     let (conv1_res1_vec, _) = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv1_weight.safetensors".to_string()).unwrap();
//     let (conv2_res1_vec, _ ) = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv2_weight.safetensors".to_string()).unwrap();
//     let (lin_res1_vec, lin_res1_vec_shape) = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_linear_weight.safetensors".to_string()).unwrap();
//     let (lin_res1_bias, lin_res1_bias_shape) = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_linear_bias.safetensors".to_string()).unwrap();
//     let (conv_down, _ ) = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample.safetensors".to_string()).unwrap();

//     let res1_params = Resnet2d_params{
//         number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
//         in_channels_1: 320, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res1_vec.to_vec(),
//         weigths: lin_res1_vec.to_vec(), weights_shape: lin_res1_vec_shape.to_vec(), bias: lin_res1_bias.to_vec(), bias_shape: lin_res1_bias_shape.to_vec(), is_bias: true,
//         number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
//         in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res1_vec.to_vec(),
//         is_shortcut: false,
//         in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_down.to_vec().clone(),
//         time_emb: Rc::clone(&time_emb)
//     };

//     let (conv1_res2_vec, _) = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
//     let (conv2_res2_vec, _ ) = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
//     let (lin_res2_vec, lin_res2_vec_shape) = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
//     let (lin_res2_bias, lin_res2_bias_shape) = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_linear_bias.safetensors".to_string()).unwrap();

//     let res2_params = Resnet2d_params{
//         number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
//         in_channels_1: 320, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res2_vec.to_vec(),
//         weigths: lin_res2_vec.to_vec(), weights_shape: lin_res2_vec_shape.to_vec(), bias: lin_res2_bias.to_vec(), bias_shape: lin_res2_bias_shape.to_vec(), is_bias: true,
//         number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., time_emb: Rc::clone(&time_emb),
//         in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res2_vec.to_vec(),
//         is_shortcut: false,
//         in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_down.to_vec().clone()
//     };

//     let (downsample_conv, _) = input(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_downsample.safetensors".to_string()).unwrap();
//     let mut trans1_params : Option<Transformer2D_params> = None;
//     let mut trans2_params : Option<Transformer2D_params> = None;
//     let mut resnet1_params : Option<Resnet2d_params> = None;
//     let mut resnet2_params : Option<Resnet2d_params> = None;

//     for i in 0..2 {
//         let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv1.safetensors", i)).unwrap();
//         let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv2.safetensors", i)).unwrap();
//         let (kernel_weights_short) = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv_short.safetensors", i)).unwrap().0}
//         else
//         {kernel_weights_1.clone()};
//         let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_temb_w.safetensors", i)).unwrap();
//         let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_temb_b.safetensors", i)).unwrap();
//         let in_ch = if i == 0 {320} else {640};
//         let shortcut_flag = if i == 0 {true} else {false};
//         let resnet_par = Resnet2d_params{
//             number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
//             in_channels_1: in_ch, out_channels_1: 640, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
//             weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
//             number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
//             in_channels_2: 640, out_channels_2: 640, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
//             is_shortcut: shortcut_flag,
//             in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
//             time_emb: Rc::clone(&time_emb)
//         };
//         if i == 0 {
//             resnet1_params = Some(resnet_par);
//         } else {
//             resnet2_params = Some(resnet_par);
//         } 
//     }
//     for j in 0..2 {
//         let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
//         for i in 0..2 {
//             let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
//             let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
//             let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
//             let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
//             let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
//             let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
//             let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
//             let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
//             let btb1_params = BasicTransofmerBlock_params {
//             eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 640, 
//             eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 640,
//             eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 640,
//             weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
//             weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
//             weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
//             weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
//             encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 10,
        
//             weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
//             weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
//             weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
//             weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
//             encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 10,
        
//             weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
//             weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
//             };
//             param_vec.push(btb1_params);
//         }
//         let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projin_w_test.safetensors", j)).unwrap(); 
//         let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projin_b_test.safetensors", j)).unwrap(); 

//         let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projout_w_test.safetensors", j)).unwrap(); 
//         let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projout_b_test.safetensors", j)).unwrap(); 
//         let params = Transformer2D_params{
//             number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
//             weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
//             weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
//             params_for_basics_vec: param_vec
//         };
//         if j == 0 {
//             trans1_params = Some(params);
//         } else{
//             trans2_params = Some(params);
//         }
//     }
//     let crossattndownblock2d_1_params = CrossAttnDownBlock2D_params {
//         is_downsample2d: true,
//         params_for_transformer1: trans1_params.unwrap(),
//         params_for_transformer2: trans2_params.unwrap(),
//         params_for_resnet1: resnet1_params.unwrap(),
//         params_for_resnet2: resnet2_params.unwrap(),
//         in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 3, kernel_weights: downsample_conv.to_vec(),
//         hidden_states: Rc::clone(&res_hidden_states)
//     };
//     let mut trans12_params : Option<Transformer2D_params> = None;
//     let mut trans22_params : Option<Transformer2D_params> = None;
//     let mut resnet12_params : Option<Resnet2d_params> = None;
//     let mut resnet22_params : Option<Resnet2d_params> = None;

//     for i in 0..2 {
//         let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv1.safetensors", i)).unwrap();
//         let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv2.safetensors", i)).unwrap();
//         let (kernel_weights_short) = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv_short.safetensors", i)).unwrap().0}
//         else
//         {kernel_weights_1.clone()};
//         let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_temb_w.safetensors", i)).unwrap();
//         let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_temb_b.safetensors", i)).unwrap();
//         let in_ch = if i == 0 {640} else {1280};
//         let shortcut_flag = if i == 0 {true} else {false};
//         let resnet_par = Resnet2d_params{
//             number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
//             in_channels_1: in_ch, out_channels_1: 1280, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
//             weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
//             number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
//             in_channels_2: 1280, out_channels_2: 1280, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
//             is_shortcut: shortcut_flag,
//             in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
//             time_emb: Rc::clone(&time_emb)
//         };
//         if i == 0 {
//             resnet12_params = Some(resnet_par);
//         } else {
//             resnet22_params = Some(resnet_par);
//         } 
//     }
//     for j in 0..2 {
//         let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
//         for i in 0..10 {
//             let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
//             let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
//             let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
//             let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
//             let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
//             let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
//             let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
//             let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
//             let btb1_params = BasicTransofmerBlock_params {
//             eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280, 
//             eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
//             eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,
//             weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
//             weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
//             weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
//             weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
//             encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 20,
        
//             weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
//             weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
//             weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
//             weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
//             encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 20,
        
//             weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
//             weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
//             };
//             param_vec.push(btb1_params);
//         }
//         let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projin_w_test.safetensors", j)).unwrap(); 
//         let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projin_b_test.safetensors", j)).unwrap(); 

//         let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projout_w_test.safetensors", j)).unwrap(); 
//         let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projout_b_test.safetensors", j)).unwrap(); 
//         let params = Transformer2D_params{
//             number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
//             weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
//             weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
//             params_for_basics_vec: param_vec
//         };
//         if j == 0 {
//             trans12_params = Some(params);
//         } else{
//             trans22_params = Some(params);
//         }
//     }
//     let crossattndownblock2d_2_params = CrossAttnDownBlock2D_params {
//         is_downsample2d: false,
//         params_for_transformer1: trans12_params.unwrap(),
//         params_for_transformer2: trans22_params.unwrap(),
//         params_for_resnet1: resnet12_params.unwrap(),
//         params_for_resnet2: resnet22_params.unwrap(),
//         in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 13123124, kernel_weights: downsample_conv.to_vec(),
//         hidden_states: Rc::clone(&res_hidden_states)
//     };
//     let (downblock2d_downsample, _) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample.safetensors")).unwrap();
//     let down_blocks = Down_blocks::Down_blocks_constr(
//         res1_params, res2_params, 
//         320, 320, 1, 2, 3, downblock2d_downsample.to_vec(), 
//         crossattndownblock2d_1_params, 
//         crossattndownblock2d_2_params, 
//         Rc::clone(&res_hidden_states));
//     let (input_vec, input_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_input.safetensors")).unwrap();
//     let (res_vec, res_vec_shape) = down_blocks.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_downblocks_output.safetensors")).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..py_vec.len() {
//         let d = (res_vec[i] - py_vec[i]).abs();
//         assert!(!d.is_nan());
//         assert!(d <= 1e-02);
//     }
//     let testings = res_hidden_states.borrow_mut();
//     assert!(testings.len() == 8);
// }