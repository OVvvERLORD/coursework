// use crate::{
//     layers::{
//         params::Resnet2d_params,
//         layer::Layer
//     },
//     blocks::resnet::Resnet2d,
//     func::functions::input
// };

// use std::rc::Rc;
// use std::cell::RefCell;

// pub struct UpBlock2d {
//     pub operations : Vec<Box<dyn Layer>>,
//     pub res_hidden_tensors: Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>,
// }

// impl UpBlock2d {
//     pub fn UpBlock2d_constr(
//         params_for_resnet1 : Resnet2d_params,
//         params_for_resnet2 : Resnet2d_params,
//         params_for_resnet3 : Resnet2d_params,
//         hidden_states : Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>
//     ) -> Self {
//         let mut vec = Vec::<Box<dyn Layer>>::new();
//         let resnet1 = Resnet2d::Resnet2d_constr(params_for_resnet1);
//         vec.push(Box::new(resnet1));
//         let resnet2 = Resnet2d::Resnet2d_constr(params_for_resnet2);
//         vec.push(Box::new(resnet2));
//         let resnet3 = Resnet2d::Resnet2d_constr(params_for_resnet3);
//         vec.push(Box::new(resnet3));
//         Self { operations: vec, res_hidden_tensors: hidden_states}
//     }
// }

// impl Layer for UpBlock2d {
//     fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
//         let operations = &self.operations;
//         let mut hidden_tensors = self.res_hidden_tensors.borrow_mut();
//         let mut res_vec = args.0;
//         let mut res_vec_shape = args.1;
//         let mut idx = &hidden_tensors.len() - 1;
//         for layer in operations {
//             let (res_hidden_vec, res_hidden_shape) = hidden_tensors.pop().unwrap();
//             let res_vec_matr = ndarray::Array4::from_shape_vec((res_vec_shape[0], res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]), res_vec.clone()).unwrap();
//             let hidden_vec_matr = ndarray::Array4::from_shape_vec((res_hidden_shape[0], res_hidden_shape[1], res_hidden_shape[2], res_hidden_shape[3]), res_hidden_vec.clone()).unwrap();
//             let done_vec_matr = ndarray::concatenate(ndarray::Axis(1), &[res_vec_matr.view(), hidden_vec_matr.view()]).unwrap().as_standard_layout().to_owned();
//             let done_vec_shape = done_vec_matr.dim();
//             let done_vec_shape = vec![done_vec_shape.0, done_vec_shape.1, done_vec_shape.2, done_vec_shape.3];
//             let done_vec = done_vec_matr.into_raw_vec_and_offset().0;
//             let (temp_vec, temp_vec_shape) = layer.operation((done_vec, done_vec_shape))?;
//             res_vec = temp_vec;
//             res_vec_shape = temp_vec_shape;
//             idx = if idx > 0 {idx - 1} else {0};
//         } 
//         Ok((res_vec, res_vec_shape))
//     }
// }

// #[test]
// fn test_unbiased () {
//     let mut hidden_vec = Vec::<(Vec<f32>, Vec<usize>)>::new();
//     let (temb, temb_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_temb.safetensors".to_string()).unwrap();
//     let (hidden_ins_vec, hidden_vec_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_res_hidden.safetensors".to_string()).unwrap();
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_test.safetensors".to_string()).unwrap();

//     hidden_vec.push((hidden_ins_vec.to_vec(), hidden_vec_shape.to_vec()));
//     hidden_vec.push((hidden_ins_vec.to_vec(), hidden_vec_shape.to_vec()));
//     hidden_vec.push((hidden_ins_vec.to_vec(), hidden_vec_shape.to_vec()));
//     let hidden_vec = Rc::new(RefCell::new(hidden_vec));
//     let (conv1_res1_vec, _) = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
//     let (conv2_res1_vec, _ ) = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
//     let (lin_res1_vec, lin_res1_vec_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
//     let (lin_res1_bias, lin_res1_bias_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_linear_bias.safetensors".to_string()).unwrap();
//     let (conv_short_res1_vec, _ ) = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_conv_short_weight.safetensors".to_string()).unwrap();
//     let time_emb = Rc::new(RefCell::new((temb.to_vec(), temb_shape.to_vec())));
//     let res1_params = Resnet2d_params{
//         number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
//         in_channels_1: 960, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res1_vec.to_vec(),
//         weigths: lin_res1_vec.to_vec(), weights_shape: lin_res1_vec_shape.to_vec(), bias: lin_res1_bias.to_vec(), bias_shape: lin_res1_bias_shape.to_vec(), is_bias: true,
//         number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
//         in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res1_vec.to_vec(),
//         is_shortcut: true,
//         in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_short_res1_vec.to_vec(),
//         time_emb: Rc::clone(&time_emb)
//     };

//     let (conv1_res2_vec, _) = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_conv1_weight.safetensors".to_string()).unwrap();
//     let (conv2_res2_vec, _ ) = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_conv2_weight.safetensors".to_string()).unwrap();
//     let (lin_res2_vec, lin_res2_vec_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_linear_weight.safetensors".to_string()).unwrap();
//     let (lin_res2_bias, lin_res2_bias_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_linear_bias.safetensors".to_string()).unwrap();
//     let (conv_short_res2_vec, _ ) = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_conv_short_weight.safetensors".to_string()).unwrap();
//     let res2_params = Resnet2d_params{
//         number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
//         in_channels_1: 640, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res2_vec.to_vec(),
//         weigths: lin_res2_vec.to_vec(), weights_shape: lin_res2_vec_shape.to_vec(), bias: lin_res2_bias.to_vec(), bias_shape: lin_res2_bias_shape.to_vec(), is_bias: true,
//         number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., time_emb: Rc::clone(&time_emb),
//         in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res2_vec.to_vec(),
//         is_shortcut: true,
//         in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_short_res2_vec.to_vec()
//     };


//     let (conv1_res3_vec, _) = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_conv1_weight.safetensors".to_string()).unwrap();
//     let (conv2_res3_vec, _ ) = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_conv2_weight.safetensors".to_string()).unwrap();
//     let (lin_res3_vec, lin_res3_vec_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_linear_weight.safetensors".to_string()).unwrap();
//     let (lin_res3_bias, lin_res3_bias_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_linear_bias.safetensors".to_string()).unwrap();
//     let (conv_short_res3_vec, _ ) = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_conv_short_weight.safetensors".to_string()).unwrap();
//     let res3_params = Resnet2d_params{
//         number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
//         in_channels_1: 640, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res3_vec.to_vec(),
//         weigths: lin_res3_vec.to_vec(), weights_shape: lin_res3_vec_shape.to_vec(), bias: lin_res3_bias.to_vec(), bias_shape: lin_res3_bias_shape.to_vec(), is_bias: true,
//         number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., time_emb: Rc::clone(&time_emb),
//         in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res3_vec.to_vec(),
//         is_shortcut: true,
//         in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_short_res3_vec.to_vec()
//     };

//     let upblock = UpBlock2d::UpBlock2d_constr(res1_params, res2_params, res3_params, hidden_vec);
//     let (res_vec, res_vec_shape) = upblock.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_upblock2d_out.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..res_vec.len() {
//         assert!((py_vec[i] - res_vec[i]).abs() <= 1e-03);
//     }
// }