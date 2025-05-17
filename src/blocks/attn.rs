use ndarray::Zip;

use crate::{
    func::functions::{Tensor_Mul,input},
    layers::{
        layer::Layer,
        linear::Linear,
        params::{
            CrossAttnUpBlock2D_params,
            CrossAttnDownBlock2D_params
        },
        upsample::Upsample2D,
        downsample::DownSample2D,
        params::{
            Resnet2d_params,
            BasicTransofmerBlock_params,
            Transformer2D_params
        }
    },
    blocks::{
        resnet::Resnet2d,
        trans::Transformer2D
    }
    
};

use std::{rc::Rc, sync::atomic};
use std::cell::RefCell;
use rayon::prelude::*;
use std::f32::consts::E;

pub struct Attention {
    pub operations : Vec<Box<dyn Layer>>,
    pub encoder_hidden_tensor : Rc<RefCell<ndarray::Array3<f32>>>,
    pub if_encoder_tensor : bool,
    pub heads : usize
}

impl Attention {
    pub fn new(
        weights_1: ndarray::Array4<f32>, bias_1: ndarray::Array4<f32> , is_bias_1 : bool,
        weights_2: ndarray::Array4<f32>, bias_2: ndarray::Array4<f32>, is_bias_2 : bool,
        weights_3: ndarray::Array4<f32>, bias_3: ndarray::Array4<f32>, is_bias_3 : bool,
        weights_4: ndarray::Array4<f32>, bias_4: ndarray::Array4<f32>, is_bias_4 : bool,
        encoder_hidden_tensor : Rc<RefCell<ndarray::Array3<f32>>>,
        if_encoder_tensor : bool, number_of_heads: usize
    ) -> Self {
        let mut vec : Vec<Box<dyn Layer>> = Vec::new();
        vec.push(Box::new(Linear {weights : weights_1, bias : bias_1, is_bias : is_bias_1}));
        vec.push(Box::new(Linear {weights : weights_2, bias : bias_2, is_bias : is_bias_2}));
        vec.push(Box::new(Linear {weights : weights_3, bias : bias_3, is_bias : is_bias_3}));
        vec.push(Box::new(Linear {weights : weights_4, bias : bias_4, is_bias : is_bias_4}));
        Self { operations: vec, encoder_hidden_tensor: encoder_hidden_tensor, if_encoder_tensor: if_encoder_tensor, heads: number_of_heads }
    }
}

impl Layer for Attention {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let initial_shape = args.shape();
        let mut q_tensor:ndarray::Array4<f32> = if initial_shape[0] != 1 
        {args
        .clone()
        .into_shape_with_order((initial_shape[0], initial_shape[1], initial_shape[2] * initial_shape[3]))
        .unwrap()
        .permuted_axes([0, 2, 1])
        .as_standard_layout()
        .to_owned()
        .insert_axis(ndarray::Axis(0))
        }
        else
        {args.view().to_owned()};
        let mut k_tensor = if !self.if_encoder_tensor 
        {q_tensor.clone()}
        else
        {self.encoder_hidden_tensor.borrow().clone().insert_axis(ndarray::Axis(0))};

        let mut v_tensor = if !self.if_encoder_tensor 
        {q_tensor.clone()}
        else
        {self.encoder_hidden_tensor.borrow().clone().insert_axis(ndarray::Axis(0))};

        let (batch_size, _, _ )= if !self.if_encoder_tensor
        {
            (initial_shape[1], initial_shape[2] * initial_shape[3], initial_shape[1])
        }
        else
        {
            self.encoder_hidden_tensor.borrow().dim()
        }; 
        let _ = &self.operations[0].operation(&mut q_tensor)?; 

        let _ = &self.operations[1].operation(&mut k_tensor)?;
        let _ = &self.operations[2].operation(&mut v_tensor)?;

        let inner_dim = k_tensor.shape()[3];
        let head_dim = inner_dim / self.heads;


        let q_shape = q_tensor.dim();
        q_tensor = q_tensor
        .into_shape_with_order((batch_size, (q_shape.0 * q_shape.1 * q_shape.2 * q_shape.3) / (batch_size * self.heads * head_dim), self.heads, head_dim))
        .unwrap().permuted_axes([0, 2, 1, 3])
        .as_standard_layout()
        .to_owned();

        let k_shape = k_tensor.dim();
        k_tensor = k_tensor
        .into_shape_with_order((batch_size, (k_shape.0 * k_shape.1 * k_shape.2 * k_shape.3) / (batch_size * self.heads * head_dim), self.heads, head_dim))
        .unwrap().permuted_axes([0, 2, 1, 3])
        .as_standard_layout()
        .to_owned();

        let v_shape = v_tensor.dim();
        v_tensor = v_tensor
        .into_shape_with_order((batch_size, (v_shape.0 * v_shape.1 * v_shape.2 * v_shape.3) / (batch_size * self.heads * head_dim), self.heads, head_dim))
        .unwrap().permuted_axes([0, 2, 3, 1])
        .as_standard_layout()
        .to_owned();
        let scale = 1. / (head_dim as f32).sqrt();
        let _ = Tensor_Mul(&mut q_tensor, &k_tensor).unwrap();
        q_tensor.mapv_inplace(|x| x * scale);
        let q_shape = q_tensor.dim();
        let max_vals = q_tensor.map_axis(ndarray::Axis(3), |row| {
            row.fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        }).insert_axis(ndarray::Axis(3));

        // q_tensor.mapv_inplace(|x| x.exp());
        Zip::from(&mut q_tensor).and(&max_vals.broadcast(q_shape).unwrap()).for_each(|x, &m| {
            *x = (*x - m).exp();
        });
        let sigmas = q_tensor.sum_axis(ndarray::Axis(3)).insert_axis(ndarray::Axis(3));
        // let q_shape = q_tensor.dim();
        Zip::from(&mut q_tensor).and(&sigmas.broadcast(q_shape).unwrap()).for_each(|x, &s| {
            *x /= s;
        });

        let _ = Tensor_Mul(&mut q_tensor, &v_tensor).unwrap();
        
        let q_shape = q_tensor.dim();
        q_tensor = q_tensor
        .permuted_axes([0, 2, 1, 3])
        .as_standard_layout()
        .into_shape_with_order((batch_size, (q_shape.0 * q_shape.1 * q_shape.2 * q_shape.3) / (batch_size * self.heads * head_dim), self.heads * head_dim))
        .unwrap()
        .as_standard_layout()
        .to_owned()
        .insert_axis(ndarray::Axis(0));
        let _ = &self.operations[3].operation(&mut q_tensor)?;
        q_tensor = if !initial_shape[0] == 1 
        {
        q_tensor
        .permuted_axes([0, 1, 3, 2])
        .as_standard_layout()
        .into_shape_with_order((initial_shape[0], initial_shape[1], initial_shape[2], initial_shape[3]))
        .unwrap()
        .to_owned()
    }
        else 
        {q_tensor};

        *args = q_tensor;
        Ok(())
    }
}

pub struct CrossAttnUpBlock2D {
    pub operations : Vec<Box<dyn Layer>>,
    pub hidden_states : Rc<RefCell<Vec<ndarray::Array4<f32>>>>,
}

impl CrossAttnUpBlock2D {
    pub fn new(
        params : CrossAttnUpBlock2D_params
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::new(params.params_for_resnet1);
        vec.push(Box::new(resnet1));
        let transformer1 = Transformer2D::new(params.params_for_transformer1);
        vec.push(Box::new(transformer1));
        let resnet2 = Resnet2d::new(params.params_for_resnet2);
        vec.push(Box::new(resnet2));
        let transformer2 = Transformer2D::new(params.params_for_transformer2);
        vec.push(Box::new(transformer2));
        let resnet3 = Resnet2d::new(params.params_for_resnet3);
        vec.push(Box::new(resnet3));
        let transformer3 = Transformer2D::new(params.params_for_transformer3);
        vec.push(Box::new(transformer3));
        let upsample = Upsample2D::new(
            params.in_channels, 
            params.out_channels, 
            params.padding, 
            params.stride,  
            params.kernel_size, 
            params.kernel_weights,
            params.bias,
            params.is_bias
        );
        vec.push(Box::new(upsample));
        Self { operations: vec , hidden_states: params.hidden_states}
    }
}

impl Layer for CrossAttnUpBlock2D {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        // let mut res_vec = args.0;
        // let mut res_vec_shape = args.1;
        let mut hidden_states = self.hidden_states.borrow_mut();
        let mut hidden_idx = hidden_states.len() - 1;
        let mut idx = 2;
        let mut i = 0;
        for layer in operations {
            if idx == 2 && i != 6 {
                // let (hidden_vec, hidden_vec_shape) = &hidden_states[hidden_idx];
                // let _ = &mut hidden_states.pop();
                let hidden_tensor = hidden_states.pop().unwrap();
                // let hidden_tensor = ndarray::Array4::from_shape_vec((hidden_vec_shape[0], hidden_vec_shape[1], hidden_vec_shape[2], hidden_vec_shape[3]), hidden_vec.to_vec()).unwrap();
                // let mut curr_tensor = ndarray::Array4::from_shape_vec((res_vec_shape[0], res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]), res_vec).unwrap();
                *args = ndarray::concatenate(ndarray::Axis(1), &[args.view(), hidden_tensor.view()])
                .unwrap()
                .as_standard_layout()
                .to_owned();
                // let temp_shape = curr_tensor.dim();
                // res_vec_shape = vec![temp_shape.0, temp_shape.1, temp_shape.2, temp_shape.3].to_vec();
                // res_vec = curr_tensor.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
                idx = 0;
                hidden_idx = if hidden_idx > 0 {hidden_idx - 1} else {0};
            }
            let _= layer.operation(args)?;
            if args.par_iter().any(|&x| x.is_nan()){
                print!("    POSLE ETOI OPS CROSSATTNUPBLOCK NAN: {:?}\n",i);
            }
            // res_vec = temp_vec;
            // res_vec_shape = temp_vec_shape;
            idx += 1;
            i += 1;
        } 
        Ok(())
    }
}

pub struct CrossAttnDownBlock2D {
    pub if_downsample2d : bool,
    pub operations : Vec<Box<dyn Layer>>,
    pub hidden_states : Rc<RefCell<Vec<ndarray::Array4<f32>>>>
}

impl CrossAttnDownBlock2D {
    pub fn new(
        params : CrossAttnDownBlock2D_params
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::new(params.params_for_resnet1);
        vec.push(Box::new(resnet1));
        let transformer1 = Transformer2D::new(params.params_for_transformer1);
        vec.push(Box::new(transformer1));
        let resnet2 = Resnet2d::new(params.params_for_resnet2);
        vec.push(Box::new(resnet2));
        let transformer2 = Transformer2D::new(params.params_for_transformer2);
        vec.push(Box::new(transformer2));
        if params.is_downsample2d {
            let downsample2d = DownSample2D::new(
                params.in_channels, 
                params.out_channels, 
                params.padding, 
                params.stride, 
                params.kernel_size,
                params.kernel_weights,
                params.bias,
                params.is_bias
            );
            vec.push(Box::new(downsample2d));
        }
        Self { operations: vec, if_downsample2d : params.is_downsample2d, hidden_states: params.hidden_states }
    }
}

impl Layer for CrossAttnDownBlock2D {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        // let mut res_vec = args.0;
        // let mut res_vec_shape = args.1;
        let mut output_states = self.hidden_states.borrow_mut();
        let mut idx = 0;
        for layer in operations {
            let _ = layer.operation(args)?;
            // res_vec = temp_vec;
            // res_vec_shape = temp_vec_shape;
            if idx == 1 {
                output_states.push(args.clone());
                idx = 0;
            } else {
                idx += 1;
            }
        } 
        if self.if_downsample2d {
            output_states.push(args.clone());
        }
        Ok(())
    }
}

// #[test]
// fn test_attn_bse_unbiased() {
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_test.safetensors".to_string()).unwrap();
//     let (weigths_1, weights_shape_1) = input(r"C:\study\coursework\src\trash\test_attn1_q_test.safetensors".to_string()).unwrap();
//     let (weigths_2, weights_shape_2) = input(r"C:\study\coursework\src\trash\test_attn1_k_test.safetensors".to_string()).unwrap();
//     let (weigths_3, weights_shape_3) = input(r"C:\study\coursework\src\trash\test_attn1_v_test.safetensors".to_string()).unwrap();
//     let (weigths_4, weights_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_w_test.safetensors".to_string()).unwrap(); 
//     let (bias_4, bias_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_b_test.safetensors".to_string()).unwrap(); 
//     let enc_placeholder = Rc::new(RefCell::new((Vec::<f32>::new(), Vec::<usize>::new())));
//     let attn1 = Attention::Attention_constr(weigths_1.to_vec(), weights_shape_1.to_vec(), weigths_1.to_vec(), weights_shape_1.to_vec(), false, 
//         weigths_2.to_vec(), weights_shape_2.to_vec(), weigths_2.to_vec(), weights_shape_2.to_vec(), false, 
//         weigths_3.to_vec(), weights_shape_3.to_vec(), weigths_3.to_vec(), weights_shape_3.to_vec(), false, 
//         weigths_4.to_vec(), weights_shape_4.to_vec(), bias_4.to_vec(), bias_shape_4.to_vec(), true, 
//         enc_placeholder, false, 20);
//     let (res_vec, res_vec_shape) = attn1.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_output_test.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..res_vec.len() {
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
//     }
// }

#[test]
fn test_attn_bchw_biased() {
    let mut tensor = input(r"C:\study\coursework\src\trash\test_attn1_test_bchw.safetensors".to_string()).unwrap();
    let encoder = input(r"C:\study\coursework\src\trash\test_attn1_encoder_test.safetensors".to_string()).unwrap().remove_axis(ndarray::Axis(0));
    let q_w = input(r"C:\study\coursework\src\trash\test_attn1_q_test.safetensors".to_string()).unwrap();
    let k_w = input(r"C:\study\coursework\src\trash\test_attn1_k_test.safetensors".to_string()).unwrap();
    let v_w = input(r"C:\study\coursework\src\trash\test_attn1_v_test.safetensors".to_string()).unwrap();
    let out_w = input(r"C:\study\coursework\src\trash\test_attn1_out_w_test.safetensors".to_string()).unwrap(); 
    let out_b = input(r"C:\study\coursework\src\trash\test_attn1_out_b_test.safetensors".to_string()).unwrap(); 
    let enc_placeholder = Rc::new(RefCell::new(encoder));
    let attn1 = Attention::new(
        q_w.clone(), q_w, false, 
        k_w.clone(), k_w, false, 
        v_w.clone(), v_w, false, 
        out_w, out_b, true, 
        enc_placeholder,false, 20);
    let _ = attn1.operation(&mut tensor).unwrap();
    let shape = tensor.shape();
    let py_tensor = input(  r"C:\study\coursework\src\trash\test_attn1_output_bchw_test.safetensors".to_string()).unwrap();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-6);
                }
            }
        }
    }
}

// #[test]
// fn test_attn_bse_encoder_unbiased() {
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_test.safetensors".to_string()).unwrap();
//     let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_encoder_test.safetensors".to_string()).unwrap();
//     let (weigths_1, weights_shape_1) = input(r"C:\study\coursework\src\trash\test_attn1_q_test.safetensors".to_string()).unwrap();
//     let (weigths_2, weights_shape_2) = input(r"C:\study\coursework\src\trash\test_attn1_k_test.safetensors".to_string()).unwrap();
//     let (weigths_3, weights_shape_3) = input(r"C:\study\coursework\src\trash\test_attn1_v_test.safetensors".to_string()).unwrap();
//     let (weigths_4, weights_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_w_test.safetensors".to_string()).unwrap(); 
//     let (bias_4, bias_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_b_test.safetensors".to_string()).unwrap(); 
//     let enc_placeholder = Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec())));
//     let attn1 = Attention::Attention_constr(weigths_1.to_vec(), weights_shape_1.to_vec(), weigths_1.to_vec(), weights_shape_1.to_vec(), false, 
//         weigths_2.to_vec(), weights_shape_2.to_vec(), weigths_2.to_vec(), weights_shape_2.to_vec(), false, 
//         weigths_3.to_vec(), weights_shape_3.to_vec(), weigths_3.to_vec(), weights_shape_3.to_vec(), false, 
//         weigths_4.to_vec(), weights_shape_4.to_vec(), bias_4.to_vec(), bias_shape_4.to_vec(), true, 
//         enc_placeholder, true, 20);
//     let (res_vec, res_vec_shape) = attn1.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_output_encoder_test.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..res_vec.len() {
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
//     }
// }

#[test]
fn test_attn_bchw_encoder_biased() {
    let mut tensor = input(r"C:\study\coursework\src\trash\test_attn1_test_bchw.safetensors".to_string()).unwrap();
    let encoder = input(r"C:\study\coursework\src\trash\test_attn1_encoder_test.safetensors".to_string()).unwrap().remove_axis(ndarray::Axis(0));
    let q_w = input(r"C:\study\coursework\src\trash\test_attn1_q_test.safetensors".to_string()).unwrap();
    let k_w = input(r"C:\study\coursework\src\trash\test_attn1_k_test.safetensors".to_string()).unwrap();
    let v_w = input(r"C:\study\coursework\src\trash\test_attn1_v_test.safetensors".to_string()).unwrap();
    let out_w = input(r"C:\study\coursework\src\trash\test_attn1_out_w_test.safetensors".to_string()).unwrap(); 
    let out_b = input(r"C:\study\coursework\src\trash\test_attn1_out_b_test.safetensors".to_string()).unwrap(); 
    let enc_placeholder = Rc::new(RefCell::new(encoder));
    let attn1 = Attention::new(
        q_w.clone(), q_w, false, 
        k_w.clone(), k_w, false, 
        v_w.clone(), v_w, false, 
        out_w, out_b, true, 
        enc_placeholder,true, 20);
    let _ = attn1.operation(&mut tensor).unwrap();
    let shape = tensor.shape();
    let py_tensor = input( r"C:\study\coursework\src\trash\test_attn1_output_bchw_encoder_test.safetensors".to_string()).unwrap();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-6);
                }
            }
        }
    }
}

#[test]
fn test_crossattnupblock() {
    let mut trans1:Transformer2D;
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut trans2:Transformer2D;
    let mut trans2_params : Option<Transformer2D_params> = None;
    let mut trans3:Transformer2D;
    let mut trans3_params : Option<Transformer2D_params> = None;
    let mut resnet1:Resnet2d;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2:Resnet2d;
    let mut resnet2_params : Option<Resnet2d_params> = None;
    let mut resnet3:Resnet2d;
    let mut resnet3_params : Option<Resnet2d_params> = None;
    let mut tensor = input(r"C:\study\coursework\src\trash\test_crossattnupblock_input.safetensors".to_string()).unwrap();

    let encoder = input(r"C:\study\coursework\src\trash\test_crossattnupblock_encoder.safetensors".to_string()).unwrap();
    let encoder = Rc::new(RefCell::new(encoder.remove_axis(ndarray::Axis(0))));
    let temb = input(r"C:\study\coursework\src\trash\test_crossattnupblock_temb.safetensors".to_string()).unwrap();
    let time_emb = Rc::new(RefCell::new(temb));
    let hidden1 = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid1.safetensors".to_string()).unwrap();
    let hidden2 = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid2.safetensors".to_string()).unwrap();
    let hidden3 = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid3.safetensors".to_string()).unwrap();
    let kernel_up = input(r"C:\study\coursework\src\trash\test_crossattnupblock_upsample.safetensors".to_string()).unwrap();

    let cup_b =  input(r"C:\study\coursework\src\trash\test_crossattnupblock_upsample_b.safetensors".to_string()).unwrap();
    for i in 0..3 {
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_norm1.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_norm2.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_norm2_b.safetensors", i)).unwrap();
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv1.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv1_b.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv2.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv2_b.safetensors", i)).unwrap();
        let kernels = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv_short.safetensors", i)).unwrap();
        let cs_b = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv_short_b.safetensors", i)).unwrap();
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 || i == 1 {2560} else {1920};
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
            is_shortcut: true,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet1_params = Some(resnet_par);
        } else if i == 1 {
            resnet2_params = Some(resnet_par);
        } else {
            resnet3_params = Some(resnet_par);
        }
    }
    for j in 0..3 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..10 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();



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
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projin_b_test.safetensors", j)).unwrap(); 
        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_norm_b_test.safetensors", j)).unwrap(); 
        let weights_out= input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans1_params = Some(params);
        } else if j == 1 {
            trans2_params = Some(params);
        } else {
            trans3_params = Some(params);
        }
    }
    let final_params = CrossAttnUpBlock2D_params {
        params_for_transformer1: trans1_params.unwrap(), 
        params_for_transformer2: trans2_params.unwrap(),
        params_for_transformer3: trans3_params.unwrap(),
        params_for_resnet1: resnet1_params.unwrap(),
        params_for_resnet2: resnet2_params.unwrap(),
        params_for_resnet3: resnet3_params.unwrap(),
        in_channels: 1280, out_channels: 1280, padding: 1, stride: 1, kernel_size: 3, kernel_weights: kernel_up.into_raw_vec_and_offset().0,
        is_bias: true, bias: cup_b,
        hidden_states: Rc::new(RefCell::new(vec![hidden1, hidden2, hidden3].to_vec()))
    };
    let crossattnupblock = CrossAttnUpBlock2D::new(final_params);
    let _ = crossattnupblock.operation(&mut tensor);

    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_output.safetensors")).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-2);
                }
            }
        }
    }
}

#[test]
fn test_crossattndownblock() {
    let mut trans1:Transformer2D;
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut trans2:Transformer2D;
    let mut trans2_params : Option<Transformer2D_params> = None;
    let mut resnet1:Resnet2d;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2:Resnet2d;
    let mut resnet2_params : Option<Resnet2d_params> = None;
    let mut tensor= input(r"C:\study\coursework\src\trash\test_crossattndownblock_input.safetensors".to_string()).unwrap();
    let encoder= input(r"C:\study\coursework\src\trash\test_crossattndownblock_encoder.safetensors".to_string()).unwrap();
    let temb = input(r"C:\study\coursework\src\trash\test_crossattndownblock_temb.safetensors".to_string()).unwrap();
    let kernel_down = input(r"C:\study\coursework\src\trash\test_crossattndownblock_downsample.safetensors".to_string()).unwrap();
    let cd_b = input(r"C:\study\coursework\src\trash\test_crossattndownblock_downsample_b.safetensors".to_string()).unwrap();
    let time_emb = Rc::new(RefCell::new(temb));
    let encoder = Rc::new(RefCell::new(encoder.remove_axis(ndarray::Axis(0))));
    for i in 0..2 {
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_norm1.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_norm2.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_norm2_b.safetensors", i)).unwrap();
        let kernel1= input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_conv1.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_conv2.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_conv1_b.safetensors", i)).unwrap();
        let c2_b  = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_conv2_b.safetensors", i)).unwrap();
        let kernels = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_conv_short.safetensors", i)).unwrap()}
        else
        {kernel1.clone()};
        let cs_b = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_conv_short_b.safetensors", i)).unwrap()}
        else {c1_b.clone()};
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {320} else {640};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: in_ch, 
        out_channels_1: 640, 
        padding_1: 1, 
        stride_1 : 1, 
        kernel_size_1 : 3, 
        kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
        bias_c1: c1_b, is_bias_c1: true,
        weights: linear_weight, bias : linear_bias, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
        in_channels_2: 640, 
        out_channels_2: 640, 
        padding_2: 1, stride_2 : 1, 
        kernel_size_2 : 3, 
        kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
        bias_c2: c2_b, is_bias_c2: true,
        is_shortcut: shortcut_flag,
        in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
        bias_s: cs_b, is_bias_s: true,
        time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet1_params = Some(resnet_par);
        } else {
            resnet2_params = Some(resnet_par);
        } 
    }
    for j in 0..2 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..2 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();

            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();

            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder), if_encoder_tensor_1 : false, number_of_heads_1: 10, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder), if_encoder_tensor_2 : true, number_of_heads_2: 10, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_projout_b_test.safetensors", j)).unwrap(); 

        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_norm_w_test.safetensors", j)).unwrap();
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_norm_b_test.safetensors", j)).unwrap();
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans1_params = Some(params);
        } else{
            trans2_params = Some(params);
        }
    }
    let mut res_hidden_states = Rc::new(RefCell::new(Vec::<ndarray::Array4<f32>>::new()));
    let final_params = CrossAttnDownBlock2D_params {
        is_downsample2d: true,
        params_for_transformer1: trans1_params.unwrap(),
        params_for_transformer2: trans2_params.unwrap(),
        params_for_resnet1: resnet1_params.unwrap(),
        params_for_resnet2: resnet2_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 3, kernel_weights: kernel_down.into_raw_vec_and_offset().0,
        is_bias: true, bias: cd_b,
        hidden_states: Rc::clone(&res_hidden_states)
    };
    let crossattndownblock2d = CrossAttnDownBlock2D::new(final_params);
    let _ = crossattndownblock2d.operation(&mut tensor);
    let shape = tensor.shape();
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_output.safetensors")).unwrap();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-03);
                }
            }
        }
    }
    let py_hid_1 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_output_hidden1.safetensors")).unwrap();
    let testings = res_hidden_states.borrow_mut();
    let hid1 = &testings[0];
    let shape = hid1.shape();
    assert!(shape == py_hid_1.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((hid1[[i, j, r, k]] - py_hid_1[[i, j, r, k]]).abs() <= 1e-04);
                }
            }
        }
    }
    let py_hid_2 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_output_hidden2.safetensors")).unwrap();
    let hid2 = &testings[1];
    let shape = hid2.shape();
    assert!(shape == py_hid_2.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((hid2[[i, j, r, k]] - py_hid_2[[i, j, r, k]]).abs() <= 1e-03);
                }
            }
        }
    }

    let py_hid_3 = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_output_hidden3.safetensors")).unwrap();
    
    let hid3 = &testings[2];
    let shape = hid3.shape();
    assert!(shape == py_hid_3.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((hid3[[i, j, r, k]] - py_hid_3[[i, j, r, k]]).abs() <= 1e-03);
                }
            }
        }
    }

    assert!(testings.len() == 3);

}

// // #[test]
// // fn test_crossattnupblock_large_unbiased() {
// //     let mut trans1:Transformer2D;
// //     let mut trans1_params : Option<Transformer2D_params> = None;
// //     let mut trans2:Transformer2D;
// //     let mut trans2_params : Option<Transformer2D_params> = None;
// //     let mut trans3:Transformer2D;
// //     let mut trans3_params : Option<Transformer2D_params> = None;
// //     let mut resnet1:Resnet2d;
// //     let mut resnet1_params : Option<Resnet2d_params> = None;
// //     let mut resnet2:Resnet2d;
// //     let mut resnet2_params : Option<Resnet2d_params> = None;
// //     let mut resnet3:Resnet2d;
// //     let mut resnet3_params : Option<Resnet2d_params> = None;
// //     let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_input_large.safetensors".to_string()).unwrap();
// //     print!("{:?} {:?} {:?}", input_vec[0], input_vec[1], input_vec[2]);
// //     let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_encoder.safetensors".to_string()).unwrap();
// //     let (temb_vec, temb_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_temb.safetensors".to_string()).unwrap();
// //     let (res_hid_1, res_hid_1_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid1_large.safetensors".to_string()).unwrap();
// //     let (res_hid_2, res_hid_2_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid2_large.safetensors".to_string()).unwrap();
// //     let (res_hid_3, res_hid_3_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid3_large.safetensors".to_string()).unwrap();
// //     let (upsample_conv, _) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_upsample.safetensors".to_string()).unwrap();
// //     for i in 0..3 {
// //         let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv1.safetensors", i)).unwrap();
// //         let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv2.safetensors", i)).unwrap();
// //         let (kernel_weights_short, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv_short.safetensors", i)).unwrap();
// //         let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_temb_w.safetensors", i)).unwrap();
// //         let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_temb_b.safetensors", i)).unwrap();
// //         let in_ch = if i == 0 || i == 1 {2560} else {1920};
// //         let resnet_par = Resnet2d_params{
// //             number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
// //             in_channels_1: in_ch, out_channels_1: 1280, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
// //             weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
// //             number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
// //             in_channels_2: 1280, out_channels_2: 1280, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
// //             is_shortcut: true,
// //             in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
// //             time_emb: temb_vec.to_vec(), time_emb_shape: temb_vec_shape.to_vec()
// //         };
// //         if i == 0 {
// //             resnet1_params = Some(resnet_par);
// //         } else if i == 1 {
// //             resnet2_params = Some(resnet_par);
// //         } else {
// //             resnet3_params = Some(resnet_par);
// //         }
// //     }
// //     for j in 0..3 {
// //         let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
// //         for i in 0..10 {
// //             let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
// //             let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
// //             let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
// //             let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
// //             let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
// //             let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
// //             let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
// //             let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
// //             let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
// //             let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
// //             let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
// //             let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
// //             let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
// //             let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
// //             let btb1_params = BasicTransofmerBlock_params {
// //             eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280, 
// //             eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
// //             eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,
// //             weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
// //             weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
// //             weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
// //             weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
// //             encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 20,
        
// //             weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
// //             weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
// //             weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
// //             weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
// //             encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 20,
        
// //             weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
// //             weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
// //             };
// //             param_vec.push(btb1_params);
// //         }
// //         let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projin_w_test.safetensors", j)).unwrap(); 
// //         let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projin_b_test.safetensors", j)).unwrap(); 

// //         let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projout_w_test.safetensors", j)).unwrap(); 
// //         let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projout_b_test.safetensors", j)).unwrap(); 
// //         let params = Transformer2D_params{
// //             number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
// //             weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
// //             weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
// //             params_for_basics_vec: param_vec
// //         };
// //         if j == 0 {
// //             trans1_params = Some(params);
// //         } else if j == 1 {
// //             trans2_params = Some(params);
// //         } else {
// //             trans3_params = Some(params);
// //         }
// //     }
// //     let final_params = CrossAttnUpBlock2D_params {
// //         params_for_transformer1: trans1_params.unwrap(), 
// //         params_for_transformer2: trans2_params.unwrap(),
// //         params_for_transformer3: trans3_params.unwrap(),
// //         params_for_resnet1: resnet1_params.unwrap(),
// //         params_for_resnet2: resnet2_params.unwrap(),
// //         params_for_resnet3: resnet3_params.unwrap(),
// //         in_channels: 1280, out_channels: 1280, padding: 1, stride: 1, kernel_size: 3, kernel_weights: upsample_conv.to_vec(),
// //         hidden_states: Rc::new(RefCell::new(vec![(res_hid_1.to_vec(), res_hid_1_shape.to_vec()), (res_hid_2.to_vec(), res_hid_2_shape.to_vec()), (res_hid_3.to_vec(), res_hid_3_shape.to_vec())].to_vec()))
// //     };
// //     let crossattnupblock = CrossAttnUpBlock2D::CrossAttnUpBlock2D_constr(final_params);
// //     let (res_vec, res_vec_shape) = crossattnupblock.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
// //     let (py_vec, py_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_output_large.safetensors")).unwrap();
// //     print!("{:?}\n", res_vec_shape);
// //     assert!(py_vec_shape.to_vec() == res_vec_shape);
// //     // for i in 0..py_vec.len() {
// //     //     // if (res_vec[i] - py_vec[i]).abs() > 1e-67{
// //     //     //     print!("{:?} ", (res_vec[i] - py_vec[i]).abs());
// //     //     // }
// //     //     print!("{:?} ", (res_vec[i] - py_vec[i]).abs());
// //     // }
// // }