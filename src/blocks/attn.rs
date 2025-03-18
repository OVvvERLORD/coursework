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

use std::rc::Rc;
use std::cell::RefCell;

use std::f32::consts::E;

pub struct Attention {
    pub operations : Vec<Box<dyn Layer>>,
    pub encoder_hidden_tensor : Rc<RefCell<(Vec<f32>, Vec<usize>)>>,
    pub if_encoder_tensor : bool,
    pub heads : usize
}
impl Attention {
    pub fn Attention_constr(
        weigths_1: Vec<f32>, weights_shape_1 : Vec<usize>, bias_1: Vec<f32>, bias_shape_1 : Vec<usize>, is_bias_1 : bool,
        weigths_2: Vec<f32>, weights_shape_2 : Vec<usize>, bias_2: Vec<f32>, bias_shape_2 : Vec<usize>, is_bias_2 : bool,
        weigths_3: Vec<f32>, weights_shape_3 : Vec<usize>, bias_3: Vec<f32>, bias_shape_3 : Vec<usize>, is_bias_3 : bool,
        weigths_4: Vec<f32>, weights_shape_4 : Vec<usize>, bias_4: Vec<f32>, bias_shape_4 : Vec<usize>, is_bias_4 : bool,
        encoder_hidden_tensor : Rc<RefCell<(Vec<f32>, Vec<usize>)>>,
        if_encoder_tensor : bool, number_of_heads: usize
    ) -> Self {
        let mut vec : Vec<Box<dyn Layer>> = Vec::new();
        vec.push(Box::new(Linear {weigths : weigths_1, weights_shape : weights_shape_1, bias : bias_1, bias_shape : bias_shape_1, is_bias : is_bias_1}));
        vec.push(Box::new(Linear {weigths : weigths_2, weights_shape : weights_shape_2, bias : bias_2, bias_shape : bias_shape_2, is_bias : is_bias_2}));
        vec.push(Box::new(Linear {weigths : weigths_3, weights_shape : weights_shape_3, bias : bias_3, bias_shape : bias_shape_3, is_bias : is_bias_3}));
        vec.push(Box::new(Linear {weigths : weigths_4, weights_shape : weights_shape_4, bias : bias_4, bias_shape : bias_shape_4, is_bias : is_bias_4}));
        Self { operations: vec, encoder_hidden_tensor: encoder_hidden_tensor, if_encoder_tensor: if_encoder_tensor, heads: number_of_heads }
    }
}

impl Layer for Attention {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let norm_vec = args.0; 
        let norm_vec_shape = args.1;
        let input_tensor = if norm_vec_shape.len() !=3 
        {ndarray::Array3::from_shape_vec((norm_vec_shape[0],norm_vec_shape[1], norm_vec_shape[2] * norm_vec_shape[3]), norm_vec)?.permuted_axes([0, 2, 1]).as_standard_layout().to_owned()}
        else 
        {ndarray::Array3::from_shape_vec((norm_vec_shape[0], norm_vec_shape[1], norm_vec_shape[2]), norm_vec)?.as_standard_layout().to_owned()};

        let for_q_vec = input_tensor.clone().into_raw_vec_and_offset().0;
        let mut for_q_vec_shape = Vec::<usize>::new();
        for_q_vec_shape.insert(0, 1);
        let input_dim = input_tensor.dim();
        let (batch_size, sequence_length, _ )= if !self.if_encoder_tensor
        {input_dim}
        else
        {(self.encoder_hidden_tensor.borrow().1.clone()[0], self.encoder_hidden_tensor.borrow().1.clone()[1], self.encoder_hidden_tensor.borrow().1.clone()[2])}; 
        for_q_vec_shape.push(input_dim.0);
        for_q_vec_shape.push(input_dim.1);
        for_q_vec_shape.push(input_dim.2);
        let (q_vec, q_vec_shape) = &self.operations[0].operation((for_q_vec.clone(), for_q_vec_shape.clone()))?; 

        let for_k_vec = if !self.if_encoder_tensor
        {for_q_vec.clone()}
        else
        {self.encoder_hidden_tensor.borrow().0.clone()};
        let for_k_vec_shape = if !self.if_encoder_tensor
        {for_q_vec_shape.clone()}
        else
        {self.encoder_hidden_tensor.borrow().1.clone()};

        let for_v_vec = if !self.if_encoder_tensor
        {for_q_vec.clone()}
        else
        {self.encoder_hidden_tensor.borrow().0.clone()};
        let for_v_vec_shape = if !self.if_encoder_tensor
        {for_q_vec_shape.clone()}
        else
        {self.encoder_hidden_tensor.borrow().1.clone()};
        let (k_vec, k_vec_shape) = &self.operations[1].operation((for_k_vec, for_k_vec_shape))?;
        let (v_vec, v_vec_shape) = &self.operations[2].operation((for_v_vec, for_v_vec_shape))?;

        let inner_dim = k_vec_shape[3];
        let head_dim = inner_dim / self.heads;

        let query = ndarray::Array4::from_shape_vec((q_vec_shape[0], q_vec_shape[1], q_vec_shape[2], q_vec_shape[3]), q_vec.to_vec())
        .unwrap()
        .into_shape_with_order((batch_size, (q_vec_shape[0] * q_vec_shape[1] * q_vec_shape[2] * q_vec_shape[3]) / (batch_size * self.heads * head_dim), self.heads, head_dim)) // ПРОВЕРИТЬ, ЧТО ИМЕННО 2 !!!!!!!!!!
        .unwrap().permuted_axes([0, 2, 1, 3])
        .as_standard_layout()
        .to_owned();
        let key = ndarray::Array4::from_shape_vec((k_vec_shape[0], k_vec_shape[1], k_vec_shape[2], k_vec_shape[3]), k_vec.to_vec())
        .unwrap()
        .into_shape_with_order((batch_size, (k_vec_shape[0] * k_vec_shape[1] * k_vec_shape[2] * k_vec_shape[3]) / (batch_size * self.heads * head_dim), self.heads, head_dim))
        .unwrap().permuted_axes([0, 2, 1, 3])
        .as_standard_layout()
        .to_owned();

        let value = ndarray::Array4::from_shape_vec((v_vec_shape[0], v_vec_shape[1], v_vec_shape[2], v_vec_shape[3]), v_vec.to_vec())
        .unwrap()
        .into_shape_with_order((batch_size, (v_vec_shape[0] * v_vec_shape[1] * v_vec_shape[2] * v_vec_shape[3]) / (batch_size * self.heads * head_dim), self.heads, head_dim))
        .unwrap().permuted_axes([0, 2, 3, 1]) // be aware, it can be painful
        .as_standard_layout()
        .to_owned();

        let scale = 1. / (head_dim as f32).sqrt();
        let query_dim = query.dim();
        let key_dim = key.dim();
        let query_vec = query.into_raw_vec_and_offset().0;
        let key_vec = key.into_raw_vec_and_offset().0;
        let (mut qkt_vec, qkt_vec_shape) = 
        Tensor_Mul((
                query_vec, 
                vec![batch_size, self.heads, (query_dim.0 * query_dim.1 * query_dim.2 * query_dim.3) / (batch_size * self.heads * head_dim), head_dim].to_vec(), 
                key_vec, 
                vec![batch_size, self.heads, (key_dim.0 * key_dim.1 * key_dim.2 * key_dim.3) / (batch_size * self.heads * head_dim), head_dim].to_vec())).unwrap();


        for i in 0..qkt_vec.len() {
            qkt_vec[i] *= scale;
        }
        let limit = qkt_vec_shape[3];
        for i in (0..qkt_vec.len()).step_by(limit) {
            let mut sigma = 0_f32;
            for j in 0..limit {
                sigma += E.powf(qkt_vec[i + j]);
            }
            for j in 0..limit {
                qkt_vec[i + j] = E.powf(qkt_vec[i + j]) / sigma; 
            }
        }
        let value_dim = value.dim();
        let value_vec = value.into_raw_vec_and_offset().0;

        let (processor_vec, processor_vec_shape) = Tensor_Mul(
            (
                qkt_vec, 
                qkt_vec_shape,
                value_vec, 
                vec![batch_size,  self.heads, head_dim, (value_dim.0 * value_dim.1 * value_dim.2 * value_dim.3) / (batch_size * self.heads * head_dim)].to_vec()
            )
        ).unwrap();

        let processor_tensor = ndarray::Array4::from_shape_vec(
            (processor_vec_shape[0], processor_vec_shape[1], processor_vec_shape[2], processor_vec_shape[3]),
            processor_vec
        )
        .unwrap()
        .permuted_axes([0, 2, 1, 3])
        .as_standard_layout()
        .into_shape_with_order((batch_size, (processor_vec_shape[0] * processor_vec_shape[1] * processor_vec_shape[2] * processor_vec_shape[3]) / (batch_size * self.heads * head_dim), self.heads * head_dim)) // zdec' tozhe her'
        .unwrap()
        .as_standard_layout()
        .to_owned();

        let out_vec_dim = processor_tensor.dim();   
        let out_vec_shape = vec![1, out_vec_dim.0, out_vec_dim.1, out_vec_dim.2].to_vec();
        let out_vec = processor_tensor.into_raw_vec_and_offset().0;
        let (res_vec, res_shape ) = &self.operations[3].operation((out_vec, out_vec_shape))?;
        let mut res_vec = res_vec.to_vec();
        if norm_vec_shape.len() != 3 {
            let res_tensor = ndarray::Array3::from_shape_vec((res_shape[1], res_shape[2], res_shape[3]), res_vec.to_vec())
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_shape_with_order(norm_vec_shape.clone())
            .unwrap()
            .to_owned();
            res_vec = res_tensor.into_raw_vec_and_offset().0;
            
        }
        let res_shape = if norm_vec_shape.len() !=3 {norm_vec_shape} else { vec![res_shape[1], res_shape[2], res_shape[3]].to_vec()};
        
        Ok(((res_vec, res_shape)))
    }
}

pub struct CrossAttnUpBlock2D {
    pub operations : Vec<Box<dyn Layer>>,
    pub hidden_states : Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>,
}

impl CrossAttnUpBlock2D {
    pub fn CrossAttnUpBlock2D_constr(
        params : CrossAttnUpBlock2D_params
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::Resnet2d_constr(params.params_for_resnet1);
        vec.push(Box::new(resnet1));
        let transformer1 = Transformer2D::Transformer2D_constr(params.params_for_transformer1);
        vec.push(Box::new(transformer1));
        let resnet2 = Resnet2d::Resnet2d_constr(params.params_for_resnet2);
        vec.push(Box::new(resnet2));
        let transformer2 = Transformer2D::Transformer2D_constr(params.params_for_transformer2);
        vec.push(Box::new(transformer2));
        let resnet3 = Resnet2d::Resnet2d_constr(params.params_for_resnet3);
        vec.push(Box::new(resnet3));
        let transformer3 = Transformer2D::Transformer2D_constr(params.params_for_transformer3);
        vec.push(Box::new(transformer3));
        let upsample = Upsample2D::Upsample2D_constr(params.in_channels, params.out_channels, params.padding, params.stride, params.kernel_size, params.kernel_weights);
        vec.push(Box::new(upsample));
        Self { operations: vec , hidden_states: params.hidden_states}
    }
}

impl Layer for CrossAttnUpBlock2D {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut res_vec = args.0;
        let mut res_vec_shape = args.1;
        let mut hidden_states = self.hidden_states.borrow_mut();
        let mut hidden_idx = hidden_states.len() - 1;
        let mut idx = 2;
        let mut i = 0;
        for layer in operations {
            if idx == 2 && i != 6 {
                // let (hidden_vec, hidden_vec_shape) = &hidden_states[hidden_idx];
                // let _ = &mut hidden_states.pop();
                let (hidden_vec, hidden_vec_shape) = hidden_states.pop().unwrap();
                let hidden_tensor = ndarray::Array4::from_shape_vec((hidden_vec_shape[0], hidden_vec_shape[1], hidden_vec_shape[2], hidden_vec_shape[3]), hidden_vec.to_vec()).unwrap();
                let mut curr_tensor = ndarray::Array4::from_shape_vec((res_vec_shape[0], res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]), res_vec).unwrap();
                curr_tensor = ndarray::concatenate(ndarray::Axis(1), &[curr_tensor.view(), hidden_tensor.view()]).unwrap();
                let temp_shape = curr_tensor.dim();
                res_vec_shape = vec![temp_shape.0, temp_shape.1, temp_shape.2, temp_shape.3].to_vec();
                res_vec = curr_tensor.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
                idx = 0;
                hidden_idx = if hidden_idx > 0 {hidden_idx - 1} else {0};
            }
            let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
            idx += 1;
            i += 1;
        } 
        Ok((res_vec, res_vec_shape))
    }
}
pub struct CrossAttnDownBlock2D {
    pub if_downsample2d : bool,
    pub operations : Vec<Box<dyn Layer>>,
    pub hidden_states : Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>
}

impl CrossAttnDownBlock2D {
    pub fn CrossAttnDownBlock2D_constr(
        params : CrossAttnDownBlock2D_params
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::Resnet2d_constr(params.params_for_resnet1);
        vec.push(Box::new(resnet1));
        let transformer1 = Transformer2D::Transformer2D_constr(params.params_for_transformer1);
        vec.push(Box::new(transformer1));
        let resnet2 = Resnet2d::Resnet2d_constr(params.params_for_resnet2);
        vec.push(Box::new(resnet2));
        let transformer2 = Transformer2D::Transformer2D_constr(params.params_for_transformer2);
        vec.push(Box::new(transformer2));
        if params.is_downsample2d {
            let downsample2d = DownSample2D::DownSample2D_constr(params.in_channels, params.out_channels, params.padding, params.stride, params.kernel_size, params.kernel_weights);
            vec.push(Box::new(downsample2d));
        }
        Self { operations: vec, if_downsample2d : params.is_downsample2d, hidden_states: params.hidden_states }
    }
}

impl Layer for CrossAttnDownBlock2D {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut res_vec = args.0;
        let mut res_vec_shape = args.1;
        let mut output_states = self.hidden_states.borrow_mut();
        let mut idx = 0;
        for layer in operations {
            let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
            if idx == 1 {
                output_states.push((res_vec.clone(), res_vec_shape.clone()));
                idx = 0;
            } else {
                idx += 1;
            }
        } 
        if self.if_downsample2d {
            output_states.push((res_vec.clone(), res_vec_shape.clone()));
        }
        Ok((res_vec, res_vec_shape))
    }
}

#[test]
fn test_attn_bse_unbiased() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_test.safetensors".to_string()).unwrap();
    let (weigths_1, weights_shape_1) = input(r"C:\study\coursework\src\trash\test_attn1_q_test.safetensors".to_string()).unwrap();
    let (weigths_2, weights_shape_2) = input(r"C:\study\coursework\src\trash\test_attn1_k_test.safetensors".to_string()).unwrap();
    let (weigths_3, weights_shape_3) = input(r"C:\study\coursework\src\trash\test_attn1_v_test.safetensors".to_string()).unwrap();
    let (weigths_4, weights_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_w_test.safetensors".to_string()).unwrap(); 
    let (bias_4, bias_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_b_test.safetensors".to_string()).unwrap(); 
    let enc_placeholder = Rc::new(RefCell::new((Vec::<f32>::new(), Vec::<usize>::new())));
    let attn1 = Attention::Attention_constr(weigths_1.to_vec(), weights_shape_1.to_vec(), weigths_1.to_vec(), weights_shape_1.to_vec(), false, 
        weigths_2.to_vec(), weights_shape_2.to_vec(), weigths_2.to_vec(), weights_shape_2.to_vec(), false, 
        weigths_3.to_vec(), weights_shape_3.to_vec(), weigths_3.to_vec(), weights_shape_3.to_vec(), false, 
        weigths_4.to_vec(), weights_shape_4.to_vec(), bias_4.to_vec(), bias_shape_4.to_vec(), true, 
        enc_placeholder, false, 20);
    let (res_vec, res_vec_shape) = attn1.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_output_test.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}

#[test]
fn test_attn_bchw_unbiased() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_test_bchw.safetensors".to_string()).unwrap();
    let (weigths_1, weights_shape_1) = input(r"C:\study\coursework\src\trash\test_attn1_q_test.safetensors".to_string()).unwrap();
    let (weigths_2, weights_shape_2) = input(r"C:\study\coursework\src\trash\test_attn1_k_test.safetensors".to_string()).unwrap();
    let (weigths_3, weights_shape_3) = input(r"C:\study\coursework\src\trash\test_attn1_v_test.safetensors".to_string()).unwrap();
    let (weigths_4, weights_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_w_test.safetensors".to_string()).unwrap(); 
    let (bias_4, bias_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_b_test.safetensors".to_string()).unwrap(); 
    let enc_placeholder = Rc::new(RefCell::new((Vec::<f32>::new(), Vec::<usize>::new())));
    let attn1 = Attention::Attention_constr(weigths_1.to_vec(), weights_shape_1.to_vec(), weigths_1.to_vec(), weights_shape_1.to_vec(), false, 
        weigths_2.to_vec(), weights_shape_2.to_vec(), weigths_2.to_vec(), weights_shape_2.to_vec(), false, 
        weigths_3.to_vec(), weights_shape_3.to_vec(), weigths_3.to_vec(), weights_shape_3.to_vec(), false, 
        weigths_4.to_vec(), weights_shape_4.to_vec(), bias_4.to_vec(), bias_shape_4.to_vec(), true, 
        enc_placeholder, false, 20);
    let (res_vec, res_vec_shape) = attn1.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_output_bchw_test.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}

#[test]
fn test_attn_bse_encoder_unbiased() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_test.safetensors".to_string()).unwrap();
    let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_encoder_test.safetensors".to_string()).unwrap();
    let (weigths_1, weights_shape_1) = input(r"C:\study\coursework\src\trash\test_attn1_q_test.safetensors".to_string()).unwrap();
    let (weigths_2, weights_shape_2) = input(r"C:\study\coursework\src\trash\test_attn1_k_test.safetensors".to_string()).unwrap();
    let (weigths_3, weights_shape_3) = input(r"C:\study\coursework\src\trash\test_attn1_v_test.safetensors".to_string()).unwrap();
    let (weigths_4, weights_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_w_test.safetensors".to_string()).unwrap(); 
    let (bias_4, bias_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_b_test.safetensors".to_string()).unwrap(); 
    let enc_placeholder = Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec())));
    let attn1 = Attention::Attention_constr(weigths_1.to_vec(), weights_shape_1.to_vec(), weigths_1.to_vec(), weights_shape_1.to_vec(), false, 
        weigths_2.to_vec(), weights_shape_2.to_vec(), weigths_2.to_vec(), weights_shape_2.to_vec(), false, 
        weigths_3.to_vec(), weights_shape_3.to_vec(), weigths_3.to_vec(), weights_shape_3.to_vec(), false, 
        weigths_4.to_vec(), weights_shape_4.to_vec(), bias_4.to_vec(), bias_shape_4.to_vec(), true, 
        enc_placeholder, true, 20);
    let (res_vec, res_vec_shape) = attn1.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_output_encoder_test.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}

#[test]
fn test_attn_bchw_encoder_unbiased() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_test_bchw.safetensors".to_string()).unwrap();
    let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_encoder_test.safetensors".to_string()).unwrap();
    let (weigths_1, weights_shape_1) = input(r"C:\study\coursework\src\trash\test_attn1_q_test.safetensors".to_string()).unwrap();
    let (weigths_2, weights_shape_2) = input(r"C:\study\coursework\src\trash\test_attn1_k_test.safetensors".to_string()).unwrap();
    let (weigths_3, weights_shape_3) = input(r"C:\study\coursework\src\trash\test_attn1_v_test.safetensors".to_string()).unwrap();
    let (weigths_4, weights_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_w_test.safetensors".to_string()).unwrap(); 
    let (bias_4, bias_shape_4) = input(r"C:\study\coursework\src\trash\test_attn1_out_b_test.safetensors".to_string()).unwrap(); 
    let enc_placeholder = Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec())));
    let attn1 = Attention::Attention_constr(weigths_1.to_vec(), weights_shape_1.to_vec(), weigths_1.to_vec(), weights_shape_1.to_vec(), false, 
        weigths_2.to_vec(), weights_shape_2.to_vec(), weigths_2.to_vec(), weights_shape_2.to_vec(), false, 
        weigths_3.to_vec(), weights_shape_3.to_vec(), weigths_3.to_vec(), weights_shape_3.to_vec(), false, 
        weigths_4.to_vec(), weights_shape_4.to_vec(), bias_4.to_vec(), bias_shape_4.to_vec(), true, 
        enc_placeholder, true, 20);
    let (res_vec, res_vec_shape) = attn1.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_attn1_output_bchw_encoder_test.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}

#[test]
fn test_crossattnupblock_small_unbiased() {
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
    let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_input.safetensors".to_string()).unwrap();
    let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_encoder.safetensors".to_string()).unwrap();
    let (temb_vec, temb_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_temb.safetensors".to_string()).unwrap();
    let (res_hid_1, res_hid_1_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid1.safetensors".to_string()).unwrap();
    let (res_hid_2, res_hid_2_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid2.safetensors".to_string()).unwrap();
    let (res_hid_3, res_hid_3_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid3.safetensors".to_string()).unwrap();
    let (upsample_conv, _) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_upsample.safetensors".to_string()).unwrap();
    for i in 0..3 {
        let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv1.safetensors", i)).unwrap();
        let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv2.safetensors", i)).unwrap();
        let (kernel_weights_short, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv_short.safetensors", i)).unwrap();
        let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_temb_w.safetensors", i)).unwrap();
        let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 || i == 1 {2560} else {1920};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
            in_channels_1: in_ch, out_channels_1: 1280, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
            weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
            in_channels_2: 1280, out_channels_2: 1280, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
            is_shortcut: true,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
            time_emb: temb_vec.to_vec(), time_emb_shape: temb_vec_shape.to_vec()
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
            let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let btb1_params = BasicTransofmerBlock_params {
            eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280, 
            eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
            eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,
            weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
            weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
            weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
            weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
            encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 20,
        
            weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
            weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
            weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
            weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
            encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 20,
        
            weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
            weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
            };
            param_vec.push(btb1_params);
        }
        let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
            weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
            weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
            params_for_basics_vec: param_vec
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
        in_channels: 1280, out_channels: 1280, padding: 1, stride: 1, kernel_size: 3, kernel_weights: upsample_conv.to_vec(),
        hidden_states: Rc::new(RefCell::new(vec![(res_hid_1.to_vec(), res_hid_1_shape.to_vec()), (res_hid_2.to_vec(), res_hid_2_shape.to_vec()), (res_hid_3.to_vec(), res_hid_3_shape.to_vec())].to_vec()))
    };
    let crossattnupblock = CrossAttnUpBlock2D::CrossAttnUpBlock2D_constr(final_params);
    let (res_vec, res_vec_shape) = crossattnupblock.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_output.safetensors")).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..py_vec.len() {
        assert!((res_vec[i] - py_vec[i]).abs() <= 1e-02);
    }
}

#[test]
fn test_crossattndownblock_unbiased() {
    let mut trans1:Transformer2D;
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut trans2:Transformer2D;
    let mut trans2_params : Option<Transformer2D_params> = None;
    let mut resnet1:Resnet2d;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2:Resnet2d;
    let mut resnet2_params : Option<Resnet2d_params> = None;
    let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattndownblock_input.safetensors".to_string()).unwrap();
    let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattndownblock_encoder.safetensors".to_string()).unwrap();
    let (temb_vec, temb_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattndownblock_temb.safetensors".to_string()).unwrap();
    let (downsample_conv, _) = input(r"C:\study\coursework\src\trash\test_crossattndownblock_downsample.safetensors".to_string()).unwrap();

    for i in 0..2 {
        let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_conv1.safetensors", i)).unwrap();
        let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_conv2.safetensors", i)).unwrap();
        let (kernel_weights_short) = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_conv_short.safetensors", i)).unwrap().0}
        else
        {kernel_weights_1.clone()};
        let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_temb_w.safetensors", i)).unwrap();
        let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {320} else {640};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
            in_channels_1: in_ch, out_channels_1: 640, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
            weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
            in_channels_2: 640, out_channels_2: 640, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
            is_shortcut: shortcut_flag,
            in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
            time_emb: temb_vec.to_vec(), time_emb_shape: temb_vec_shape.to_vec()
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
            let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let btb1_params = BasicTransofmerBlock_params {
            eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 640, 
            eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 640,
            eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 640,
            weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
            weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
            weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
            weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
            encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 10,
        
            weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
            weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
            weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
            weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
            encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 10,
        
            weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
            weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
            };
            param_vec.push(btb1_params);
        }
        let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
            weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
            weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
            params_for_basics_vec: param_vec
        };
        if j == 0 {
            trans1_params = Some(params);
        } else{
            trans2_params = Some(params);
        }
    }
    let mut res_hidden_states = Rc::new(RefCell::new(Vec::<(Vec::<f32>, Vec::<usize>)>::new()));
    let final_params = CrossAttnDownBlock2D_params {
        is_downsample2d: true,
        params_for_transformer1: trans1_params.unwrap(),
        params_for_transformer2: trans2_params.unwrap(),
        params_for_resnet1: resnet1_params.unwrap(),
        params_for_resnet2: resnet2_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 3, kernel_weights: downsample_conv.to_vec(),
        hidden_states: Rc::clone(&res_hidden_states)
    };
    let crossattndownblock2d = CrossAttnDownBlock2D::CrossAttnDownBlock2D_constr(final_params);
    let (res_vec, res_vec_shape) = crossattndownblock2d.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_output.safetensors")).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..py_vec_shape.len() {
        let d = (res_vec[i] - py_vec[i]).abs();
        assert!(d <= 1e-04);
        assert!(!d.is_nan());
    }
    let (py_hidden1, py_hidden1_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_output_hidden1.safetensors")).unwrap();
    let testings = res_hidden_states.borrow_mut();
    assert!(testings[0].1 == py_hidden1_shape.to_vec());
    for i in 0..py_hidden1.len() {
        let d = (testings[0].0[i] - py_hidden1[i]).abs();
        assert!(d <= 1e-04);
        assert!(!d.is_nan());
    }
    let (py_hidden2, py_hidden2_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_output_hidden2.safetensors")).unwrap();
    assert!(testings[1].1 == py_hidden2_shape.to_vec());
    for i in 0..py_hidden2.len() {
        let d = (testings[1].0[i] - py_hidden2[i]).abs();
        assert!(d <= 1e-04);
        assert!(!d.is_nan());
    }
    let (py_hidden3, py_hidden3_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattndownblock_output_hidden3.safetensors")).unwrap();
    assert!(testings[2].1 == py_hidden3_shape.to_vec());
    for i in 0..py_hidden3.len() {
        let d = (testings[2].0[i] - py_hidden3[i]).abs();
        assert!(d <= 1e-04);
        assert!(!d.is_nan());
    }
    assert!(testings.len() == 3);
}

// #[test]
// fn test_crossattnupblock_large_unbiased() {
//     let mut trans1:Transformer2D;
//     let mut trans1_params : Option<Transformer2D_params> = None;
//     let mut trans2:Transformer2D;
//     let mut trans2_params : Option<Transformer2D_params> = None;
//     let mut trans3:Transformer2D;
//     let mut trans3_params : Option<Transformer2D_params> = None;
//     let mut resnet1:Resnet2d;
//     let mut resnet1_params : Option<Resnet2d_params> = None;
//     let mut resnet2:Resnet2d;
//     let mut resnet2_params : Option<Resnet2d_params> = None;
//     let mut resnet3:Resnet2d;
//     let mut resnet3_params : Option<Resnet2d_params> = None;
//     let (input_vec, input_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_input_large.safetensors".to_string()).unwrap();
//     print!("{:?} {:?} {:?}", input_vec[0], input_vec[1], input_vec[2]);
//     let (encoder_vec, encoder_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_encoder.safetensors".to_string()).unwrap();
//     let (temb_vec, temb_vec_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_temb.safetensors".to_string()).unwrap();
//     let (res_hid_1, res_hid_1_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid1_large.safetensors".to_string()).unwrap();
//     let (res_hid_2, res_hid_2_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid2_large.safetensors".to_string()).unwrap();
//     let (res_hid_3, res_hid_3_shape) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_reshid3_large.safetensors".to_string()).unwrap();
//     let (upsample_conv, _) = input(r"C:\study\coursework\src\trash\test_crossattnupblock_upsample.safetensors".to_string()).unwrap();
//     for i in 0..3 {
//         let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv1.safetensors", i)).unwrap();
//         let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv2.safetensors", i)).unwrap();
//         let (kernel_weights_short, _) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_conv_short.safetensors", i)).unwrap();
//         let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_temb_w.safetensors", i)).unwrap();
//         let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_resnet{}_temb_b.safetensors", i)).unwrap();
//         let in_ch = if i == 0 || i == 1 {2560} else {1920};
//         let resnet_par = Resnet2d_params{
//             number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
//             in_channels_1: in_ch, out_channels_1: 1280, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
//             weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
//             number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
//             in_channels_2: 1280, out_channels_2: 1280, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
//             is_shortcut: true,
//             in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
//             time_emb: temb_vec.to_vec(), time_emb_shape: temb_vec_shape.to_vec()
//         };
//         if i == 0 {
//             resnet1_params = Some(resnet_par);
//         } else if i == 1 {
//             resnet2_params = Some(resnet_par);
//         } else {
//             resnet3_params = Some(resnet_par);
//         }
//     }
//     for j in 0..3 {
//         let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
//         for i in 0..10 {
//             let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
//             let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
//             let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
//             let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
//             let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
//             let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
//             let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
//             let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
//             let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
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
//         let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projin_w_test.safetensors", j)).unwrap(); 
//         let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projin_b_test.safetensors", j)).unwrap(); 

//         let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projout_w_test.safetensors", j)).unwrap(); 
//         let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_trans{}_projout_b_test.safetensors", j)).unwrap(); 
//         let params = Transformer2D_params{
//             number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
//             weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
//             weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
//             params_for_basics_vec: param_vec
//         };
//         if j == 0 {
//             trans1_params = Some(params);
//         } else if j == 1 {
//             trans2_params = Some(params);
//         } else {
//             trans3_params = Some(params);
//         }
//     }
//     let final_params = CrossAttnUpBlock2D_params {
//         params_for_transformer1: trans1_params.unwrap(), 
//         params_for_transformer2: trans2_params.unwrap(),
//         params_for_transformer3: trans3_params.unwrap(),
//         params_for_resnet1: resnet1_params.unwrap(),
//         params_for_resnet2: resnet2_params.unwrap(),
//         params_for_resnet3: resnet3_params.unwrap(),
//         in_channels: 1280, out_channels: 1280, padding: 1, stride: 1, kernel_size: 3, kernel_weights: upsample_conv.to_vec(),
//         hidden_states: Rc::new(RefCell::new(vec![(res_hid_1.to_vec(), res_hid_1_shape.to_vec()), (res_hid_2.to_vec(), res_hid_2_shape.to_vec()), (res_hid_3.to_vec(), res_hid_3_shape.to_vec())].to_vec()))
//     };
//     let crossattnupblock = CrossAttnUpBlock2D::CrossAttnUpBlock2D_constr(final_params);
//     let (res_vec, res_vec_shape) = crossattnupblock.operation((input_vec.to_vec(), input_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_crossattnupblock_output_large.safetensors")).unwrap();
//     print!("{:?}\n", res_vec_shape);
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     // for i in 0..py_vec.len() {
//     //     // if (res_vec[i] - py_vec[i]).abs() > 1e-67{
//     //     //     print!("{:?} ", (res_vec[i] - py_vec[i]).abs());
//     //     // }
//     //     print!("{:?} ", (res_vec[i] - py_vec[i]).abs());
//     // }
// }