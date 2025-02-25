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
    downsample::DownSample2D
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
        Self { operations: vec }
    }
}

impl Layer for CrossAttnUpBlock2D {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut res_vec = args.0;
        let mut res_vec_shape = args.1;
        for layer in operations {
            let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
        } 
        Ok((res_vec, res_vec_shape))
    }
}
pub struct CrossAttnDownBlock2D {
    pub if_downsample2d : bool,
    pub operations : Vec<Box<dyn Layer>>,
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
        Self { operations: vec, if_downsample2d : params.is_downsample2d }
    }
}

impl Layer for CrossAttnDownBlock2D {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut res_vec = args.0;
        let mut res_vec_shape = args.1;
        for layer in operations {
            let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
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