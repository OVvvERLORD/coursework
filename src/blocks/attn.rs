use crate::{
    func::functions::Tensor_Mul,
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

use std::f32::consts::E;

pub struct Attention {
    pub operations : Vec<Box<dyn Layer>>,
}
impl Attention {
    pub fn Attention_constr(
        weigths_1: Vec<f32>, weights_shape_1 : Vec<usize>, bias_1: Vec<f32>, bias_shape_1 : Vec<usize>, is_bias_1 : bool,
        weigths_2: Vec<f32>, weights_shape_2 : Vec<usize>, bias_2: Vec<f32>, bias_shape_2 : Vec<usize>, is_bias_2 : bool,
        weigths_3: Vec<f32>, weights_shape_3 : Vec<usize>, bias_3: Vec<f32>, bias_shape_3 : Vec<usize>, is_bias_3 : bool,
        weigths_4: Vec<f32>, weights_shape_4 : Vec<usize>, bias_4: Vec<f32>, bias_shape_4 : Vec<usize>, is_bias_4 : bool,
    ) -> Self {
        let mut vec : Vec<Box<dyn Layer>> = Vec::new();
        vec.push(Box::new(Linear {weigths : weigths_1, weights_shape : weights_shape_1, bias : bias_1, bias_shape : bias_shape_1, is_bias : is_bias_1}));
        vec.push(Box::new(Linear {weigths : weigths_2, weights_shape : weights_shape_2, bias : bias_2, bias_shape : bias_shape_2, is_bias : is_bias_2}));
        vec.push(Box::new(Linear {weigths : weigths_3, weights_shape : weights_shape_3, bias : bias_3, bias_shape : bias_shape_3, is_bias : is_bias_3}));
        vec.push(Box::new(Linear {weigths : weigths_4, weights_shape : weights_shape_4, bias : bias_4, bias_shape : bias_shape_4, is_bias : is_bias_4}));
        Self { operations: vec }
    }
}

impl Layer for Attention {
    // fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        // let norm_vec = args.0;
        // let norm_vec_shape = args.1;
        // let (q_vec, q_vec_shape) = &self.operations[0].operation((norm_vec.clone(), norm_vec_shape.clone()))?; 
        // let (k_vec, k_vec_shape) = &self.operations[1].operation((norm_vec.clone(), norm_vec_shape.clone()))?;
        // let (v_vec, v_vec_shape) = &self.operations[2].operation((norm_vec.clone(), norm_vec_shape.clone()))?;
    //     let Q = ndarray::Array2::from_shape_vec((q_vec_shape[0] * q_vec_shape[1] * q_vec_shape[2], q_vec_shape[3]), q_vec.to_vec())?;
    //     let K = ndarray::Array2::from_shape_vec((k_vec_shape[0] * k_vec_shape[1] * k_vec_shape[2], k_vec_shape[3]), k_vec.to_vec())?;
    //     let V = ndarray::Array2::from_shape_vec((v_vec_shape[0] * v_vec_shape[1] * v_vec_shape[2], v_vec_shape[3]), v_vec.to_vec())?;
    //     let mut matmul_q_k = Q.dot(&K.t());
    //     matmul_q_k = matmul_q_k / (k_vec_shape[3] as f32).sqrt();
    //     let mmqk_shape_vec = matmul_q_k.shape().to_vec();
    //     let limit = mmqk_shape_vec[3];
    //     let mut temp_vec = matmul_q_k.into_raw_vec_and_offset().0;
        // for i in (0..temp_vec.len()).step_by(limit) {
        //     let mut sigma = 0_f32;
        //     for j in 0..limit {
        //         sigma += E.powf(temp_vec[i + j]);
        //     }
        //     for j in 0..limit {
        //         temp_vec[i + j] = E.powf(temp_vec[i + j]) / sigma; 
        //     }
        // }
    //     let matmul_q_k =  ndarray::Array2::from_shape_vec((mmqk_shape_vec[0] * mmqk_shape_vec[1] * mmqk_shape_vec[2], mmqk_shape_vec[3] ), temp_vec)?;
    //     let res = matmul_q_k.clone().dot(&V);
    //     let res_vec = res.into_raw_vec_and_offset().0;
    //     let mut res_vec_shape = matmul_q_k.clone().shape().to_vec();
    //     res_vec_shape.insert(0, 1);
    //     Ok((res_vec, res_vec_shape))
    // }
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let norm_vec = args.0;
        let norm_vec_shape = args.1;
        let (q_vec, q_vec_shape) = &self.operations[0].operation((norm_vec.clone(), norm_vec_shape.clone()))?; 
        let (k_vec, k_vec_shape) = &self.operations[1].operation((norm_vec.clone(), norm_vec_shape.clone()))?;
        let (v_vec, v_vec_shape) = &self.operations[2].operation((norm_vec.clone(), norm_vec_shape.clone()))?;
        let (mut qkt_vec, qkt_shape) = Tensor_Mul((q_vec.to_vec(), q_vec_shape.to_vec(), k_vec.to_vec(), k_vec_shape.to_vec()))?;
        for i in 0..qkt_vec.len() {
            qkt_vec[i] /= (k_vec_shape[3] as f32).sqrt();
        }
        let limit = qkt_shape[3];
        for i in (0..qkt_vec.len()).step_by(limit) {
            let mut sigma = 0_f32;
            for j in 0..limit {
                sigma += E.powf(qkt_vec[i + j]);
            }
            for j in 0..limit {
                qkt_vec[i + j] = E.powf(qkt_vec[i + j]) / sigma; 
            }
        }
        let (qktv_vec, qktv_shape) = Tensor_Mul((qkt_vec.to_vec(), qkt_shape.to_vec(), v_vec.to_vec(), v_vec_shape.to_vec()))?;
        let (res_vec, res_shape ) = &self.operations[3].operation((qktv_vec, qkt_shape))?;
        Ok((res_vec.to_vec(), res_shape.to_vec()))
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