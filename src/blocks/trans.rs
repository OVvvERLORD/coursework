use crate::{
    layers::{
        layer::Layer,
        norm::GroupNorm,
        linear::Linear,
        params::Transformer2D_params
    },
    blocks::{
        btb::BasicTransofmerBlock
    }
};

pub struct Transformer2D {
    pub operations : Vec<Box<dyn Layer>>,
    pub number_of_basic : usize,
}

impl Transformer2D {
    pub fn Transformer2D_constr(
        params : Transformer2D_params
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(GroupNorm {number_of_groups : params.number_of_groups, eps: params.eps, gamma : params.gamma, beta : params.beta}));
        vec.push(Box::new(Linear {weigths: params.weigths_in.clone(), weights_shape: params.weights_shape_in.clone(), bias: params.bias_in.clone(), bias_shape : params.bias_shape_in.clone(), is_bias : params.is_bias_in.clone()}));
        for param in &params.params_for_basics_vec {
            let basic_ins = BasicTransofmerBlock::BasicTransofmerBlock_constr(param);
            vec.push(Box::new(basic_ins));
        }
        vec.push(Box::new(Linear {weigths: params.weigths_in, weights_shape: params.weights_shape_in, bias: params.bias_in, bias_shape : params.bias_shape_in, is_bias : params.is_bias_in}));
        Self { operations: vec ,  number_of_basic : params.params_for_basics_vec.len()}
    }
}
impl Layer for Transformer2D {
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
