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
        let mut res_vec = args.0.clone();
        let mut res_vec_shape = args.1.clone();

        for i in 0..operations.len() {
            if i == 1 {
                let input_tensor = 
                ndarray::Array4::from_shape_vec((res_vec_shape[0], res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]), res_vec)
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .into_shape_with_order([res_vec_shape[0], res_vec_shape[3] * res_vec_shape[2], res_vec_shape[1]])
                .unwrap()
                .as_standard_layout()
                .to_owned();
                // inner_dim = res_vec_shape[1]
                res_vec = input_tensor
                .as_standard_layout().
                to_owned().
                into_raw_vec_and_offset().0;
                res_vec_shape = vec![res_vec_shape[0], res_vec_shape[3] * res_vec_shape[2], res_vec_shape[1]].to_vec();
            }
            if i == 2{
                if (res_vec_shape.len() == 4 && res_vec_shape[0] == 1) {
                    res_vec_shape = vec![res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]].to_vec();
                }
            }
            let (temp_vec, temp_vec_shape) = operations[i].operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
        }
        for i in 0..1280 {
            print!("{:?} ",res_vec[i]);
        }
        print!("\n");
        for i in 1280..1280*2 {
            print!("{:?} ", res_vec[i]);
        }

        let mut output = ndarray::Array4::from_shape_vec((res_vec_shape[0], res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]), res_vec)
        .unwrap()
        .into_shape_with_order([args.1[0], args.1[2], args.1[3], args.1[1]]) // batches, height, weight, channels (innerd_dim)
        .unwrap()
        .as_standard_layout()
        .to_owned()
        .permuted_axes([0, 3, 1, 2])
        .as_standard_layout()
        .to_owned()
        ;
        let residual = ndarray::Array4::from_shape_vec([args.1[0], args.1[1], args.1[2], args.1[3]], args.0).unwrap();
        output = output + residual;
        res_vec = output.as_standard_layout().to_owned().into_raw_vec_and_offset().0;

        Ok((res_vec, res_vec_shape))
    }
}
