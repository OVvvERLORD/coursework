use crate::{
    layers::layer::Layer,
    func::functions::Tensor_Mul
};

pub struct Linear{
    pub weigths: Vec<f32>,
    pub weights_shape : Vec<usize>,
    pub bias: Vec<f32>,
    pub bias_shape : Vec<usize>,
    pub is_bias : bool,
}

impl Layer for Linear {
    // fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    //     let weight = self.weigths.clone();
    //     let bias = self.bias.clone();
    //     let input_matr = ndarray::Array2::from_shape_vec((args.1[0] * args.1[1] *args.1[2], args.1[3]), args.0.clone())?.to_owned();
    //     let weight_matr = ndarray::Array2::from_shape_vec((self.weights_shape[0],self.weights_shape[1]), weight)?.to_owned();
    //     let tr_weight_matr = weight_matr.t().to_owned();
    //     let mut res = input_matr.dot(&tr_weight_matr);
    //     if self.is_bias {
            // let bias_matr = ndarray::Array1::from_shape_vec(self.bias_shape[0], bias)?;
            // res = res.clone() + bias_matr.broadcast(res.dim()).unwrap();
    //     }
    //     let test = res.shape().to_vec();
    //     let vec_res = res.into_raw_vec_and_offset().0;
    //     print!("{:?}", vec_res);
    //     Ok((vec_res, test))
    // }
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let weights = self.weigths.clone();
        let weights_shape = self.weights_shape.clone();
        let input = args.0;
        let input_shape = args.1;
        let (mut res_vec, res_vec_shape) = Tensor_Mul((input, input_shape, weights, weights_shape))?;
        if self.is_bias {
            let mut res_matr = ndarray::Array4::from_shape_vec((res_vec_shape[0],res_vec_shape[1], res_vec_shape[2], res_vec_shape[3]), res_vec)?;
            let bias_matr = ndarray::Array1::from_shape_vec(self.bias_shape[0], self.bias.clone())?;
            res_matr = res_matr.clone() + bias_matr.broadcast(res_matr.dim()).unwrap();
            res_vec = res_matr.into_raw_vec_and_offset().0;
        }
        Ok((res_vec, res_vec_shape))
    }
}

impl Linear {
    pub fn linear_constr(in_features : usize, out_features : usize, bias : bool) -> Self {
            let mut weights_shape : Vec<usize> = Vec::new();
            weights_shape.push(out_features);
            weights_shape.push(in_features);
            let mut weights_vec : Vec<f32> = Vec::new();
            for _ in 0..in_features*out_features {
                weights_vec.push(rand::random::<f32>());
            }
            let mut bias_shape : Vec<usize> = Vec::new();
            let mut bias_vec : Vec<f32> = Vec::new();
            if bias {
                bias_shape.push(out_features);
                for _ in 0..out_features {
                    bias_vec.push(rand::random::<f32>());
                }
            }
            Self { weigths: weights_vec, weights_shape: weights_shape, bias: bias_vec, bias_shape: bias_shape, is_bias: bias }
    }
}