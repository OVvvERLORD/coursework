use crate::{
    layers::layer::Layer,
    func::functions::{Tensor_Mul, input, output}
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
        let mut weights_shape = self.weights_shape.clone();
        let input = args.0;
        let input_shape = args.1;
        if weights_shape.len() != 4 {
            weights_shape.insert(0, 1);
            weights_shape.insert(0, 1);
        }
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

#[test]
fn test_linear_big_unbiased_sym(){
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_inp_linear.safetensors".to_string()).unwrap();
    let (lin_w, lin_w_shape) = input(r"C:\study\coursework\src\trash\test_weight_linear.safetensors".to_string()).unwrap();
    let lin = Linear{weigths: lin_w.to_vec(), weights_shape: lin_w_shape.to_vec(), bias : lin_w.to_vec(), bias_shape: lin_w_shape.to_vec(), is_bias: false};
    let (res_vec, res_vec_shape) = lin.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let _ = output(r"C:\study\coursework\src\trash\test_linear_rust.safetensors".to_string(), res_vec.clone(), res_vec_shape.clone()).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_linear_python.safetensors".to_string()).unwrap();
    assert!(res_vec_shape == py_vec_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-67 );
    }
}

#[test]
fn test_linear_big_biased_sym() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_inp_bias_linear.safetensors".to_string()).unwrap();
    let (lin_w, lin_w_shape) = input(r"C:\study\coursework\src\trash\test_weight_bias_linear.safetensors".to_string()).unwrap();
    let (lin_b, lin_b_shape) = input(r"C:\study\coursework\src\trash\test_bias_linear.safetensors".to_string()).unwrap();
    let lin = Linear{weigths: lin_w.to_vec(), weights_shape: lin_w_shape.to_vec(), bias : lin_b.to_vec(), bias_shape: lin_b_shape.to_vec(), is_bias: true};
    let (res_vec, res_vec_shape) = lin.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let _ = output(r"C:\study\coursework\src\trash\test_linear_bias_rust.safetensors".to_string(), res_vec.clone(), res_vec_shape.clone()).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_linear_bias_python.safetensors".to_string()).unwrap();
    assert!(res_vec_shape == py_vec_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-67 );
    }
}

#[test]
fn test_linear_big_unbiased_unsym(){
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_inp_unsym_linear.safetensors".to_string()).unwrap();
    let (lin_w, lin_w_shape) = input(r"C:\study\coursework\src\trash\test_weight_unsym_linear.safetensors".to_string()).unwrap();
    let lin = Linear{weigths: lin_w.to_vec(), weights_shape: lin_w_shape.to_vec(), bias : lin_w.to_vec(), bias_shape: lin_w_shape.to_vec(), is_bias: false};
    let (res_vec, res_vec_shape) = lin.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let _ = output(r"C:\study\coursework\src\trash\test_linear_unsym_rust.safetensors".to_string(), res_vec.clone(), res_vec_shape.clone()).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_linear_unsym_python.safetensors".to_string()).unwrap();
    assert!(res_vec_shape == py_vec_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-67 );
    }
}

#[test]
fn test_linear_big_biased_unsym() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_inp_unsym_bias_linear.safetensors".to_string()).unwrap();
    let (lin_w, lin_w_shape) = input(r"C:\study\coursework\src\trash\test_weight_unsym_bias_linear.safetensors".to_string()).unwrap();
    let (lin_b, lin_b_shape) = input(r"C:\study\coursework\src\trash\test_unsym_bias_linear.safetensors".to_string()).unwrap();
    let lin = Linear{weigths: lin_w.to_vec(), weights_shape: lin_w_shape.to_vec(), bias : lin_b.to_vec(), bias_shape: lin_b_shape.to_vec(), is_bias: true};
    let (res_vec, res_vec_shape) = lin.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let _ = output(r"C:\study\coursework\src\trash\test_linear_unsym_bias_rust.safetensors".to_string(), res_vec.clone(), res_vec_shape.clone()).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_linear_unsym_bias_python.safetensors".to_string()).unwrap();
    assert!(res_vec_shape == py_vec_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-67 );
    }
}