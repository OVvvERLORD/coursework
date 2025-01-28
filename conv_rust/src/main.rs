use cudarc::cudnn::result;
use cudarc::driver::CudaSlice;
use cudarc::{self, cudnn};
use rand_distr::num_traits::Pow;
use safetensors::serialize;
use safetensors::{SafeTensors};
use core::{f32, time};
use std::fs::File;
use std::io::{Read, Write};
use crate::f32::consts::E;
use rand;
use rand_distr::Distribution;
use std::sync::Arc;
use ndarray;
use statrs::function::erf::erf;

pub fn Tensor_Mul(args:(Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>)) ->  Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let A = ndarray::Array4::from_shape_vec((args.1[0], args.1[1], args.1[2], args.1[3]), args.0)?;
    let B = ndarray::Array4::from_shape_vec((args.3[0], args.3[1], args.3[2], args.3[3]), args.2)?;
    let mut result_shape = args.1;
    result_shape[3] = result_shape[2];
    let mut result = ndarray::Array4::<f32>::zeros((result_shape[0], result_shape[1], result_shape[2], result_shape[2]));
    for ((mut res, a_batch), b_batch) in result
        .axis_iter_mut(ndarray::Axis(0))
        .zip(A.axis_iter(ndarray::Axis(0)))
        .zip(B.axis_iter(ndarray::Axis(0)))
    {
        for ((mut res_slice, a_slice), b_slice) in res
            .axis_iter_mut(ndarray::Axis(0))
            .zip(a_batch.axis_iter(ndarray::Axis(0)))
            .zip(b_batch.axis_iter(ndarray::Axis(0)))
        {
            let b_transposed = b_slice.clone().reversed_axes();
            let product = a_slice.dot(&b_transposed);
            res_slice.assign(&product);
        }
    }
    let result = result.into_raw_vec_and_offset().0;
    Ok((result, result_shape))
}

pub fn nearest_neighbour_interpolation (args:(Vec<f32>, Vec<usize>)) ->  Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let (input_vec, input_vec_shape) = args;
    let input_tensor = ndarray::Array4::from_shape_vec((input_vec_shape[0], input_vec_shape[1], input_vec_shape[2], input_vec_shape[3]), input_vec)?;
    let mut output_tensor = ndarray::Array4::<f32>::zeros((input_vec_shape[0], input_vec_shape[1], input_vec_shape[2] * 2, input_vec_shape[3] * 2));
    for batches in 0..input_vec_shape[0] {
        for channels in 0..input_vec_shape[1] {
            for height in 0..input_vec_shape[2] * 2 {
                for width in 0..input_vec_shape[3] * 2 {
                    let round_height = height / 2;
                    let round_width = width / 2;
                    output_tensor[[batches, channels, height, width]] = input_tensor[[batches, channels, round_height, round_width]];
                }
            }
        }
    }
    let res_vec = output_tensor.into_raw_vec_and_offset().0;
    let mut res_vec_shape = input_vec_shape;
    res_vec_shape[2] *= 2;
    res_vec_shape[3] *= 2;
    Ok((res_vec, res_vec_shape))
}

struct Transformer2D_params {
    number_of_groups: usize, eps: f32, gamma: f32, beta: f32,
    weigths_in: Vec<f32>, weights_shape_in: Vec<usize>, bias_in: Vec<f32>, bias_shape_in : Vec<usize>, is_bias_in : bool,
    weigths_out: Vec<f32>, weights_shape_out: Vec<usize>, bias_out: Vec<f32>, bias_shape_out : Vec<usize>, is_bias_out : bool,
    params_for_basics_vec : Vec<BasicTransofmerBlock_params>,
}

struct BasicTransofmerBlock_params {
    eps_1 : f32, gamma_1 : f32, beta_1 : f32, number_1 : usize, // LayerNorm 
    eps_2 : f32, gamma_2 : f32, beta_2 : f32, number_2 : usize, // LayerNorm 
    eps_3 : f32, gamma_3 : f32, beta_3 : f32, number_3 : usize, // LayerNorm 
    weigths_1: Vec<f32>, weights_shape_1 : Vec<usize>, bias_1: Vec<f32>, bias_shape_1 : Vec<usize>, is_bias_1 : bool,  // Attn1
    weigths_2: Vec<f32>, weights_shape_2 : Vec<usize>, bias_2: Vec<f32>, bias_shape_2 : Vec<usize>, is_bias_2 : bool,
    weigths_3: Vec<f32>, weights_shape_3 : Vec<usize>, bias_3: Vec<f32>, bias_shape_3 : Vec<usize>, is_bias_3 : bool,
    weigths_4: Vec<f32>, weights_shape_4 : Vec<usize>, bias_4: Vec<f32>, bias_shape_4 : Vec<usize>, is_bias_4 : bool,

    weigths_5: Vec<f32>, weights_shape_5 : Vec<usize>, bias_5: Vec<f32>, bias_shape_5 : Vec<usize>, is_bias_5 : bool, // Attn2
    weigths_6: Vec<f32>, weights_shape_6 : Vec<usize>, bias_6: Vec<f32>, bias_shape_6 : Vec<usize>, is_bias_6 : bool,
    weigths_7: Vec<f32>, weights_shape_7 : Vec<usize>, bias_7: Vec<f32>, bias_shape_7 : Vec<usize>, is_bias_7 : bool,
    weigths_8: Vec<f32>, weights_shape_8 : Vec<usize>, bias_8: Vec<f32>, bias_shape_8 : Vec<usize>, is_bias_8 : bool,
    
    weigths_ff1: Vec<f32>,  weights_shape_ff1 : Vec<usize>, bias_ff1: Vec<f32>, bias_shape_ff1 : Vec<usize>, is_bias_ff1 : bool, // FeedForward
    weigths_ff2: Vec<f32>, weights_shape_ff2 : Vec<usize>, bias_ff2: Vec<f32>, bias_shape_ff2 : Vec<usize>, is_bias_ff2 : bool,
}

struct Resnet2d_params {
    number_of_groups_1 : usize, eps_1: f32, gamma_1: f32, beta_1: f32,
    in_channels_1 : usize, out_channels_1 : usize, padding_1 : i32, stride_1 : i32, kernel_size_1 : usize, kernel_weights_1 : Vec<f32>,
    weigths: Vec<f32>, weights_shape : Vec<usize>, bias: Vec<f32>, bias_shape : Vec<usize>, is_bias : bool,
    number_of_groups_2 : usize, eps_2: f32, gamma_2: f32, beta_2: f32,
    in_channels_2 : usize, out_channels_2 : usize, padding_2 : i32,  stride_2 : i32, kernel_size_2 : usize, kernel_weights_2 : Vec<f32>,
    is_shortcut : bool,
    in_channels_short : usize, out_channels_short : usize, padding_short : i32,  stride_short : i32, kernel_size_short : usize, kernel_weights_short: Vec<f32>,
    time_emb : Vec<f32>, time_emb_shape : Vec<usize>,
}

struct CrossAttnUpBlock2D_params {
    params_for_transformer1 : Transformer2D_params,
    params_for_transformer2 : Transformer2D_params,
    params_for_transformer3 : Transformer2D_params,
    params_for_resnet1 : Resnet2d_params,
    params_for_resnet2 : Resnet2d_params,
    params_for_resnet3 : Resnet2d_params,
    in_channels: usize, out_channels: usize, padding: i32, stride : i32, kernel_size: usize, kernel_weights: Vec<f32>
}

struct CrossAttnDownBlock2D_params {
    is_downsample2d: bool,
    params_for_transformer1 : Transformer2D_params,
    params_for_transformer2 : Transformer2D_params,
    params_for_resnet1 : Resnet2d_params,
    params_for_resnet2: Resnet2d_params,
    in_channels : usize, out_channels : usize, padding : i32, stride : i32, kernel_size : usize, kernel_weights : Vec<f32>
}
pub trait Layer {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        Ok(args)
    }
}

struct SiLU;
impl Layer for SiLU {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let mut vec = args.0;
        for i in 0..vec.len() {
            vec[i] = vec[i] * (1.0 / (1.0 + E.powf(-vec[i])));
        }
        Ok((vec, args.1))
    }
}
struct GeLU;
impl Layer for GeLU {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let mut vec = args.0;
        for i in 0..vec.len() {
            vec[i] = vec[i] * (1. / 2.) * ((1. + erf((vec[i] as f64) / (2_f64).powf(1. / 2.))) as f32);
        }
        Ok((vec, args.1))
    }
}
struct Dropout {
    probability: f32,
}
impl Layer for Dropout {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let probability = self.probability;
        let mut vec = args.0.clone();
        let bern = rand_distr::Bernoulli::new(probability.into()).unwrap();
        let scale = 1.0 / (1.0 - probability);
        let mut rng = rand::thread_rng();
    
        for i in 0..vec.len() {
            if bern.sample(&mut rng) == true {
                vec[i] = 0.0;
            } else {
                vec[i] *= scale as f32;
            }
        }
        print!("{:?}", vec);
        Ok((vec, args.1))
    
    }
}

struct GroupNorm {
    number_of_groups: usize,
    eps: f32,
    gamma: f32,
    beta: f32,
}
impl Layer for GroupNorm {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let number_of_groups = self.number_of_groups;
        let eps = self.eps;
        let gamma = self.gamma;
        let beta = self.beta;
        let mut vec = args.0; 
        let ch_per_group = args.1[1] / number_of_groups;
        let for_index_opt =  ch_per_group * args.1[2] * args.1[3];
        for batch_idx in 0..args.1[0]{
            for ch in 0..number_of_groups {
                // let start_index = batch_idx * ch * for_index_opt;
                // let end_index = batch_idx * (ch + 1) * for_index_opt;
                let start_index = batch_idx * args.1[1] * args.1[2] * args.1[3] + ch * for_index_opt;
                let end_index = start_index + for_index_opt;
                let mut mean: f32 = 0.;
                let cnt: f32 = (end_index - start_index) as f32;
                for x in start_index..end_index {
                    mean += vec[x];
                }
                mean = mean / cnt;
                let mut var: f32 = 0.;
        
                for x in start_index..end_index {
                    var += (vec[x] - mean).powf(2.);
                }
                var = var / (cnt);
                let std = (var+eps).sqrt();
                for x in start_index..end_index {
                    vec[x] = ((vec[x] - mean) * gamma) / (std);
                    vec[x] += beta;
                }
            }
            // print!("{:?}\n", batch_idx);
        }
        Ok((vec, args.1))
    }
}

struct Conv2d {
    in_channels : usize,
    out_channels : usize,
    padding : i32,
    stride : i32,
    kernel_size : usize,
    kernel_weights : Vec<f32>,
}
impl Layer for Conv2d {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let dev = cudarc::driver::CudaDevice::new(0)?;
        let cudnn = cudarc::cudnn::Cudnn::new(dev.clone())?;
        
        let vec = args.0; // здесь вектор со значениями input_image
        let prep_shape = args.1.iter().map(|&x| x as i32).collect::<Vec<i32>>().into_boxed_slice();
        let prep_shape: [i32; 4] = prep_shape.to_vec().try_into().expect("Failed");

        let x = dev.htod_copy(vec.to_vec())?; // наше входное изображение, данные на gpu
        let x_disc = cudnn::Cudnn::create_4d_tensor::<f32>(&cudnn, cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, prep_shape)?;
        let mut y: CudaSlice<f32> = dev.alloc_zeros(vec.len())?;
        let y_disc = cudnn::Cudnn::create_4d_tensor::<f32>(&cudnn, cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, prep_shape)?;
        let vec = self.kernel_weights.clone();
        let kernel = dev.htod_copy(vec.to_vec())?;
        let mut actual_prep_shape = prep_shape.clone();
        actual_prep_shape[0] = self.in_channels as i32;
        actual_prep_shape[1] = self.out_channels as i32;
        actual_prep_shape[2] = self.kernel_size as i32;
        actual_prep_shape[3] = self.kernel_size as i32;
        let filter = cudnn::Cudnn::create_4d_filter::<f32>(&cudnn, cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, actual_prep_shape)?;
        let mut conv = cudnn::Cudnn::create_conv2d::<f32>(&cudnn, [self.padding; 2], [self.stride; 2],[1;2], cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION)?;
        conv.set_math_type(cudarc::cudnn::sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH).unwrap();
        
            let op = cudnn::ConvForward {
                conv: &conv,
                x: &x_disc,
                w: &filter,
                y: &y_disc,
            };
            
            let algo = op.pick_algorithm()?;

            let workspace_size = op.get_workspace_size(algo)?;
            let mut workspace = dev.alloc_zeros::<u8>(workspace_size).unwrap();

            unsafe {
                op.launch(algo, Some(&mut workspace), (1.0, 0.0), &x, &kernel, &mut y)?;
            }
            let y_host = dev.sync_reclaim(y).unwrap();
        
            Ok((y_host, args.1))
    }

}
// (input_vec, input_shape, conv_vec, conv_shape)
pub fn input(input_name: String) -> Result<(Arc<[f32]>, Arc<Vec<usize>>, Arc<[f32]>, Arc<Vec<usize>>), Box<dyn std::error::Error>> { 
    let mut file = File::open(input_name)?;
    let mut buffer:Vec<u8> = Vec::new();
    file.read_to_end(&mut buffer)?;
    let tensors = SafeTensors::deserialize(&buffer)?;

    let mut data_ptrs: Vec<&[u8]> = Vec::new();
    let mut shape_ptrs: Vec<Vec<usize>> = Vec::new();
    let mut k = 0;
    let mut input_index = 0;

    for (name,tensor) in tensors.tensors() {
        if name == "input_image" {
            input_index = k;
        }
        let data = tensor.data();
        let shape = tensor.shape().to_vec();
        data_ptrs.push(&data); // now we have vec of <f32> vec of our conv.weights tensor and input_image tensor
        shape_ptrs.push(shape);
        k = 1;
    }
    let kernel_index;
    if input_index == 0 {
        kernel_index = 1;
    } else {
        kernel_index = 0;
    }
    let input_vec: Arc<[f32]>= Arc::from(bytemuck::cast_slice(data_ptrs[input_index]));
    let input_shape: Arc<Vec<usize>> = Arc::from(shape_ptrs[input_index].clone());
    let kernel_vec: Arc<[f32]> = Arc::from(bytemuck::cast_slice(data_ptrs[kernel_index]));
    let kernel_shape: Arc<Vec<usize>> = Arc::from(shape_ptrs[kernel_index].clone());
    Ok((input_vec, input_shape, kernel_vec, kernel_shape))
}

pub fn output(output_name: String, tensor_vec:Vec<f32>, shape_vec:Vec<usize>) -> Result<(), Box<dyn std::error::Error>> {
    let binding = tensor_vec.to_vec();
    let prep_output:&[u8] = bytemuck::cast_slice(&binding);
    let mut tensors = std::collections::HashMap::new();
    tensors.insert("output_tensor".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape_vec.clone(), prep_output)?);
    let serialized_data = serialize(&tensors, &None)?;
    let mut file = File::create(output_name.to_string())?;
    file.write_all(&serialized_data)?;
    Ok(())
}

struct Linear{
    weigths: Vec<f32>,
    weights_shape : Vec<usize>,
    bias: Vec<f32>,
    bias_shape : Vec<usize>,
    is_bias : bool,
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

struct Resnet2d {
    if_shortcut:bool,
    operations: Vec<Box<dyn Layer>>,
    time_emb : Vec<f32>,
    time_emb_shape : Vec<usize>,
}

impl Resnet2d {
    pub fn Resnet2d_constr (
        params : Resnet2d_params
        ) -> Self {
            let mut layer_vec : Vec<Box<dyn Layer>> = Vec::new();
            layer_vec.push(Box::new(GroupNorm {number_of_groups: params.number_of_groups_1, eps : params.eps_1, gamma : params.gamma_1, beta : params.beta_1}));
            layer_vec.push(Box::new(SiLU));
            layer_vec.push(Box::new(Conv2d {in_channels : params.in_channels_1, out_channels : params.out_channels_1, padding : params.padding_1, stride: params.stride_2, kernel_size : params.kernel_size_1, kernel_weights : params.kernel_weights_1}));
            layer_vec.push(Box::new(Linear{weigths : params.weigths, weights_shape : params.weights_shape, bias : params.bias, bias_shape : params.bias_shape, is_bias : params.is_bias}));
            layer_vec.push(Box::new(GroupNorm {number_of_groups: params.number_of_groups_2, eps : params.eps_2, gamma : params.gamma_2, beta : params.beta_2}));
            layer_vec.push(Box::new(SiLU));
            layer_vec.push(Box::new(Conv2d {in_channels : params.in_channels_2, out_channels : params.out_channels_2, padding : params.padding_2, stride: params.stride_2, kernel_size : params.kernel_size_2, kernel_weights : params.kernel_weights_2}));
            if params.is_shortcut {
                layer_vec.push(Box::new(Conv2d {in_channels : params.in_channels_short, out_channels : params.out_channels_short, stride: params.stride_short, padding : params.padding_short, kernel_size : params.kernel_size_short, kernel_weights : params.kernel_weights_short}));
            }
            Self { if_shortcut: params.is_shortcut, operations: layer_vec, time_emb : params.time_emb, time_emb_shape : params.time_emb_shape}
    }   
}
impl Layer for Resnet2d {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let mut res_vec = args.0.clone();
        let mut res_shape_vec = args.1.clone();
        for i in 0..self.operations.len()-(self.if_shortcut as usize) {
            if i == 3 {
                let lin_res = self.operations[i].operation((self.time_emb.clone(), self.time_emb_shape.clone()))?;
                let mut curr_tensor = ndarray::Array4::from_shape_vec((res_shape_vec[0],res_shape_vec[1],res_shape_vec[2],res_shape_vec[3]), res_vec.clone())?;
                let time_tensor = ndarray::Array2::from_shape_vec((lin_res.1[0], lin_res.1[1]), lin_res.0)?;
                curr_tensor = curr_tensor.clone() + time_tensor.broadcast(curr_tensor.dim()).unwrap();
                res_vec = curr_tensor.into_raw_vec_and_offset().0;
                continue;
            }
            let res = self.operations[i].operation((res_vec, res_shape_vec))?;
            res_vec = res.0.clone();
            res_shape_vec = res.1.clone();
        }
        if self.if_shortcut {
            let shortcut_res = self.operations[self.operations.len() - 1].operation(args.clone())?;
            let shortcut_vec = shortcut_res.0;
            let mut curr_tensor = ndarray::Array4::from_shape_vec((res_shape_vec[0],res_shape_vec[1],res_shape_vec[2],res_shape_vec[3]), res_vec.clone())?;
            let mut short_tensor = ndarray::Array4::from_shape_vec((shortcut_res.1[0], shortcut_res.1[1], shortcut_res.1[2], shortcut_res.1[3]), shortcut_vec.clone())?;
            curr_tensor = curr_tensor + short_tensor;
            res_vec = curr_tensor.into_raw_vec_and_offset().0;
        }
        Ok((res_vec, res_shape_vec))
    }
}

struct LayerNorm {
    eps : f32,
    gamma : f32,
    beta : f32,
    number : usize,
}
impl Layer for LayerNorm {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let mut vec = args.0.clone();
        let limit = self.number;
        for i in (0..args.0.len()).step_by(limit) {
            let mut mean: f32 = 0.;
            let mut var: f32 = 0.;
            for j in 0..limit {
                mean = mean + vec[i + j];
            }
            mean = mean / (limit as f32);
            for j in 0..limit {
                var = var + (vec[i + j] - mean).powf(2.);
            }
            var = var / (limit as f32);
            let std = (var + self.eps).sqrt();
            for j in 0..limit {
                vec[i + j] = ((vec[i + j] - mean) * self.gamma) / (std);
                vec[i + j] = vec[i + j] + self.beta;
            }
        }
        print!("{:?}", vec);
        Ok((vec, args.1))
    }
}
struct FeedForward {
    operations: Vec<Box<dyn Layer>>,
}
impl FeedForward {
    pub fn FeedForward_constr (
        weigths_1: Vec<f32>,
        weights_shape_1 : Vec<usize>,
        bias_1: Vec<f32>,
        bias_shape_1 : Vec<usize>,
        is_bias_1 : bool,
        weigths_2: Vec<f32>,
        weights_shape_2 : Vec<usize>,
        bias_2: Vec<f32>,
        bias_shape_2 : Vec<usize>,
        is_bias_2 : bool,
    ) -> Self {
        let mut vec: Vec<Box<dyn Layer>> = Vec::new();
        vec.push(Box::new(Linear {weigths : weigths_1, weights_shape : weights_shape_1, bias : bias_1, bias_shape : bias_shape_1, is_bias : is_bias_1}));
        vec.push(Box::new(GeLU));
        vec.push(Box::new(Linear { weigths : weigths_2, weights_shape : weights_shape_2, bias : bias_2, bias_shape : bias_shape_2, is_bias : is_bias_2}));
        Self { operations: vec }
    }
}
impl Layer for FeedForward {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let res = &self.operations[0].operation(args)?;
        let mut res_vec = res.0.clone();
        let mut res_vec_shape = res.1.clone();
        let limit = (res_vec_shape[0] *res_vec_shape[1] *res_vec_shape[2] *res_vec_shape[3]) / 2;
        let (part_vec_1, part_vec_2) = res_vec.split_at(limit);
        let part_vec_1 = part_vec_1.to_vec();
        let part_vec_2 = part_vec_2.to_vec();
        let mut part_vec_shape = res_vec_shape;
        part_vec_shape[3] = part_vec_shape[3] / 2;
        let act_part_vec_1 = &self.operations[1].operation((part_vec_1, part_vec_shape.clone()))?;
        let part_vec_1 = act_part_vec_1.0.clone();
        let act_vec = ndarray::Array1::from_shape_vec(limit, part_vec_1)?;
        let another_vec = ndarray::Array1::from_shape_vec(limit, part_vec_2)?;
        res_vec = (act_vec * another_vec).to_vec();
        res_vec_shape = part_vec_shape;
        let res = &self.operations[2].operation((res_vec, res_vec_shape))?;
        Ok((res.0.clone(), res.1.clone()))
    }
}
struct Attention {
    operations : Vec<Box<dyn Layer>>,
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
struct BasicTransofmerBlock {
    operations: Vec<Box<dyn Layer>>,
}
impl BasicTransofmerBlock {
    pub fn BasicTransofmerBlock_constr(
        params : &BasicTransofmerBlock_params
    ) -> Self {
        let mut vec: Vec<Box<dyn Layer>> = Vec::new();
        vec.push(Box::new(LayerNorm { eps : params.eps_1.clone(), gamma : params.gamma_1.clone(), beta : params.beta_1.clone(), number : params.number_1.clone()}));
        let attn1 = Attention::Attention_constr(params.weigths_1.clone(), params.weights_shape_1.clone(), params.bias_1.clone(), params.bias_shape_1.clone(), params.is_bias_1.clone(), params.weigths_2.clone(), params.weights_shape_2.clone(), params.bias_2.clone(), params.bias_shape_2.clone(), params.is_bias_2.clone(), params.weigths_3.clone(), params.weights_shape_3.clone(), params.bias_3.clone(), params.bias_shape_3.clone(), params.is_bias_3.clone(), params.weigths_4.clone(), params.weights_shape_4.clone(), params.bias_4.clone(), params.bias_shape_4.clone(), params.is_bias_4.clone());
        vec.push(Box::new(attn1));
        vec.push(Box::new(LayerNorm { eps :params.eps_2.clone(), gamma : params.gamma_2.clone(), beta : params.beta_2.clone(), number : params.number_2.clone()}));
        let attn2 = Attention::Attention_constr(params.weigths_5.clone(), params.weights_shape_5.clone(), params.bias_5.clone(), params.bias_shape_5.clone(), params.is_bias_5.clone(), params.weigths_6.clone(), params.weights_shape_6.clone(), params.bias_6.clone(), params.bias_shape_6.clone(), params.is_bias_6.clone(), params.weigths_7.clone(), params.weights_shape_7.clone(), params.bias_7.clone(), params.bias_shape_7.clone(), params.is_bias_7.clone(), params.weigths_8.clone(), params.weights_shape_8.clone(), params.bias_8.clone(), params.bias_shape_8.clone(), params.is_bias_8.clone());
        vec.push(Box::new(attn2));
        vec.push(Box::new(LayerNorm { eps : params.eps_3.clone(), gamma : params.gamma_3.clone(), beta : params.beta_3.clone(), number : params.number_3.clone()}));
        let ff = FeedForward::FeedForward_constr(params.weigths_ff1.clone(), params.weights_shape_ff1.clone(), params.bias_ff1.clone(), params.bias_shape_ff1.clone(), params.is_bias_ff1.clone(), params.weigths_ff2.clone(), params.weights_shape_ff2.clone(), params.bias_ff2.clone(), params.bias_shape_ff2.clone(), params.is_bias_ff2.clone());
        vec.push(Box::new(ff));
        Self { operations: vec }
    }
}
impl Layer for BasicTransofmerBlock {
    // fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    //     let operations = &self.operations;
        // let mut res_vec = args.0;
        // let mut res_vec_shape = args.1;
    //     for layer in operations {
    //         let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
    //         res_vec = temp_vec;
    //         res_vec_shape = temp_vec_shape;
    //     } 
    //     Ok((res_vec, res_vec_shape))
    // }
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut res_vec = args.0.clone();
        let mut res_vec_shape = args.1.clone();
        let (norm_vec, norm_vec_shape) = operations[0].operation(args)?;
        let (attn1_vec, attn1_vec_shape) = operations[1].operation((norm_vec, norm_vec_shape))?;
        let mut res_matr = ndarray::Array1::from_shape_vec((res_vec_shape[0] * res_vec_shape[1] * res_vec_shape[2] * res_vec_shape[3]), res_vec)?;
        let attn_matr = ndarray::Array1::from_shape_vec((attn1_vec_shape[0] * attn1_vec_shape[1] * attn1_vec_shape[2] * attn1_vec_shape[3]), attn1_vec)?;
        res_matr = res_matr.clone() + attn_matr;
        let temp_matr = res_matr.clone();
        res_vec = temp_matr.into_raw_vec_and_offset().0;
        let (norm_vec, norm_vec_shape) = operations[2].operation((res_vec, res_vec_shape.clone()))?;
        let (attn2_vec, attn2_vec_shape) = operations[3].operation((norm_vec, norm_vec_shape))?;
        let attn_matr = ndarray::Array1::from_shape_vec((attn2_vec_shape[0] * attn2_vec_shape[1] * attn2_vec_shape[2] * attn2_vec_shape[3]), attn2_vec)?;
        res_matr = res_matr.clone() + attn_matr;
        let temp_matr = res_matr.clone();
        res_vec = temp_matr.into_raw_vec_and_offset().0;
        let (norm_vec, norm_vec_shape) = operations[4].operation((res_vec, res_vec_shape.clone()))?;
        let (ff_vec, ff_vec_shape) = operations[5].operation((norm_vec, norm_vec_shape))?;
        let ff_matr = ndarray::Array1::from_shape_vec(ff_vec_shape[0] * ff_vec_shape[1] * ff_vec_shape[2] * ff_vec_shape[3], ff_vec)?;
        res_matr = res_matr.clone() + ff_matr;
        let temp_matr = res_matr.clone();
        res_vec = temp_matr.into_raw_vec_and_offset().0;
        Ok((res_vec, res_vec_shape))
    }
}
struct Transformer2D {
    operations : Vec<Box<dyn Layer>>,
    number_of_basic : usize,
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

struct mid_block {
    operations : Vec<Box<dyn Layer>>,
}

impl mid_block {
    pub fn mid_block_constr (
        params_for_transformer2d :Transformer2D_params,
        params_for_resnet_1: Resnet2d_params,
        params_for_resnet_2: Resnet2d_params
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::Resnet2d_constr(params_for_resnet_1);
        vec.push(Box::new(resnet1));
        let transformer = Transformer2D::Transformer2D_constr(
            params_for_transformer2d);
        vec.push(Box::new(transformer));
        let resnet2 = Resnet2d::Resnet2d_constr(params_for_resnet_2);
        vec.push(Box::new(resnet2));
        Self { operations: vec }
    }
}
impl Layer for mid_block {
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
struct Upsample2D {
    operations: Vec<Box<dyn Layer>>,
}
impl Upsample2D {
    pub fn Upsample2D_constr(
        in_channels : usize,
        out_channels : usize,
        padding : i32,
        stride: i32,
        kernel_size : usize,
        kernel_weights : Vec<f32>
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(Conv2d{in_channels : in_channels, out_channels : out_channels, padding : padding, stride:stride,  kernel_size : kernel_size, kernel_weights : kernel_weights}));
        Self { operations: vec }
    }
}
impl Layer for Upsample2D {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let (input_tensor, input_tensor_shape) = args;
        let (near_vec, near_vec_shape) = nearest_neighbour_interpolation((input_tensor, input_tensor_shape))?;
        let (res_vec, res_vec_shape) = &self.operations[0].operation((near_vec, near_vec_shape))?;
        Ok((res_vec.to_vec(), res_vec_shape.to_vec()))
    }
}
struct CrossAttnUpBlock2D {
    operations : Vec<Box<dyn Layer>>,
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

struct UpBlock2d {
    operations : Vec<Box<dyn Layer>>,
}

impl UpBlock2d {
    pub fn UpBlock2d_constr(
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        params_for_resnet3 : Resnet2d_params,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::Resnet2d_constr(params_for_resnet1);
        vec.push(Box::new(resnet1));
        let resnet2 = Resnet2d::Resnet2d_constr(params_for_resnet2);
        vec.push(Box::new(resnet2));
        let resnet3 = Resnet2d::Resnet2d_constr(params_for_resnet3);
        vec.push(Box::new(resnet3));
        Self { operations: vec }
    }
}

impl Layer for UpBlock2d {
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

struct Up_blocks {
    operations : Vec<Box<dyn Layer>>,
}

impl Up_blocks {
    pub fn Up_block_constr (
        params_for_crossblock1 : CrossAttnUpBlock2D_params,
        params_for_crossblock2 : CrossAttnUpBlock2D_params,
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        params_for_resnet3 : Resnet2d_params,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let crossAttnUpBlock1 = CrossAttnUpBlock2D::CrossAttnUpBlock2D_constr(params_for_crossblock1);
        vec.push(Box::new(crossAttnUpBlock1));
        let crossAttnUpBlock2 = CrossAttnUpBlock2D::CrossAttnUpBlock2D_constr(params_for_crossblock2);
        vec.push(Box::new(crossAttnUpBlock2));
        let upblock2d = UpBlock2d::UpBlock2d_constr(params_for_resnet1, params_for_resnet2, params_for_resnet3);
        vec.push(Box::new(upblock2d));
        Self { operations: vec }
    }
}

impl Layer for Up_blocks {
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

struct DownSample2D {
    operations : Vec<Box<dyn Layer>>,
}

impl DownSample2D {
    pub fn DownSample2D_constr(
        in_channels : usize,
        out_channels : usize,
        padding : i32,
        stride : i32,
        kernel_size : usize,
        kernel_weights : Vec<f32>,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(Conv2d{in_channels : in_channels, out_channels : out_channels, padding : padding, stride : stride, kernel_size : kernel_size, kernel_weights : kernel_weights}));
        Self { operations: vec }
    }
}

impl Layer for DownSample2D {
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

struct CrossAttnDownBlock2D {
    if_downsample2d : bool,
    operations : Vec<Box<dyn Layer>>,
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

struct DownBlock2D {
    operations : Vec<Box<dyn Layer>>,
}

impl DownBlock2D {
    pub fn DownBlock2D_constr(
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        in_channels : usize, out_channels : usize, padding : i32, stride : i32, kernel_size : usize, kernel_weights : Vec<f32>,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::Resnet2d_constr(params_for_resnet1);
        vec.push(Box::new(resnet1));
        let resnet2 = Resnet2d::Resnet2d_constr(params_for_resnet2);
        vec.push(Box::new(resnet2));
        let downsample = DownSample2D::DownSample2D_constr(in_channels, out_channels, padding, stride, kernel_size, kernel_weights);
        vec.push(Box::new(downsample));
        Self { operations: vec }
    }
}

impl Layer for DownBlock2D {
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

struct Down_blocks {
    operations : Vec<Box<dyn Layer>>,
}

impl Down_blocks {
    pub fn Down_blocks_constr (
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        in_channels : usize, out_channels : usize, padding : i32, stride : i32, kernel_size : usize, kernel_weights : Vec<f32>,
        params_for_crattbl1 : CrossAttnDownBlock2D_params,
        params_for_crattbl2 : CrossAttnDownBlock2D_params,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let downblock = DownBlock2D::DownBlock2D_constr(params_for_resnet1, params_for_resnet2, in_channels, out_channels, padding, stride, kernel_size, kernel_weights);
        vec.push(Box::new(downblock));
        let crossattnblock1 = CrossAttnDownBlock2D::CrossAttnDownBlock2D_constr(params_for_crattbl1);
        vec.push(Box::new(crossattnblock1));
        let crossattnblock2 = CrossAttnDownBlock2D::CrossAttnDownBlock2D_constr(params_for_crattbl2);
        vec.push(Box::new(crossattnblock2));
        Self { operations: vec }
    }
}

impl Layer for Down_blocks {
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
struct Unet2dConditionModel {
    operations : Vec<Box<dyn Layer>>,
    time_emb : Vec<f32>,
    time_emb_shape : Vec<usize>,
}

impl Unet2dConditionModel {
    pub fn Unet2dConditionModel(
        time_emb : Vec<f32>, time_emb_shape : Vec<usize>, // time
        weigths1: Vec<f32>, weights_shape1 : Vec<usize>, bias1: Vec<f32>, bias_shape1 : Vec<usize>, is_bias1 : bool,
        weigths2 : Vec<f32>, weights_shape2 : Vec<usize>, bias2: Vec<f32>, bias_shape2 : Vec<usize>, is_bias2 : bool,

        in_channels_in : usize, out_channels_in: usize, padding_in : i32, stride_in : i32, kernel_size_in : usize, kernel_weights_in : Vec<f32>, //in

        params_for_resnet1_down : Resnet2d_params, // down
        params_for_resnet2_down : Resnet2d_params,
        in_channels_down : usize, out_channels_down : usize, padding_down : i32, stride_down : i32, kernel_size_down : usize, kernel_weights_down : Vec<f32>,
        params_for_crattbl1 : CrossAttnDownBlock2D_params,
        params_for_crattbl2 : CrossAttnDownBlock2D_params,

        params_for_crossblock1 : CrossAttnUpBlock2D_params, // up
        params_for_crossblock2 : CrossAttnUpBlock2D_params,
        params_for_resnet1_up : Resnet2d_params,
        params_for_resnet2_up : Resnet2d_params,
        params_for_resnet3_up : Resnet2d_params,

        params_for_transformer2d :Transformer2D_params, // mid
        params_for_resnet_1_mid: Resnet2d_params,
        params_for_resnet_2_mid: Resnet2d_params,

        number_of_groups_out: usize, eps_out: f32, gamma_out: f32, beta_out: f32, //out
        in_channels_out : usize, out_channels_out: usize, padding_out : i32, stride_out : i32, kernel_size_out : usize, kernel_weights_out : Vec<f32>,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(Linear {weigths : weigths1, weights_shape : weights_shape1, bias : bias1, bias_shape : bias_shape1, is_bias : is_bias1}));
        vec.push(Box::new(Linear {weigths : weigths2, weights_shape : weights_shape2, bias : bias2, bias_shape : bias_shape2, is_bias : is_bias2}));
        vec.push(Box::new(Conv2d{in_channels : in_channels_in, out_channels : out_channels_in, padding : padding_in, stride : stride_in, kernel_size : kernel_size_in, kernel_weights : kernel_weights_in}));
        let down = Down_blocks::Down_blocks_constr(params_for_resnet1_down, params_for_resnet2_down, in_channels_down, out_channels_down, padding_down, stride_down, kernel_size_down, kernel_weights_down, params_for_crattbl1, params_for_crattbl2);
        vec.push(Box::new(down));
        let mid = mid_block::mid_block_constr(params_for_transformer2d, params_for_resnet_1_mid, params_for_resnet_2_mid);
        vec.push(Box::new(mid));
        let up = Up_blocks::Up_block_constr(params_for_crossblock1, params_for_crossblock2, params_for_resnet1_up, params_for_resnet2_up, params_for_resnet3_up);
        vec.push(Box::new(up));
        vec.push(Box::new(GroupNorm{number_of_groups : number_of_groups_out, eps : eps_out, gamma : gamma_out, beta: beta_out}));
        vec.push(Box::new(SiLU));
        vec.push(Box::new(Conv2d{in_channels: in_channels_out, out_channels : out_channels_out, padding : padding_out, stride: stride_out, kernel_size : kernel_size_out, kernel_weights : kernel_weights_out}));
        Self { operations: vec, time_emb: time_emb, time_emb_shape: time_emb_shape }
    }
}

impl Layer for Unet2dConditionModel {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let (temp_emb_vec, temp_emb_shape) = operations[0].operation((self.time_emb.clone(), self.time_emb_shape.clone()))?;
        let (time_emb_vec, time_emb_shape) = operations[1].operation((temp_emb_vec.clone(), temp_emb_shape.clone()))?;
        let mut res_vec = args.0;
        let mut res_vec_shape = args.1;
        for i in 2..operations.len() {
            let (temp_vec, temp_vec_shape) = operations[i].operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
        } 
        Ok((res_vec, res_vec_shape))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
}
