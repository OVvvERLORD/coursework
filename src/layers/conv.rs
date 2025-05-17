use crate::layers::layer::Layer;
use cudarc::cudnn::Cudnn;
use cudarc::driver::CudaSlice;
use cudarc::{self, cudnn};
use crate::func::functions::input;

pub struct Conv2d{
    pub in_channels : usize,
    pub out_channels : usize,
    pub padding : i32,
    pub stride : i32,
    pub kernel_size : usize,
    pub kernel_weights : Vec<f32>,
    pub bias: ndarray::Array4<f32>,
    pub is_bias: bool
}
impl Layer for Conv2d {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let dev = cudarc::driver::CudaDevice::new(0)?;
        let cudnn = cudarc::cudnn::Cudnn::new(dev.clone())?;
        
        let vec = args.clone().into_raw_vec_and_offset().0; 
        let prep_shape = args.shape().iter().map(|&x| x as i32).collect::<Vec<i32>>().into_boxed_slice();
        let prep_shape: [i32; 4] = prep_shape.to_vec().try_into().expect("Failed");
        let x = dev.htod_copy(vec.to_vec())?; 
        let x_disc = cudnn::Cudnn::create_4d_tensor::<f32>(&cudnn, cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, prep_shape)?;
        let mut y_shape = prep_shape;
        y_shape[1] = self.out_channels as i32;
        y_shape[2] = ((prep_shape[2] + 2 * self.padding - (self.kernel_size as i32)) / self.stride)  + 1;
        y_shape[3] = ((prep_shape[3] + 2 * self.padding - (self.kernel_size as i32)) / self.stride) + 1;
        let mut y: CudaSlice<f32> = dev.alloc_zeros(y_shape[0] as usize * y_shape[1] as usize * y_shape[2] as usize * y_shape[3] as usize)?;
        let y_disc = cudnn::Cudnn::create_4d_tensor::<f32>(&cudnn, cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, y_shape)?;
        let vec = self.kernel_weights.clone();
        let kernel = dev.htod_copy(vec.to_vec())?;
        let actual_prep_shape = [
            self.out_channels as i32, 
            self.in_channels as i32,
            self.kernel_size as i32,
            self.kernel_size as i32,
        ];
        let filter = cudnn::Cudnn::create_4d_filter::<f32>(&cudnn, cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, actual_prep_shape)?;
        let mut conv = cudnn::Cudnn::create_conv2d::<f32>(&cudnn, [self.padding, self.padding], [self.stride, self.stride],[1;2], cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION)?;
        // conv.set_math_type(cudarc::cudnn::sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION).unwrap();
        // conv.set_math_type(cudarc::cudnn::sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH).unwrap();
        conv.set_math_type(cudarc::cudnn::sys::cudnnMathType_t::CUDNN_DEFAULT_MATH).unwrap();
            let op = cudnn::ConvForward {
                conv: &conv,
                x: &x_disc,
                w: &filter,
                y: &y_disc,
            };

            let algo = if self.stride == 1 && self.kernel_size != 1
            {cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD}
            // {cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM}
            else {cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM};
            let workspace_size = op.get_workspace_size(algo)?;
            let mut workspace = dev.alloc_zeros::<u8>(workspace_size).unwrap();

            unsafe {
                op.launch(algo, Some(&mut workspace), (1.0, 0.0), &x, &kernel, &mut y)?;
            }
            let y_host = dev.sync_reclaim(y).unwrap();
            *args = ndarray::Array4::
            from_shape_vec( (y_shape[0] as usize, y_shape[1] as usize, y_shape[2] as usize, y_shape[3] as usize),
             y_host).unwrap().to_owned();
            // Ok((y_host, y_shape.iter().map(|&x| x as usize).collect()))
            if self.is_bias {
                let shape = args.shape();
                let target = self.bias.shape()[3];

                let bias = &self.bias.clone().into_shape_with_order((1, target, 1, 1)).unwrap();
                let bias = bias.broadcast((shape[0], target, shape[2], shape[3])).unwrap();
                *args += &bias;
            }
            Ok(())
    }
}

// #[test]
// fn test_conv_std(){
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_std.safetensors".to_string()).unwrap();
//     let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_std.safetensors".to_string()).unwrap();
//     let conv = Conv2d {kernel_size: 3, in_channels: 1280, out_channels: 1280, padding: 1, stride: 1, kernel_weights: weight_vec.to_vec()};
//     let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_std_python.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..res_vec.len(){
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
//     }
// }

// #[test]
// fn test_conv_out() {
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_out.safetensors".to_string()).unwrap();
//     let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_out.safetensors".to_string()).unwrap();
//     let conv = Conv2d {kernel_size: 3, in_channels: 640, out_channels: 320, padding: 1, stride: 1, kernel_weights: weight_vec.to_vec()};
//     let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_out_python.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..res_vec.len(){
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
//     }
// }

// #[test]
// fn test_conv_in() {
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_in.safetensors".to_string()).unwrap();
//     let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_in.safetensors".to_string()).unwrap();
//     let conv = Conv2d {kernel_size: 3, in_channels: 4, out_channels: 320, padding: 1, stride: 1, kernel_weights: weight_vec.to_vec()};
//     let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_in_python.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..res_vec.len(){
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
//     }
// }

// #[test]
// fn test_conv_stride(){
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_stride.safetensors".to_string()).unwrap();
//     let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_stride.safetensors".to_string()).unwrap();
//     let conv = Conv2d {kernel_size: 3, in_channels: 320, out_channels: 320, padding: 1, stride: 2, kernel_weights: weight_vec.to_vec()};
//     let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_stride_python.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..res_vec.len(){
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
//     }
// }

#[test]
fn test_conv_kernel_in(){
    let mut tensor = input(r"C:\study\coursework\src\trash\test_conv_inp_kernel_in.safetensors".to_string()).unwrap();
    let kernel = input(r"C:\study\coursework\src\trash\test_conv_weight_kernel_in.safetensors".to_string()).unwrap();
    let bias = input(r"C:\study\coursework\src\trash\test_conv_bias_kernel_in.safetensors".to_string()).unwrap();
    let conv = Conv2d{in_channels: 640, out_channels: 1280, padding: 0, stride: 1, kernel_size: 1, kernel_weights: kernel.into_raw_vec_and_offset().0, bias: bias, is_bias: true};
    let _ = conv.operation(&mut tensor);
    let py_tensor = input(r"C:\study\coursework\src\trash\test_conv_kernel_in_python.safetensors".to_string()).unwrap();
    let shape = tensor.shape();
    print!("{:?}", shape);
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-05);
                }
            }
        }
    }
}

// #[test]
// fn test_conv_kernel_out(){
//     let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_kernel_out.safetensors".to_string()).unwrap();
//     let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_kernel_out.safetensors".to_string()).unwrap();
//     let conv = Conv2d {kernel_size: 1, in_channels: 960, out_channels: 640, padding: 0, stride: 1, kernel_weights: weight_vec.to_vec()};
//     let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
//     let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_kernel_out_python.safetensors".to_string()).unwrap();
//     assert!(py_vec_shape.to_vec() == res_vec_shape);
//     for i in 0..res_vec.len(){
//         assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
//     }
// }