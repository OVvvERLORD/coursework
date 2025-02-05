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
}
impl Layer for Conv2d {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let dev = cudarc::driver::CudaDevice::new(0)?;
        let cudnn = cudarc::cudnn::Cudnn::new(dev.clone())?;
        
        let vec = args.0; 
        let prep_shape = args.1.iter().map(|&x| x as i32).collect::<Vec<i32>>().into_boxed_slice();
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
        conv.set_math_type(cudarc::cudnn::sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH).unwrap();
        
            let op = cudnn::ConvForward {
                conv: &conv,
                x: &x_disc,
                w: &filter,
                y: &y_disc,
            };

            let algo = if (self.stride == 1 && self.kernel_size != 1) {cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD} else {cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM};
            let workspace_size = op.get_workspace_size(algo)?;
            let mut workspace = dev.alloc_zeros::<u8>(workspace_size).unwrap();

            unsafe {
                op.launch(algo, Some(&mut workspace), (1.0, 0.0), &x, &kernel, &mut y)?;
            }
            let y_host = dev.sync_reclaim(y).unwrap();
            
            Ok((y_host, y_shape.iter().map(|&x| x as usize).collect()))
    }
}

#[test]
fn test_conv_std(){
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_std.safetensors".to_string()).unwrap();
    let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_std.safetensors".to_string()).unwrap();
    let conv = Conv2d {kernel_size: 3, in_channels: 1280, out_channels: 1280, padding: 1, stride: 1, kernel_weights: weight_vec.to_vec()};
    let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_std_python.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len(){
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}

#[test]
fn test_conv_out() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_out.safetensors".to_string()).unwrap();
    let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_out.safetensors".to_string()).unwrap();
    let conv = Conv2d {kernel_size: 3, in_channels: 640, out_channels: 320, padding: 1, stride: 1, kernel_weights: weight_vec.to_vec()};
    let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_out_python.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len(){
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}

#[test]
fn test_conv_in() {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_in.safetensors".to_string()).unwrap();
    let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_in.safetensors".to_string()).unwrap();
    let conv = Conv2d {kernel_size: 3, in_channels: 4, out_channels: 320, padding: 1, stride: 1, kernel_weights: weight_vec.to_vec()};
    let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_in_python.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len(){
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}

#[test]
fn test_conv_stride(){
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_stride.safetensors".to_string()).unwrap();
    let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_stride.safetensors".to_string()).unwrap();
    let conv = Conv2d {kernel_size: 3, in_channels: 320, out_channels: 320, padding: 1, stride: 2, kernel_weights: weight_vec.to_vec()};
    let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_stride_python.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len(){
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}

#[test]
fn test_conv_kernel_in(){
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_kernel_in.safetensors".to_string()).unwrap();
    let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_kernel_in.safetensors".to_string()).unwrap();
    let conv = Conv2d {kernel_size: 1, in_channels: 640, out_channels: 1280, padding: 0, stride: 1, kernel_weights: weight_vec.to_vec()};
    let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_kernel_in_python.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len(){
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}

#[test]
fn test_conv_kernel_out(){
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp_kernel_out.safetensors".to_string()).unwrap();
    let (weight_vec, _) = input(r"C:\study\coursework\src\trash\test_conv_weight_kernel_out.safetensors".to_string()).unwrap();
    let conv = Conv2d {kernel_size: 1, in_channels: 960, out_channels: 640, padding: 0, stride: 1, kernel_weights: weight_vec.to_vec()};
    let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_kernel_out_python.safetensors".to_string()).unwrap();
    assert!(py_vec_shape.to_vec() == res_vec_shape);
    for i in 0..res_vec.len(){
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-05);
    }
}