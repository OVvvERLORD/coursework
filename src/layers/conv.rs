use crate::layers::layer::Layer;
use cudarc::driver::CudaSlice;
use cudarc::{self, cudnn};

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