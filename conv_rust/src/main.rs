use cudarc::cudnn::Cudnn;
use cudarc::driver::CudaSlice;
use safetensors::serialize;
use safetensors::{SafeTensors};
use cudarc::{self, cudnn};
use core::f32;
use std::fs::File;
use std::env;
use std::io::{Read, Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {

    let dev = cudarc::driver::CudaDevice::new(0).unwrap();
    let cudnn = cudnn::Cudnn::new(dev.clone())?;
    let args : Vec<String> = env::args().collect();
    if args.len() < 2 {
        std::process::exit(1);
    }
    let file_name = &args[1];
    let mut file = File::open(file_name)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let dev = cudarc::driver::CudaDevice::new(0)?;
    let cudnn = cudarc::cudnn::Cudnn::new(dev.clone())?;
    
    let tensors = SafeTensors::deserialize(&buffer)?;

    let mut data_ptrs: Vec<&[u8]> = Vec::new();
    let mut shape_ptrs: Vec<Vec<usize>> = Vec::new();
    let mut k = 0;
    let mut input_index = 0;

    for (name, tensor) in tensors.tensors() {

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

    let input_shape = vec![shape_ptrs[input_index][2], shape_ptrs[input_index][3]];
    let prep_data = data_ptrs[input_index];

    let vec: &[f32] = bytemuck::cast_slice(prep_data); // здесь вектор со значениями input_image
    let prep_shape = shape_ptrs[input_index].iter().map(|&x| x as i32).collect::<Vec<i32>>().into_boxed_slice();
    let prep_shape: [i32; 4] = prep_shape.to_vec().try_into().expect("Failed");


    let x = dev.htod_copy(vec.to_vec())?; // наше входное изображение, данные на gpu
    let x_disc = cudnn::Cudnn::create_4d_tensor::<f32>(&cudnn, cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, prep_shape)?;

    let mut y: CudaSlice<f32> = dev.alloc_zeros(shape_ptrs[input_index][0]*shape_ptrs[input_index][1]*shape_ptrs[input_index][2]*shape_ptrs[input_index][3])?;
    let y_disc =cudnn::Cudnn::create_4d_tensor::<f32>(&cudnn, cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, prep_shape)?;

    let vec: &[f32] = bytemuck::cast_slice(data_ptrs[kernel_index]);
    let kernel = dev.htod_copy(vec.to_vec())?;
    let prep_shape = shape_ptrs[kernel_index].iter().map(|&x| x as i32).collect::<Vec<i32>>().into_boxed_slice();

    let prep_shape: [i32; 4] = prep_shape.to_vec().try_into().expect("Failed");


    let filter = cudnn::Cudnn::create_nd_filter::<f32>(&cudnn, cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, &prep_shape)?;
    let conv = cudnn::Cudnn::create_conv2d::<f32>(&cudnn, [1;2], [1;2],[1;2], cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION)?;
     
    
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
  


        let prep_output: &[u8] = bytemuck::cast_slice(&y_host);
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("output_tensor".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape_ptrs[input_index].clone(), prep_output)?);
        let serialized_data = serialize(&tensors, &None)?;
        let mut file = File::create("outputcudnn.safetensors")?;
        file.write_all(&serialized_data)?;
 
    Ok(())
}
