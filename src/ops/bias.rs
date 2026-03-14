use crate::{buffer_kind, GpuContext, Result, VolticError, ID};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BiasDims {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarDims {
    n: u32,
    scalar: f32,
}

macro_rules! impl_bias_op {
    ($name:ident, $fwd_key:literal, $bwd_key:literal, $fwd_shader:expr, $bwd_shader:expr) => {
        #[derive(Debug)]
        pub struct $name {
            inputs: [ID; 2],
            output: ID,
            rows: u32,
            cols: u32,
        }

        impl $name {
            pub fn new(input: ID, bias: ID, output: ID, rows: u32, cols: u32) -> Self {
                Self {
                    inputs: [input, bias],
                    output,
                    rows,
                    cols,
                }
            }
        }

        impl super::Op for $name {
            fn inputs(&self) -> &[ID] {
                &self.inputs
            }
            fn outputs(&self) -> &[ID] {
                std::slice::from_ref(&self.output)
            }

            fn infer_shape(&self, input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
                Ok(input_shapes[0].clone())
            }

            fn pipeline_keys(&self) -> Vec<&'static str> {
                vec![$fwd_key, $bwd_key]
            }

            fn create_pipelines(
                &self,
                device: &wgpu::Device,
            ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
                let make = |label, src: &str| {
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                };
                vec![
                    ($fwd_key, make($fwd_key, $fwd_shader)),
                    ($bwd_key, make($bwd_key, $bwd_shader)),
                ]
            }

            fn buffers_needed(
                &self,
                shapes: &HashMap<ID, Vec<u32>>,
            ) -> Vec<(ID, &'static str, u32)> {
                let input_n: u32 = shapes[&self.inputs[0]].iter().product();
                let bias_n: u32 = shapes[&self.inputs[1]].iter().product();
                vec![
                    (self.inputs[0], buffer_kind::GRAD, input_n),
                    (self.inputs[1], buffer_kind::GRAD, bias_n),
                ]
            }

            fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
                let pipeline = ctx
                    .pipelines
                    .get($fwd_key)
                    .ok_or_else(|| VolticError::Internal(concat!($fwd_key, " not found").into()))?;
                let in_buf = ctx.buffers.get(&self.inputs[0]).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.inputs[0]))
                })?;
                let bias_buf = ctx.buffers.get(&self.inputs[1]).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.inputs[1]))
                })?;
                let out_buf = ctx.buffers.get(&self.output).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.output))
                })?;
                let dims = BiasDims {
                    rows: self.rows,
                    cols: self.cols,
                };
                let dims_buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("bias_dims"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some($fwd_key),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: in_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: bias_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some($fwd_key),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((self.rows * self.cols).div_ceil(256), 1, 1);
                Ok(())
            }

            fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
                let pipeline = ctx
                    .pipelines
                    .get($bwd_key)
                    .ok_or_else(|| VolticError::Internal(concat!($bwd_key, " not found").into()))?;
                // let in_buf = ctx.buffers.get(&self.inputs[0]).ok_or_else(|| {
                //     VolticError::Internal(format!("buffer not found: {:?}", self.inputs[0]))
                // })?;
                // let bias_buf = ctx.buffers.get(&self.inputs[1]).ok_or_else(|| {
                //     VolticError::Internal(format!("buffer not found: {:?}", self.inputs[1]))
                // })?;
                let grad_out_buf = ctx
                    .training_buffers
                    .get(&(self.output, buffer_kind::GRAD))
                    .ok_or_else(|| VolticError::Internal("bias grad_out not found".into()))?;
                let grad_in_buf = ctx
                    .training_buffers
                    .get(&(self.inputs[0], buffer_kind::GRAD))
                    .ok_or_else(|| VolticError::Internal("bias grad_input not found".into()))?;
                let grad_bias_buf = ctx
                    .training_buffers
                    .get(&(self.inputs[1], buffer_kind::GRAD))
                    .ok_or_else(|| VolticError::Internal("bias grad_bias not found".into()))?;
                let dims = BiasDims {
                    rows: self.rows,
                    cols: self.cols,
                };
                let dims_buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("bias_bwd_dims"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some($bwd_key),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: grad_out_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: grad_in_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: grad_bias_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some($bwd_key),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.cols.div_ceil(256), 1, 1);
                Ok(())
            }
        }
    };
}

macro_rules! impl_scalar_op {
    ($name:ident, $fwd_key:literal, $bwd_key:literal, $fwd_shader:expr, $bwd_shader:expr) => {
        #[derive(Debug)]
        pub struct $name {
            input: ID,
            output: ID,
            n: u32,
            scalar: f32,
        }

        impl $name {
            pub fn new(input: ID, output: ID, n: u32, scalar: f32) -> Self {
                Self {
                    input,
                    output,
                    n,
                    scalar,
                }
            }
        }

        impl super::Op for $name {
            fn inputs(&self) -> &[ID] {
                std::slice::from_ref(&self.input)
            }
            fn outputs(&self) -> &[ID] {
                std::slice::from_ref(&self.output)
            }

            fn infer_shape(&self, input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
                Ok(input_shapes[0].clone())
            }

            fn pipeline_keys(&self) -> Vec<&'static str> {
                vec![$fwd_key, $bwd_key]
            }

            fn create_pipelines(
                &self,
                device: &wgpu::Device,
            ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
                let make = |label, src: &str| {
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                };
                vec![
                    ($fwd_key, make($fwd_key, $fwd_shader)),
                    ($bwd_key, make($bwd_key, $bwd_shader)),
                ]
            }

            fn buffers_needed(
                &self,
                shapes: &HashMap<ID, Vec<u32>>,
            ) -> Vec<(ID, &'static str, u32)> {
                let n: u32 = shapes[&self.input].iter().product();
                vec![(self.input, buffer_kind::GRAD, n)]
            }

            fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
                let pipeline = ctx
                    .pipelines
                    .get($fwd_key)
                    .ok_or_else(|| VolticError::Internal(concat!($fwd_key, " not found").into()))?;
                let in_buf = ctx.buffers.get(&self.input).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.input))
                })?;
                let out_buf = ctx.buffers.get(&self.output).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.output))
                })?;
                let dims = ScalarDims {
                    n: self.n,
                    scalar: self.scalar,
                };
                let dims_buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("scalar_dims"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some($fwd_key),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: in_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some($fwd_key),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.n.div_ceil(256), 1, 1);
                Ok(())
            }

            fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
                let pipeline = ctx
                    .pipelines
                    .get($bwd_key)
                    .ok_or_else(|| VolticError::Internal(concat!($bwd_key, " not found").into()))?;
                let grad_out_buf = ctx
                    .training_buffers
                    .get(&(self.output, buffer_kind::GRAD))
                    .ok_or_else(|| VolticError::Internal("scalar grad_out not found".into()))?;
                let grad_in_buf = ctx
                    .training_buffers
                    .get(&(self.input, buffer_kind::GRAD))
                    .ok_or_else(|| VolticError::Internal("scalar grad_input not found".into()))?;
                let dims = ScalarDims {
                    n: self.n,
                    scalar: self.scalar,
                };
                let dims_buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("scalar_bwd_dims"),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some($bwd_key),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: grad_out_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: grad_in_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some($bwd_key),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.n.div_ceil(256), 1, 1);
                Ok(())
            }
        }
    };
}

// Bias ops
impl_bias_op!(
    BiasAddOp,
    "bias_add",
    "bias_add_bwd",
    include_str!("shaders/bias_add.wgsl"),
    include_str!("shaders/bias_add_backward.wgsl")
);
impl_bias_op!(
    BiasSubOp,
    "bias_sub",
    "bias_sub_bwd",
    include_str!("shaders/bias_sub.wgsl"),
    include_str!("shaders/bias_sub_backward.wgsl")
);
impl_bias_op!(
    BiasMulOp,
    "bias_mul",
    "bias_mul_bwd",
    include_str!("shaders/bias_mul.wgsl"),
    include_str!("shaders/bias_mul_backward.wgsl")
);
impl_bias_op!(
    BiasDivOp,
    "bias_div",
    "bias_div_bwd",
    include_str!("shaders/bias_div.wgsl"),
    include_str!("shaders/bias_div_backward.wgsl")
);

// Scalar ops
impl_scalar_op!(
    ScalarAddOp,
    "scalar_add",
    "scalar_add_bwd",
    include_str!("shaders/scalar_add.wgsl"),
    include_str!("shaders/scalar_add_backward.wgsl")
);
impl_scalar_op!(
    ScalarSubOp,
    "scalar_sub",
    "scalar_sub_bwd",
    include_str!("shaders/scalar_sub.wgsl"),
    include_str!("shaders/scalar_sub_backward.wgsl")
);
impl_scalar_op!(
    ScalarMulOp,
    "scalar_mul",
    "scalar_mul_bwd",
    include_str!("shaders/scalar_mul.wgsl"),
    include_str!("shaders/scalar_mul_backward.wgsl")
);
impl_scalar_op!(
    ScalarDivOp,
    "scalar_div",
    "scalar_div_bwd",
    include_str!("shaders/scalar_div.wgsl"),
    include_str!("shaders/scalar_div_backward.wgsl")
);
