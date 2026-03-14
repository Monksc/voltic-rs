use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{buffer_kind, GpuContext, Result, VolticError, ID};

macro_rules! impl_group_op {
    ($name:ident, $shader:expr, $key:literal) => {
        #[derive(Debug)]
        pub struct $name {
            input: ID,
            output: ID,
            n: u32,
            group_size: u32,
        }

        impl $name {
            pub fn new(input: ID, output: ID, n: u32, group_size: u32) -> Self {
                Self {
                    input,
                    output,
                    n,
                    group_size,
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

            fn infer_shape(&self, _input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
                let num_groups = self.n / self.group_size;
                Ok(vec![num_groups])
            }

            fn pipeline_keys(&self) -> Vec<&'static str> {
                vec![$key]
            }

            fn create_pipelines(
                &self,
                device: &wgpu::Device,
            ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(concat!($key, "_shader")),
                    source: wgpu::ShaderSource::Wgsl($shader.into()),
                });

                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(concat!($key, "_pipeline")),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

                vec![($key, pipeline)]
            }

            fn buffers_needed(
                &self,
                _shapes: &HashMap<ID, Vec<u32>>,
            ) -> Vec<(ID, &'static str, u32)> {
                vec![]
            }

            fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
                let pipeline = ctx.pipelines.get($key).ok_or_else(|| {
                    VolticError::Internal(concat!($key, " pipeline not found").into())
                })?;

                let input_buf = ctx.buffers.get(&self.input).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.input))
                })?;
                let output_buf = ctx.buffers.get(&self.output).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.output))
                })?;

                #[repr(C)]
                #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
                struct GroupDims {
                    n: u32,
                    group_size: u32,
                    num_groups: u32,
                    _pad: u32,
                }

                let num_groups = self.n / self.group_size;
                let dims = GroupDims {
                    n: self.n,
                    group_size: self.group_size,
                    num_groups,
                    _pad: 0,
                };
                let dims_buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(concat!($key, "_dims")),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(concat!($key, "_bind_group")),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

                let workgroups = num_groups.div_ceil(256);

                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(concat!($key, "_pass")),
                        timestamp_writes: None,
                    });

                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);

                Ok(())
            }
        }
    };
}

impl_group_op!(
    GroupSumOp,
    include_str!("shaders/group_sum.wgsl"),
    "group_sum"
);
impl_group_op!(
    GroupMaxOp,
    include_str!("shaders/group_max.wgsl"),
    "group_max"
);
impl_group_op!(
    GroupMulOp,
    include_str!("shaders/group_mul.wgsl"),
    "group_mul"
);
