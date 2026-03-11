use crate::{GpuContext, ID, Result, VolticError, buffer_kind};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BroadcastDims {
    outer: u32,
    reduce: u32,
    inner: u32,
}

macro_rules! impl_broadcast_op {
    (
        $name:ident,
        $fwd_key:literal,
        $bwd_key:literal,
        $fwd_shader:expr,
        $bwd_shader:expr,
        $extra_bwd_bindings:literal  // true for mul/div which need input+rhs in backward
    ) => {
        #[derive(Debug)]
        pub struct $name {
            inputs: [ID; 2], // [input, rhs]
            output: ID,
            outer: u32,
            reduce: u32,
            inner: u32,
        }

        impl $name {
            pub fn new(
                input: ID,
                rhs: ID,
                output: ID,
                outer: u32,
                reduce: u32,
                inner: u32,
            ) -> Self {
                Self {
                    inputs: [input, rhs],
                    output,
                    outer,
                    reduce,
                    inner,
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
                let make = |label, src: &'static str| {
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
                let rhs_n: u32 = shapes[&self.inputs[1]].iter().product();
                vec![
                    (self.inputs[0], buffer_kind::GRAD, input_n),
                    (self.inputs[1], buffer_kind::GRAD, rhs_n),
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
                let rhs_buf = ctx.buffers.get(&self.inputs[1]).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.inputs[1]))
                })?;
                let out_buf = ctx.buffers.get(&self.output).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.output))
                })?;

                let dims = BroadcastDims {
                    outer: self.outer,
                    reduce: self.reduce,
                    inner: self.inner,
                };
                let dims_buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(concat!($fwd_key, "_dims")),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(concat!($fwd_key, "_bind_group")),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: in_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: rhs_buf.as_entire_binding(),
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

                let total = self.outer * self.reduce * self.inner;
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(concat!($fwd_key, "_pass")),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
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
                    .ok_or_else(|| VolticError::Internal("broadcast grad_out not found".into()))?;
                let grad_in_buf = ctx
                    .training_buffers
                    .get(&(self.inputs[0], buffer_kind::GRAD))
                    .ok_or_else(|| {
                        VolticError::Internal("broadcast grad_input not found".into())
                    })?;
                let grad_rhs_buf = ctx
                    .training_buffers
                    .get(&(self.inputs[1], buffer_kind::GRAD))
                    .ok_or_else(|| VolticError::Internal("broadcast grad_rhs not found".into()))?;

                let dims = BroadcastDims {
                    outer: self.outer,
                    reduce: self.reduce,
                    inner: self.inner,
                };
                let dims_buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(concat!($bwd_key, "_dims")),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                // mul/div need original input and rhs for their backward
                let bind_group = if $extra_bwd_bindings {
                    let in_buf = ctx.buffers.get(&self.inputs[0]).ok_or_else(|| {
                        VolticError::Internal(format!("buffer not found: {:?}", self.inputs[0]))
                    })?;
                    let rhs_buf = ctx.buffers.get(&self.inputs[1]).ok_or_else(|| {
                        VolticError::Internal(format!("buffer not found: {:?}", self.inputs[1]))
                    })?;
                    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(concat!($bwd_key, "_bind_group")),
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: grad_out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: in_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: rhs_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: grad_in_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: grad_rhs_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: dims_buf.as_entire_binding(),
                            },
                        ],
                    })
                } else {
                    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(concat!($bwd_key, "_bind_group")),
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
                                resource: grad_rhs_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: dims_buf.as_entire_binding(),
                            },
                        ],
                    })
                };

                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(concat!($bwd_key, "_pass")),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((self.outer * self.inner).div_ceil(256), 1, 1);
                Ok(())
            }
        }
    };
}

impl_broadcast_op!(
    BroadcastAddOp,
    "broadcast_add",
    "broadcast_add_bwd",
    include_str!("shaders/broadcast_add.wgsl"),
    include_str!("shaders/broadcast_add_backward.wgsl"),
    false
);

impl_broadcast_op!(
    BroadcastSubOp,
    "broadcast_sub",
    "broadcast_sub_bwd",
    include_str!("shaders/broadcast_sub.wgsl"),
    include_str!("shaders/broadcast_sub_backward.wgsl"),
    false
);

impl_broadcast_op!(
    BroadcastMulOp,
    "broadcast_mul",
    "broadcast_mul_bwd",
    include_str!("shaders/broadcast_mul.wgsl"),
    include_str!("shaders/broadcast_mul_backward.wgsl"),
    true
);

impl_broadcast_op!(
    BroadcastDivOp,
    "broadcast_div",
    "broadcast_div_bwd",
    include_str!("shaders/broadcast_div.wgsl"),
    include_str!("shaders/broadcast_div_backward.wgsl"),
    true
);
