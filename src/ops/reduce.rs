use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{buffer_kind, GpuContext, Result, VolticError, ID};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceDims {
    outer: u32,
    reduce: u32,
    inner: u32,
}

macro_rules! impl_reduce_op {
    (
        $name:ident,
        $fwd_pass1_key:literal,
        $fwd_pass2_key:literal,
        $bwd_key:literal,
        $fwd_pass1_shader:expr,
        $fwd_pass2_shader:expr,
        $bwd_shader:expr,
        $has_backward:literal
    ) => {
        #[derive(Debug)]
        pub struct $name {
            input: ID,
            output: ID,
            outer: u32,  // product of dims before reduce axis
            reduce: u32, // size of the axis being reduced
            inner: u32,  // product of dims after reduce axis
        }

        impl $name {
            pub fn new(input: ID, output: ID, outer: u32, reduce: u32, inner: u32) -> Self {
                Self {
                    input,
                    output,
                    outer,
                    reduce,
                    inner,
                }
            }

            // Compute output shape by removing the reduce axis
            pub fn infer_output_shape(input_shape: &[u32], axis: usize) -> Vec<u32> {
                input_shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != axis)
                    .map(|(_, &d)| d)
                    .collect()
            }

            // Compute outer, reduce, inner from shape and axis
            pub fn compute_dims(shape: &[u32], axis: usize) -> (u32, u32, u32) {
                let outer: u32 = shape[..axis].iter().product();
                let reduce: u32 = shape[axis];
                let inner: u32 = shape[axis + 1..].iter().product();
                (outer, reduce, inner)
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
                // Output is input with reduce dimension removed
                // We can't recover which axis was reduced here, but the shape
                // is already stored in Context at graph build time
                Ok(input_shapes[0].clone())
            }

            fn pipeline_keys(&self) -> Vec<&'static str> {
                if $has_backward {
                    vec![$fwd_pass1_key, $fwd_pass2_key, $bwd_key]
                } else {
                    vec![$fwd_pass1_key, $fwd_pass2_key]
                }
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

                let mut pipelines = vec![
                    ($fwd_pass1_key, make($fwd_pass1_key, $fwd_pass1_shader)),
                    ($fwd_pass2_key, make($fwd_pass2_key, $fwd_pass2_shader)),
                ];

                if $has_backward {
                    pipelines.push(($bwd_key, make($bwd_key, $bwd_shader)));
                }

                pipelines
            }

            fn buffers_needed(
                &self,
                _shapes: &HashMap<ID, Vec<u32>>,
            ) -> Vec<(ID, &'static str, u32)> {
                let n_partials = self.outer * self.inner * self.reduce.div_ceil(256);
                let input_n = self.outer * self.reduce * self.inner;
                let mut bufs = vec![(self.output, buffer_kind::PARTIAL, n_partials)];
                if $has_backward {
                    bufs.push((self.input, buffer_kind::GRAD, input_n));
                }
                bufs
            }

            fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
                let dims = ReduceDims {
                    outer: self.outer,
                    reduce: self.reduce,
                    inner: self.inner,
                };
                let dims_buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(concat!($fwd_pass1_key, "_dims")),
                        contents: bytemuck::bytes_of(&dims),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                let input_buf = ctx.buffers.get(&self.input).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.input))
                })?;
                let partials_buf = ctx
                    .training_buffers
                    .get(&(self.output, buffer_kind::PARTIAL))
                    .ok_or_else(|| {
                        VolticError::Internal(format!(
                            "partial buffer not found: {:?}",
                            self.output
                        ))
                    })?;
                let output_buf = ctx.buffers.get(&self.output).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.output))
                })?;

                // Pass 1 — partial reduction into partials buffer
                {
                    let pipeline = ctx.pipelines.get($fwd_pass1_key).ok_or_else(|| {
                        VolticError::Internal(concat!($fwd_pass1_key, " pipeline not found").into())
                    })?;
                    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(concat!($fwd_pass1_key, "_bind_group")),
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: partials_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: dims_buf.as_entire_binding(),
                            },
                        ],
                    });
                    let n_chunks = self.reduce.div_ceil(256);
                    let mut pass = ctx
                        .encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(concat!($fwd_pass1_key, "_pass")),
                            timestamp_writes: None,
                        });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(self.outer * self.inner * n_chunks, 1, 1);
                }

                // Pass 2 — reduce partials to output
                {
                    let pipeline = ctx.pipelines.get($fwd_pass2_key).ok_or_else(|| {
                        VolticError::Internal(concat!($fwd_pass2_key, " pipeline not found").into())
                    })?;
                    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(concat!($fwd_pass2_key, "_bind_group")),
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: partials_buf.as_entire_binding(),
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
                    let mut pass = ctx
                        .encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(concat!($fwd_pass2_key, "_pass")),
                            timestamp_writes: None,
                        });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups((self.outer * self.inner).div_ceil(256), 1, 1);
                }

                Ok(())
            }

            fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
                if !$has_backward {
                    return Ok(());
                }

                let pipeline = ctx.pipelines.get($bwd_key).ok_or_else(|| {
                    VolticError::Internal(concat!($bwd_key, " pipeline not found").into())
                })?;

                let grad_out_buf = ctx
                    .training_buffers
                    .get(&(self.output, buffer_kind::GRAD))
                    .ok_or_else(|| {
                        VolticError::Internal(format!("grad buffer not found: {:?}", self.output))
                    })?;
                let grad_in_buf = ctx
                    .training_buffers
                    .get(&(self.input, buffer_kind::GRAD))
                    .ok_or_else(|| {
                        VolticError::Internal(format!("grad buffer not found: {:?}", self.input))
                    })?;

                let dims = ReduceDims {
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

                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                            resource: dims_buf.as_entire_binding(),
                        },
                    ],
                });

                let total = self.outer * self.reduce * self.inner;
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(concat!($bwd_key, "_pass")),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total.div_ceil(256), 1, 1);

                Ok(())
            }
        }
    };
}

impl_reduce_op!(
    ReduceSumOp,
    "reduce_sum_pass1",
    "reduce_sum_pass2",
    "reduce_sum_backward",
    include_str!("shaders/reduce_sum.wgsl"),
    include_str!("shaders/reduce_sum_pass2.wgsl"),
    include_str!("shaders/reduce_sum_backward.wgsl"),
    true
);

impl_reduce_op!(
    ReduceMaxOp,
    "reduce_max_pass1",
    "reduce_max_pass2",
    "", // no backward
    include_str!("shaders/reduce_max.wgsl"),
    include_str!("shaders/reduce_max_pass2.wgsl"),
    "", // no backward shader
    false
);

impl_reduce_op!(
    ReduceMeanOp,
    "reduce_sum_pass1",  // pass1 is identical to sum — just accumulate partials
    "reduce_mean_pass2", // pass2 divides by reduce dim
    "reduce_mean_backward",
    include_str!("shaders/reduce_sum.wgsl"),
    include_str!("shaders/reduce_mean_pass2.wgsl"),
    include_str!("shaders/reduce_mean_backward.wgsl"),
    true
);
