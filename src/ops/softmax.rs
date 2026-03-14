use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{buffer_kind, GpuContext, Result, VolticError, ID};

const PASS1_SHADER: &str = include_str!("shaders/softmax_pass1.wgsl");
const PASS2_SHADER: &str = include_str!("shaders/softmax_pass2.wgsl");
const BACKWARD_SHADER: &str = include_str!("shaders/softmax_backward.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxDims {
    outer: u32,
    reduce: u32,
    inner: u32,
}

#[derive(Debug)]
pub struct SoftmaxOp {
    input: ID,
    output: ID,
    outer: u32,
    reduce: u32,
    inner: u32,
}

impl SoftmaxOp {
    pub fn new(input: ID, output: ID, outer: u32, reduce: u32, inner: u32) -> Self {
        Self {
            input,
            output,
            outer,
            reduce,
            inner,
        }
    }
}

impl super::Op for SoftmaxOp {
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
        vec!["softmax_pass1", "softmax_pass2", "softmax_backward"]
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
            ("softmax_pass1", make("softmax_pass1", PASS1_SHADER)),
            ("softmax_pass2", make("softmax_pass2", PASS2_SHADER)),
            (
                "softmax_backward",
                make("softmax_backward", BACKWARD_SHADER),
            ),
        ]
    }

    fn buffers_needed(&self, _shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        let n_partials = self.outer * self.inner * (self.reduce + 255) / 256;
        let input_n = self.outer * self.reduce * self.inner;
        vec![
            (self.input, buffer_kind::GRAD, input_n),
            (self.output, buffer_kind::PARTIAL, n_partials), // partials_max
            (self.output, buffer_kind::PARTIAL_SUM, n_partials), // partials_sum
        ]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let dims = SoftmaxDims {
            outer: self.outer,
            reduce: self.reduce,
            inner: self.inner,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("softmax_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let input_buf = ctx
            .buffers
            .get(&self.input)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.input)))?;
        let output_buf = ctx
            .buffers
            .get(&self.output)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.output)))?;
        let partials_max_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::PARTIAL))
            .ok_or_else(|| VolticError::Internal("softmax partials_max not found".into()))?;
        let partials_sum_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::PARTIAL_SUM))
            .ok_or_else(|| VolticError::Internal("softmax partials_sum not found".into()))?;

        let n_chunks = (self.reduce + 255) / 256;

        // Pass 1 — compute partial max and sum(exp(x - max))
        {
            let pipeline = ctx
                .pipelines
                .get("softmax_pass1")
                .ok_or_else(|| VolticError::Internal("softmax_pass1 pipeline not found".into()))?;
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("softmax_pass1_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: partials_max_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: partials_sum_buf.as_entire_binding(),
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
                    label: Some("softmax_pass1_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.outer * self.inner * n_chunks, 1, 1);
        }

        // Pass 2 — apply softmax using reduced max and sum
        {
            let pipeline = ctx
                .pipelines
                .get("softmax_pass2")
                .ok_or_else(|| VolticError::Internal("softmax_pass2 pipeline not found".into()))?;
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("softmax_pass2_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: partials_max_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: partials_sum_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dims_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("softmax_pass2_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.outer * self.reduce * self.inner).div_ceil(256), 1, 1);
        }

        Ok(())
    }

    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("softmax_backward")
            .ok_or_else(|| VolticError::Internal("softmax_backward pipeline not found".into()))?;

        let output_buf = ctx
            .buffers
            .get(&self.output)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.output)))?;
        let grad_out_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("softmax grad_out not found".into()))?;
        let grad_in_buf = ctx
            .training_buffers
            .get(&(self.input, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("softmax grad_input not found".into()))?;

        let dims = SoftmaxDims {
            outer: self.outer,
            reduce: self.reduce,
            inner: self.inner,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("softmax_backward_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_backward_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_in_buf.as_entire_binding(),
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
                label: Some("softmax_backward_pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.outer * self.reduce * self.inner).div_ceil(256), 1, 1);

        Ok(())
    }
}
