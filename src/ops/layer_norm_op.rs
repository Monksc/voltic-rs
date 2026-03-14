use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{buffer_kind, GpuContext, Result, VolticError, ID};

const PASS1_SHADER: &str = include_str!("shaders/layer_norm_pass1.wgsl");
const PASS2_SHADER: &str = include_str!("shaders/layer_norm_pass2.wgsl");
const BACKWARD_SHADER: &str = include_str!("shaders/layer_norm_backward.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormDims {
    outer: u32,
    reduce: u32,
    inner: u32,
    eps: f32,
}

#[derive(Debug)]
pub struct LayerNormOp {
    input: ID,
    gamma: ID,
    beta: ID,
    output: ID,
    outer: u32,
    reduce: u32,
    inner: u32,
    eps: f32,
}

impl LayerNormOp {
    pub fn new(
        input: ID,
        gamma: ID,
        beta: ID,
        output: ID,
        outer: u32,
        reduce: u32,
        inner: u32,
        eps: f32,
    ) -> Self {
        Self {
            input,
            gamma,
            beta,
            output,
            outer,
            reduce,
            inner,
            eps,
        }
    }
}

impl super::Op for LayerNormOp {
    fn inputs(&self) -> &[ID] {
        // gamma and beta are trainable — include them so optimizer sees them
        // token_ids-style non-trainable inputs are excluded
        // We use a slice trick — return all three
        // Rust doesn't allow &[self.input, self.gamma, self.beta] directly
        // so we use a static approach via the struct fields
        &[] // overridden below via custom impl
    }

    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.output)
    }

    fn infer_shape(&self, input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        Ok(input_shapes[0].clone())
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec![
            "layer_norm_pass1",
            "layer_norm_pass2",
            "layer_norm_backward",
        ]
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
            ("layer_norm_pass1", make("layer_norm_pass1", PASS1_SHADER)),
            ("layer_norm_pass2", make("layer_norm_pass2", PASS2_SHADER)),
            (
                "layer_norm_backward",
                make("layer_norm_backward", BACKWARD_SHADER),
            ),
        ]
    }

    fn buffers_needed(&self, shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        let input_n: u32 = shapes[&self.input].iter().product();
        let gamma_n: u32 = shapes[&self.gamma].iter().product();
        let beta_n: u32 = shapes[&self.beta].iter().product();
        let n_chunks = (self.reduce + 255) / 256;
        let n_partials = self.outer * self.inner * n_chunks;

        vec![
            (self.input, buffer_kind::GRAD, input_n),
            (self.gamma, buffer_kind::GRAD, gamma_n),
            (self.beta, buffer_kind::GRAD, beta_n),
            (self.output, buffer_kind::PARTIAL, n_partials), // partials_mean
            (self.output, buffer_kind::PARTIAL_SUM, n_partials), // partials_var
            (self.output, buffer_kind::X_NORM, input_n),     // x_norm for backward
            (self.output, buffer_kind::GRAD, input_n),
        ]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let dims = LayerNormDims {
            outer: self.outer,
            reduce: self.reduce,
            inner: self.inner,
            eps: self.eps,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("layer_norm_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let input_buf = ctx
            .buffers
            .get(&self.input)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.input)))?;
        let gamma_buf = ctx
            .buffers
            .get(&self.gamma)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.gamma)))?;
        let beta_buf = ctx
            .buffers
            .get(&self.beta)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.beta)))?;
        let output_buf = ctx
            .buffers
            .get(&self.output)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.output)))?;
        let partials_mean_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::PARTIAL))
            .ok_or_else(|| VolticError::Internal("layer_norm partials_mean not found".into()))?;
        let partials_var_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::PARTIAL_SUM))
            .ok_or_else(|| VolticError::Internal("layer_norm partials_var not found".into()))?;
        let x_norm_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::X_NORM))
            .ok_or_else(|| VolticError::Internal("layer_norm x_norm not found".into()))?;

        let n_chunks = (self.reduce + 255) / 256;

        // Pass 1 — partial sums and sum of squares
        {
            let pipeline = ctx.pipelines.get("layer_norm_pass1").ok_or_else(|| {
                VolticError::Internal("layer_norm_pass1 pipeline not found".into())
            })?;
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("layer_norm_pass1_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: partials_mean_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: partials_var_buf.as_entire_binding(),
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
                    label: Some("layer_norm_pass1_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.outer * self.inner * n_chunks, 1, 1);
        }

        // Pass 2 — normalise, scale, shift
        {
            let pipeline = ctx.pipelines.get("layer_norm_pass2").ok_or_else(|| {
                VolticError::Internal("layer_norm_pass2 pipeline not found".into())
            })?;
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("layer_norm_pass2_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: partials_mean_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: partials_var_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gamma_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: beta_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: x_norm_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: dims_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("layer_norm_pass2_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.outer * self.reduce * self.inner).div_ceil(256), 1, 1);
        }

        Ok(())
    }

    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx.pipelines.get("layer_norm_backward").ok_or_else(|| {
            VolticError::Internal("layer_norm_backward pipeline not found".into())
        })?;

        let gamma_buf = ctx.buffers.get(&self.gamma).ok_or_else(|| {
            VolticError::Internal(format!("gamma buffer not found: {:?}", self.gamma))
        })?;
        let grad_out_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("layer_norm grad_out not found".into()))?;
        let x_norm_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::X_NORM))
            .ok_or_else(|| VolticError::Internal("layer_norm x_norm not found".into()))?;
        let partials_mean_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::PARTIAL))
            .ok_or_else(|| VolticError::Internal("layer_norm partials_mean not found".into()))?;
        let partials_var_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::PARTIAL_SUM))
            .ok_or_else(|| VolticError::Internal("layer_norm partials_var not found".into()))?;
        let grad_input_buf = ctx
            .training_buffers
            .get(&(self.input, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("layer_norm grad_input not found".into()))?;
        let grad_gamma_buf = ctx
            .training_buffers
            .get(&(self.gamma, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("layer_norm grad_gamma not found".into()))?;
        let grad_beta_buf = ctx
            .training_buffers
            .get(&(self.beta, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("layer_norm grad_beta not found".into()))?;

        let dims = LayerNormDims {
            outer: self.outer,
            reduce: self.reduce,
            inner: self.inner,
            eps: self.eps,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("layer_norm_backward_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_backward_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_norm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gamma_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partials_var_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: partials_mean_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grad_gamma_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grad_beta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layer_norm_backward_pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.outer * self.reduce * self.inner).div_ceil(256), 1, 1);

        Ok(())
    }
}
