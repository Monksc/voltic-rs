use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{GpuContext, ID, Result, VolticError, buffer_kind};

const SHADER: &str = include_str!("shaders/matmul.wgsl");
const BACKWARD_INPUT: &str = include_str!("shaders/matmul_backward_input.wgsl");
const BACKWARD_WEIGHTS: &str = include_str!("shaders/matmul_backward_weights.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatMulDims {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    rhs_batched: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug)]
pub struct MatMulOp {
    lhs: ID,
    rhs: ID,
    output: ID,
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    rhs_batched: u32,
}

impl MatMulOp {
    pub fn new(
        lhs: ID,
        rhs: ID,
        output: ID,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
        rhs_batched: u32,
    ) -> Self {
        Self {
            lhs,
            rhs,
            output,
            batch,
            m,
            k,
            n,
            rhs_batched,
        }
    }

    fn make_dims(&self) -> MatMulDims {
        MatMulDims {
            batch: self.batch,
            m: self.m,
            k: self.k,
            n: self.n,
            rhs_batched: self.rhs_batched,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    fn dispatch(&self) -> (u32, u32, u32) {
        ((self.n + 15) / 16, (self.m + 15) / 16, self.batch)
    }
}

impl super::Op for MatMulOp {
    fn inputs(&self) -> &[ID] {
        // stored as [lhs, rhs] — use a helper or store as array
        // for now return slice via unsafe-free trick
        unsafe { std::slice::from_raw_parts(&self.lhs, 2) }
    }

    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.output)
    }

    fn infer_shape(&self, _input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        // output is [...batch_dims, M, N] — batch is already flattened in the op
        // we return [batch, M, N] for rank>2, or [M, N] for rank==2
        if self.batch == 1 {
            Ok(vec![self.m, self.n])
        } else {
            Ok(vec![self.batch, self.m, self.n])
        }
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec!["matmul", "matmul_backward_input", "matmul_backward_weights"]
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
            ("matmul", make("matmul", SHADER)),
            (
                "matmul_backward_input",
                make("matmul_backward_input", BACKWARD_INPUT),
            ),
            (
                "matmul_backward_weights",
                make("matmul_backward_weights", BACKWARD_WEIGHTS),
            ),
        ]
    }

    fn buffers_needed(&self, _shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        let lhs_n = self.batch * self.m * self.k;
        let rhs_n = if self.rhs_batched == 1 {
            self.batch * self.k * self.n
        } else {
            self.k * self.n
        };
        let output_n = self.batch * self.m * self.n;
        vec![
            (self.lhs, buffer_kind::GRAD, lhs_n),
            (self.rhs, buffer_kind::GRAD, rhs_n),
            (self.output, buffer_kind::GRAD, output_n),
        ]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("matmul")
            .ok_or_else(|| VolticError::Internal("matmul pipeline not found".into()))?;

        let lhs_buf = ctx.buffers.get(&self.lhs).ok_or_else(|| {
            VolticError::Internal(format!("lhs buffer not found: {:?}", self.lhs))
        })?;
        let rhs_buf = ctx.buffers.get(&self.rhs).ok_or_else(|| {
            VolticError::Internal(format!("rhs buffer not found: {:?}", self.rhs))
        })?;
        let output_buf = ctx.buffers.get(&self.output).ok_or_else(|| {
            VolticError::Internal(format!("output buffer not found: {:?}", self.output))
        })?;

        let dims = self.make_dims();
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let (dx, dy, dz) = self.dispatch();
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dx, dy, dz);
        Ok(())
    }

    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        let dims = self.make_dims();
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_backward_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let lhs_buf = ctx.buffers.get(&self.lhs).ok_or_else(|| {
            VolticError::Internal(format!("lhs buffer not found: {:?}", self.lhs))
        })?;
        let rhs_buf = ctx.buffers.get(&self.rhs).ok_or_else(|| {
            VolticError::Internal(format!("rhs buffer not found: {:?}", self.rhs))
        })?;
        let grad_out_buf = ctx
            .training_buffers
            .get(&(self.output, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("matmul grad_out not found".into()))?;
        let grad_lhs_buf = ctx
            .training_buffers
            .get(&(self.lhs, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("matmul grad_lhs not found".into()))?;
        let grad_rhs_buf = ctx
            .training_buffers
            .get(&(self.rhs, buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("matmul grad_rhs not found".into()))?;

        // let (dx, dy, dz) = self.dispatch();

        // grad_A = grad_C @ B^T  — shape [batch, M, K]
        {
            let pipeline = ctx.pipelines.get("matmul_backward_input").ok_or_else(|| {
                VolticError::Internal("matmul_backward_input pipeline not found".into())
            })?;
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_bwd_input_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: grad_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grad_lhs_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buf.as_entire_binding(),
                    },
                ],
            });
            // dispatch is [K/16, M/16, batch]
            let dx_a = (self.k + 15) / 16;
            let dy_a = (self.m + 15) / 16;
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("matmul_bwd_input_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dx_a, dy_a, self.batch);
        }

        // grad_B = A^T @ grad_C  — shape [batch, K, N]
        {
            let pipeline = ctx
                .pipelines
                .get("matmul_backward_weights")
                .ok_or_else(|| {
                    VolticError::Internal("matmul_backward_weights pipeline not found".into())
                })?;
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_bwd_weights_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: grad_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lhs_buf.as_entire_binding(),
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
            });
            // dispatch is [N/16, K/16, batch]
            let dx_b = (self.n + 15) / 16;
            let dy_b = (self.k + 15) / 16;
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("matmul_bwd_weights_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dx_b, dy_b, self.batch);
        }

        Ok(())
    }
}
