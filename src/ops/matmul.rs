use std::collections::HashMap;

use crate::{GpuContext, ID, Result, VolticError, buffer_kind};
use wgpu::util::DeviceExt;

const TILE: u32 = 16;
const BACKWARD_WEIGHTS_SHADER: &str = include_str!("shaders/matmul_backward_weights.wgsl");
const BACKWARD_INPUT_SHADER: &str = include_str!("shaders/matmul_backward_input.wgsl");
const SHADER: &str = include_str!("shaders/matmul.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatMulDims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32, // uniforms require 16-byte alignment
}

#[derive(Debug, Clone)]
pub struct MatMulOp {
    inputs: [ID; 2],
    output: ID,
    m: u32,
    k: u32,
    n: u32,
}

impl MatMulOp {
    pub fn new(lhs: ID, rhs: ID, output: ID, m: u32, k: u32, n: u32) -> Self {
        Self {
            inputs: [lhs, rhs],
            output,
            m,
            k,
            n,
        }
    }
}

impl super::Op for MatMulOp {
    fn inputs(&self) -> &[ID] {
        &self.inputs
    }
    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.output)
    }

    fn infer_shape(&self, input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        let a = input_shapes[0];
        let b = input_shapes[1];

        if a.len() != 2 || b.len() != 2 {
            return Err(VolticError::InvalidDimension {
                dim: a.len(),
                ndim: 2,
            });
        }

        if a[1] != b[0] {
            return Err(VolticError::MatrixMultiplyMismatch {
                lhs: (a[0], a[1]),
                rhs: (b[0], b[1]),
            });
        }

        Ok(vec![a[0], b[1]])
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec!["matmul"]
    }

    fn create_pipelines(
        &self,
        device: &wgpu::Device,
    ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_backward_weights_shader"),
            source: wgpu::ShaderSource::Wgsl(BACKWARD_WEIGHTS_SHADER.into()),
        });
        let bw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_backward_weights_pipeline"),
            layout: None,
            module: &bw_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bi_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_backward_input_shader"),
            source: wgpu::ShaderSource::Wgsl(BACKWARD_INPUT_SHADER.into()),
        });
        let bi_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_backward_input_pipeline"),
            layout: None,
            module: &bi_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        vec![
            ("matmul", pipeline),
            ("matmul_backward_weights", bw_pipeline),
            ("matmul_backward_input", bi_pipeline),
        ]
    }

    fn buffers_needed(&self, shapes: &HashMap<ID, Vec<u32>>) -> Vec<(ID, &'static str, u32)> {
        let x_n: u32 = shapes[&self.inputs[0]].iter().product();
        let w_n: u32 = shapes[&self.inputs[1]].iter().product();
        vec![
            (self.inputs[0], buffer_kind::GRAD, x_n),
            (self.inputs[1], buffer_kind::GRAD, w_n),
        ]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("matmul")
            .ok_or_else(|| VolticError::Internal("matmul pipeline not found".into()))?;

        let a_buf = ctx.buffers.get(&self.inputs[0]).ok_or_else(|| {
            VolticError::Internal(format!("buffer not found: {:?}", self.inputs[0]))
        })?;
        let b_buf = ctx.buffers.get(&self.inputs[1]).ok_or_else(|| {
            VolticError::Internal(format!("buffer not found: {:?}", self.inputs[1]))
        })?;
        let c_buf = ctx
            .buffers
            .get(&self.output)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.output)))?;

        let dims = MatMulDims {
            m: self.m,
            k: self.k,
            n: self.n,
            _pad: 0,
        };
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
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let wg_x = self.n.div_ceil(TILE);
        let wg_y = self.m.div_ceil(TILE);

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);

        Ok(())
    }

    fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
        let dims = MatMulDims {
            m: self.m,
            k: self.k,
            n: self.n,
            _pad: 0,
        };
        let dims_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_backward_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let x_buf = ctx.buffers.get(&self.inputs[0]).ok_or_else(|| {
            VolticError::Internal(format!("x buffer not found: {:?}", self.inputs[0]))
        })?;
        let w_buf = ctx.buffers.get(&self.inputs[1]).ok_or_else(|| {
            VolticError::Internal(format!("w buffer not found: {:?}", self.inputs[1]))
        })?;

        // grad_y comes from mse backward — lives in training_buffers
        let grad_y_buf = ctx
            .training_buffers
            .get(&(self.outputs()[0], buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("grad_y buffer not found".into()))?;

        let grad_w_buf = ctx
            .training_buffers
            .get(&(self.inputs[1], buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("grad_w buffer not found".into()))?;
        let grad_x_buf = ctx
            .training_buffers
            .get(&(self.inputs[0], buffer_kind::GRAD))
            .ok_or_else(|| VolticError::Internal("grad_x buffer not found".into()))?;

        // dL/dW = X^T @ grad_y
        let bw_pipeline = ctx
            .pipelines
            .get("matmul_backward_weights")
            .ok_or_else(|| {
                VolticError::Internal("matmul_backward_weights pipeline not found".into())
            })?;

        let bw_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bw_bind_group"),
            layout: &bw_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("matmul_backward_weights_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(bw_pipeline);
            pass.set_bind_group(0, &bw_bind_group, &[]);
            pass.dispatch_workgroups(self.n.div_ceil(TILE), self.k.div_ceil(TILE), 1);
        }

        // dL/dX = grad_y @ W^T
        let bi_pipeline = ctx.pipelines.get("matmul_backward_input").ok_or_else(|| {
            VolticError::Internal("matmul_backward_input pipeline not found".into())
        })?;

        let bi_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bi_bind_group"),
            layout: &bi_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("matmul_backward_input_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(bi_pipeline);
            pass.set_bind_group(0, &bi_bind_group, &[]);
            pass.dispatch_workgroups(self.k.div_ceil(TILE), self.m.div_ceil(TILE), 1);
        }

        Ok(())
    }
}
