use crate::{buffer_kind, GpuContext, Result, VolticError, ID};
use std::{borrow::Cow, collections::HashMap};

macro_rules! impl_activation {
    (
        $name:ident,
        $forward_shader:expr,
        $backward_shader:expr,
        $forward_key:literal,
        $backward_key:literal,
        // tanh/sigmoid reuse output in backward, relu reuses input
        $backward_source:ident
    ) => {
        #[derive(Debug)]
        pub struct $name {
            input: ID,
            output: ID,
            n: u32,
        }

        impl $name {
            pub fn new(input: ID, output: ID, n: u32) -> Self {
                Self { input, output, n }
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
                vec![$forward_key, $backward_key]
            }

            fn create_pipelines(
                &self,
                device: &wgpu::Device,
            ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
                let make = |label, src: &'static str, pipeline_label| {
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
                    });
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(pipeline_label),
                        layout: None,
                        module: &shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                };

                vec![
                    (
                        $forward_key,
                        make("fwd_shader", $forward_shader, "fwd_pipeline"),
                    ),
                    (
                        $backward_key,
                        make("bwd_shader", $backward_shader, "bwd_pipeline"),
                    ),
                ]
            }

            fn buffers_needed(
                &self,
                shapes: &HashMap<ID, Vec<u32>>,
            ) -> Vec<(ID, &'static str, u32)> {
                let n: u32 = shapes[&self.input].iter().product();
                vec![
                    (self.input, buffer_kind::GRAD, n),
                    (self.output, buffer_kind::GRAD, n),
                ]
            }

            fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
                let pipeline = ctx.pipelines.get($forward_key).ok_or_else(|| {
                    VolticError::Internal(concat!($forward_key, " pipeline not found").into())
                })?;

                let in_buf = ctx.buffers.get(&self.input).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.input))
                })?;
                let out_buf = ctx.buffers.get(&self.output).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.output))
                })?;

                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(concat!($forward_key, "_bind_group")),
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
                    ],
                });

                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(concat!($forward_key, "_pass")),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.n.div_ceil(256), 1, 1);

                Ok(())
            }

            fn backward(&self, ctx: &mut GpuContext) -> Result<()> {
                let pipeline = ctx.pipelines.get($backward_key).ok_or_else(|| {
                    VolticError::Internal(concat!($backward_key, " pipeline not found").into())
                })?;

                // tanh/sigmoid use output, relu uses input
                let source_buf = ctx.buffers.get(&self.$backward_source).ok_or_else(|| {
                    VolticError::Internal(format!("buffer not found: {:?}", self.$backward_source))
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

                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(concat!($backward_key, "_bind_group")),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: source_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: grad_out_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: grad_in_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(concat!($backward_key, "_pass")),
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

impl_activation!(
    TanhOp,
    include_str!("shaders/tanh.wgsl"),
    include_str!("shaders/tanh_backward.wgsl"),
    "tanh",
    "tanh_backward",
    output // tanh backward uses forward output
);

impl_activation!(
    ReLUOp,
    include_str!("shaders/relu.wgsl"),
    include_str!("shaders/relu_backward.wgsl"),
    "relu",
    "relu_backward",
    input // relu backward uses original input
);

impl_activation!(
    SigmoidOp,
    include_str!("shaders/sigmoid.wgsl"),
    include_str!("shaders/sigmoid_backward.wgsl"),
    "sigmoid",
    "sigmoid_backward",
    output // sigmoid backward uses forward output
);

impl_activation!(
    GeluOp,
    include_str!("shaders/gelu.wgsl"),
    include_str!("shaders/gelu_backward.wgsl"),
    "gelu",
    "gelu_backward",
    input // gelu backward needs original input
);

impl_activation!(
    ExpOp,
    include_str!("shaders/exp.wgsl"),
    include_str!("shaders/exp_backward.wgsl"),
    "exp",
    "exp_backward",
    output // exp backward reuses output like tanh/sigmoid
);
