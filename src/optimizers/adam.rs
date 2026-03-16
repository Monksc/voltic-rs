use crate::{buffer_kind, context::Context, Result, VolticError, ID, VARIABLES_ID_COUNTER};
use std::{borrow::Cow, collections::HashMap, sync::atomic::Ordering};

const SHADER: &str = include_str!("../ops/shaders/adam.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamDims {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: u32,
    n: u32,
}

pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: u32, // step counter for bias correction
    cached_ids: Vec<(ID, u32)>,
    cache_generation: u64,
    pipeline: Option<wgpu::ComputePipeline>,
    momentum: HashMap<ID, wgpu::Buffer>,
    variance: HashMap<ID, wgpu::Buffer>,
    dims_buffer: Option<wgpu::Buffer>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self::with_params(lr, 0.9, 0.999, 1e-8)
    }

    pub fn with_params(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            t: 0,
            cached_ids: Vec::new(),
            cache_generation: u64::MAX,
            pipeline: None,
            momentum: Default::default(),
            variance: Default::default(),
            dims_buffer: None,
        }
    }
    pub fn init(&mut self) -> Result<()> {
        let ctx = Context::get();
        let gpu = ctx
            .gpu_context()
            .as_ref()
            .ok_or(VolticError::GpuNotAvailable)?;

        let mut seen = std::collections::HashSet::new();
        for op in ctx.operations() {
            for id in op.inputs() {
                if !seen.insert(*id) {
                    continue;
                }
                let n = match ctx.shape_total_with_context(*id) {
                    Some(n) => n,
                    None => continue,
                };
                if !gpu.training_buffers.contains_key(&(*id, buffer_kind::GRAD)) {
                    continue;
                }

                self.momentum.insert(
                    *id,
                    gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("adam_momentum:{:?}", id)),
                        size: (n * 4) as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
                self.variance.insert(
                    *id,
                    gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("adam_variance:{:?}", id)),
                        size: (n * 4) as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
            }
        }

        Ok(())
    }

    pub fn update_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn invalidate(&mut self) {
        self.cached_ids.clear();
        self.cache_generation = u64::MAX;
    }

    fn ensure_pipeline(&mut self, device: &wgpu::Device) {
        if self.pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adam_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });

        self.pipeline = Some(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("adam_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
        );
    }

    fn build_cache(&mut self) {
        let ctx = Context::get();

        let mut seen = std::collections::HashSet::new();
        self.cached_ids = ctx
            .operations()
            .iter()
            .flat_map(|op| op.inputs())
            .filter_map(|id| {
                if !seen.insert(*id) {
                    return None;
                }
                let has_grad = ctx
                    .gpu_context()
                    .as_ref()
                    .and_then(|gpu| gpu.training_buffers.get(&(*id, buffer_kind::GRAD)))
                    .is_some();

                if has_grad {
                    let n = ctx.shape_total_with_context(*id).unwrap_or(0);
                    Some((*id, n))
                } else {
                    None
                }
            })
            .collect();

        self.cache_generation = VARIABLES_ID_COUNTER.load(Ordering::Relaxed);
    }

    pub fn step(&mut self) -> Result<()> {
        let generation = VARIABLES_ID_COUNTER.load(Ordering::Relaxed);
        if self.cached_ids.is_empty() || self.cache_generation != generation {
            self.build_cache();
        }

        self.t += 1;

        let mut ctx = Context::get_mut();
        let gpu = ctx
            .gpu_context_mut()
            .as_mut()
            .ok_or(VolticError::GpuNotAvailable)?;

        self.ensure_pipeline(&gpu.device);
        let pipeline = self.pipeline.as_ref().unwrap();

        for &(id, n) in &self.cached_ids {
            let weight_buf = gpu
                .buffers
                .get(&id)
                .ok_or_else(|| VolticError::Internal(format!("no weight buffer: {:?}", id)))?;
            let grad_buf = gpu
                .training_buffers
                .get(&(id, buffer_kind::GRAD))
                .ok_or_else(|| VolticError::Internal(format!("no grad buffer: {:?}", id)))?;

            // Create momentum and variance buffers on-demand if they don't exist
            if !self.momentum.contains_key(&id) {
                let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("adam_momentum:{:?}", id)),
                    size: (n * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.momentum.insert(id, buf);
            }
            if !self.variance.contains_key(&id) {
                let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("adam_variance:{:?}", id)),
                    size: (n * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.variance.insert(id, buf);
            }

            let momentum_buf = self
                .momentum
                .get(&id)
                .ok_or_else(|| VolticError::Internal(format!("no momentum buffer: {:?}", id)))?;
            let variance_buf = self
                .variance
                .get(&id)
                .ok_or_else(|| VolticError::Internal(format!("no variance buffer: {:?}", id)))?;

            let dims = AdamDims {
                lr: self.lr,
                beta1: self.beta1,
                beta2: self.beta2,
                epsilon: self.epsilon,
                t: self.t,
                n,
            };

            // Reuse or create dims buffer
            if self.dims_buffer.is_none()
                || self.dims_buffer.as_ref().map(|b| b.size()).unwrap_or(0)
                    < std::mem::size_of::<AdamDims>() as u64
            {
                self.dims_buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("adam_dims"),
                    size: std::mem::size_of::<AdamDims>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let dims_buf = self.dims_buffer.as_ref().unwrap();
            gpu.queue
                .write_buffer(dims_buf, 0, bytemuck::bytes_of(&dims));

            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("adam_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grad_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: momentum_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: variance_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dims_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = gpu
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("adam_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }

        gpu.flush();
        Ok(())
    }
}
