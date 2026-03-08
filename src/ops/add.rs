use crate::{GpuContext, ID, Result, VolticError};

const SHADER: &str = include_str!("shaders/add.wgsl");

#[derive(Debug, Clone)]
pub struct AddOp {
    inputs: [ID; 2],
    output: ID,
}

impl AddOp {
    pub fn new(lhs: ID, rhs: ID, output: ID) -> Self {
        Self {
            inputs: [lhs, rhs],
            output,
        }
    }
}

impl super::Op for AddOp {
    fn inputs(&self) -> &[ID] {
        &self.inputs
    }
    fn outputs(&self) -> &[ID] {
        std::slice::from_ref(&self.output)
    }

    fn infer_shape(&self, input_shapes: &[&Vec<u32>]) -> Result<Vec<u32>> {
        if input_shapes[0] != input_shapes[1] {
            return Err(VolticError::IncompatibleShapes {
                lhs: input_shapes[0].clone(),
                rhs: input_shapes[1].clone(),
                op: "add",
            });
        }
        Ok(input_shapes[0].clone())
    }

    fn pipeline_keys(&self) -> Vec<&'static str> {
        vec!["add"]
    }

    fn create_pipelines(
        &self,
        device: &wgpu::Device,
    ) -> Vec<(&'static str, wgpu::ComputePipeline)> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("add_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("add_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        vec![("add", pipeline)]
    }

    fn forward_gpu(&self, ctx: &mut GpuContext) -> Result<()> {
        let pipeline = ctx
            .pipelines
            .get("add")
            .ok_or_else(|| VolticError::Internal("add pipeline not found".into()))?;

        let lhs_buf = ctx.buffers.get(&self.inputs[0]).ok_or_else(|| {
            VolticError::Internal(format!("buffer not found: {:?}", self.inputs[0]))
        })?;
        let rhs_buf = ctx.buffers.get(&self.inputs[1]).ok_or_else(|| {
            VolticError::Internal(format!("buffer not found: {:?}", self.inputs[1]))
        })?;
        let out_buf = ctx
            .buffers
            .get(&self.output)
            .ok_or_else(|| VolticError::Internal(format!("buffer not found: {:?}", self.output)))?;

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let n_elements = (out_buf.size() / 4) as u32;
        let workgroups = n_elements.div_ceil(256);

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("add_pass"),
                timestamp_writes: None,
            });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }
}
