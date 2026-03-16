macro_rules! make_pipeline {
    ($device:ident, $label:literal, $shader:expr) => {{
        let shader = $device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some($label),
            source: wgpu::ShaderSource::Wgsl($shader.into()),
        });
        $device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some($label),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }};
}

macro_rules! create_dims_buffer {
    ($ctx:ident, $label:literal, $dims:expr) => {{
        $ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some($label),
                contents: bytemuck::bytes_of(&$dims),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }};
}
