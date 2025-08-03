use log::debug;
use miniquad::{
    conf, window, Backend, Bindings, BlendFactor, BlendState, BlendValue, BufferLayout,
    BufferSource, BufferType, BufferUsage, Equation, EventHandler, KeyCode, KeyMods, MouseButton,
    PassAction, Pipeline, PipelineParams, RenderingBackend, ShaderMeta, ShaderSource, TouchPhase,
    UniformBlockLayout, UniformDesc, UniformType, VertexAttribute, VertexFormat,
};

use lazy_static::lazy_static;

use log::{LevelFilter, Log, Metadata, Record};
use simplelog::{CombinedLogger, Config, ConfigBuilder, SharedLogger};

lazy_static! {
    static ref START: std::time::Instant = std::time::Instant::now();
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

struct Stage {
    ctx: Box<dyn RenderingBackend>,
    pipeline: Pipeline,
    white_texture: miniquad::TextureId,

    buff: Option<miniquad::BufferId>,
    vert: Option<miniquad::BufferId>,
}

impl Stage {
    pub fn new() -> Self {
        let mut ctx: Box<dyn RenderingBackend> = window::new_rendering_backend();

        let white_texture = ctx.new_texture_from_rgba8(1, 1, &[255, 255, 255, 255]);

        let mut shader_meta: ShaderMeta = shader_meta();
        shader_meta.uniforms.uniforms.push(UniformDesc::new("Projection", UniformType::Mat4));
        shader_meta.uniforms.uniforms.push(UniformDesc::new("Model", UniformType::Mat4));

        let shader = ctx
            .new_shader(
                match ctx.info().backend {
                    Backend::OpenGl => {
                        ShaderSource::Glsl { vertex: GL_VERTEX, fragment: GL_FRAGMENT }
                    }
                    Backend::Metal => ShaderSource::Msl { program: METAL },
                },
                shader_meta,
            )
            .unwrap();

        let params = PipelineParams {
            color_blend: Some(BlendState::new(
                Equation::Add,
                BlendFactor::Value(BlendValue::SourceAlpha),
                BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
            )),
            ..Default::default()
        };

        let pipeline = ctx.new_pipeline(
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("in_pos", VertexFormat::Float2),
                VertexAttribute::new("in_color", VertexFormat::Float4),
                VertexAttribute::new("in_uv", VertexFormat::Float2),
            ],
            shader,
            params,
        );

        Stage { ctx, pipeline, white_texture, buff: None, vert: None }
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {
        let elapsed = START.elapsed().as_secs();

        // it doesnt crash when we swap the logic.
        // crash only happens when creating vert buffer is delayed

        if self.vert.is_none() && elapsed >= 4 {
        //if self.vert.is_none() {
            debug!("LFG! vert");
            let verts = [
                Vertex { pos: [0.0, 0.0], color: [1.0, 0.0, 0.0, 1.0], uv: [0.0, 0.0] },
                Vertex { pos: [100.0, 0.0], color: [1.0, 0.0, 0.0, 1.0], uv: [0.0, 0.0] },
                Vertex { pos: [0.0, 100.0], color: [1.0, 0.0, 0.0, 1.0], uv: [0.0, 0.0] },
            ];
            self.vert = Some(self.ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&verts),
            ));
        }

        //if self.buff.is_none() && elapsed >= 4 {
        if self.buff.is_none() {
            debug!("LFG! idx");
            let idx = [0, 2, 1];
            self.buff = Some(self.ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&idx),
            ));
        }
    }

    fn draw(&mut self) {
        self.ctx.begin_default_pass(PassAction::clear_color(0., 0., 0., 1.));
        self.ctx.apply_pipeline(&self.pipeline);

        let (screen_w, screen_h) = miniquad::window::screen_size();
        // This will make the top left (0, 0) and the bottom right (1, 1)
        // Default is (-1, 1) -> (1, -1)
        let proj = glam::Mat4::from_translation(glam::Vec3::new(-1., 1., 0.)) *
            glam::Mat4::from_scale(glam::Vec3::new(2., -2., 1.));
        let model = glam::Mat4::from_translation(glam::Vec3::new(0., 0., 0.)) *
            glam::Mat4::from_scale(glam::Vec3::new(1. / screen_w, 1. / screen_h, 1.));

        let mut uniforms_data = [0u8; 128];
        let data: [u8; 64] = unsafe { std::mem::transmute_copy(&proj) };
        uniforms_data[0..64].copy_from_slice(&data);
        let data: [u8; 64] = unsafe { std::mem::transmute_copy(&model) };
        uniforms_data[64..].copy_from_slice(&data);
        assert_eq!(128, 2 * UniformType::Mat4.size());
        self.ctx.apply_uniforms_from_bytes(uniforms_data.as_ptr(), uniforms_data.len());

        if let Some(buff) = self.buff {
            if let Some(vert) = self.vert {
                let bindings = Bindings {
                    vertex_buffers: vec![vert],
                    index_buffer: buff,
                    images: vec![self.white_texture],
                };
                self.ctx.apply_bindings(&bindings);
                self.ctx.draw(0, 3, 1);
            }
        }

        self.ctx.commit_frame();
    }
}

pub fn run_gui() {
    let mut conf = miniquad::conf::Conf {
        window_title: "MqApp".to_string(),
        high_dpi: true,
        ..Default::default()
    };
    miniquad::start(conf, || Box::new(Stage::new()));
}

#[cfg(target_os = "android")]
mod android_logwrap {
    use super::*;
    use android_logger::{AndroidLogger, Config as AndroidConfig};

    /// Implements a wrapper around the android logger so it's compatible with simplelog.
    pub struct AndroidLoggerWrapper {
        logger: AndroidLogger,
        level: LevelFilter,
        config: Config,
    }

    impl AndroidLoggerWrapper {
        pub fn new(level: LevelFilter, config: Config) -> Box<Self> {
            let cfg = AndroidConfig::default().with_max_level(level).with_tag("mqapp");
            Box::new(Self { logger: AndroidLogger::new(cfg), level, config })
        }
    }

    impl Log for AndroidLoggerWrapper {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            self.logger.enabled(metadata)
        }

        fn log(&self, record: &Record<'_>) {
            if self.enabled(record.metadata()) {
                self.logger.log(record)
            }
        }

        fn flush(&self) {}
    }

    impl SharedLogger for AndroidLoggerWrapper {
        fn level(&self) -> LevelFilter {
            self.level
        }

        fn config(&self) -> Option<&Config> {
            Some(&self.config)
        }

        fn as_log(self: Box<Self>) -> Box<dyn Log> {
            Box::new(*self)
        }
    }
}

fn main() {
    // Setup logging
    let mut loggers: Vec<Box<dyn SharedLogger>> = vec![];
    let mut cfg = ConfigBuilder::new();
    let cfg = cfg.build();
    let android_logger = android_logwrap::AndroidLoggerWrapper::new(LevelFilter::Trace, cfg);
    loggers.push(android_logger);
    CombinedLogger::init(loggers).expect("logger");

    run_gui();
    debug!(target: "main", "Started GFX backend");
}

pub const GL_VERTEX: &str = r#"#version 100
attribute vec2 in_pos;
attribute vec4 in_color;
attribute vec2 in_uv;

varying lowp vec4 color;
varying lowp vec2 uv;

uniform mat4 Projection;
uniform mat4 Model;

void main() {
    gl_Position = Projection * Model * vec4(in_pos, 0, 1);
    color = in_color;
    uv = in_uv;
}"#;

pub const GL_FRAGMENT: &str = r#"#version 100
varying lowp vec4 color;
varying lowp vec2 uv;

uniform sampler2D tex;

void main() {
    gl_FragColor = color * texture2D(tex, uv);
}"#;

pub const METAL: &str = r#"
#include <metal_stdlib>

using namespace metal;

struct Uniforms
{
    float4x4 Projection;
    float4x4 Model;
};

struct Vertex
{
    float2 in_pos   [[attribute(0)]];
    float4 in_color [[attribute(1)]];
    float2 in_uv    [[attribute(2)]];
};

struct RasterizerData
{
    float4 position [[position]];
    float4 color [[user(locn0)]];
    float2 uv [[user(locn1)]];
};

vertex RasterizerData vertexShader(Vertex v [[stage_in]])
{
    RasterizerData out;

    out.position = uniforms.Model * uniforms.Projection * float4(v.in_pos.xy, 0.0, 1.0);
    out.color = v.in_color;
    out.uv = v.texcoord;

    return out
}

fragment float4 fragmentShader(RasterizerData in [[stage_in]], texture2d<float> tex [[texture(0)]], sampler texSmplr [[sampler(0)]])
{
    return in.color * tex.sample(texSmplr, in.uv)
}

"#;

fn shader_meta() -> ShaderMeta {
    ShaderMeta {
        images: vec!["tex".to_string()],
        uniforms: UniformBlockLayout { uniforms: vec![] },
    }
}
