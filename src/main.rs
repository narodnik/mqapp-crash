use log::debug;
use miniquad::{
    conf, native::gl, window, Backend, Bindings, BlendFactor, BlendState, BlendValue, BufferLayout,
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
    buff: Option<u32>,
    vert: Option<u32>,
    fb: gl::GLuint,
    shader_program: gl::GLuint,

    texture: gl::GLuint,

    tex_loc: i32,
    proj_loc: i32,
    model_loc: i32,
}

impl Stage {
    pub fn new() -> Self {
        let mut ctx: Box<dyn RenderingBackend> = window::new_rendering_backend();

        let mut texture: gl::GLuint = 0;
        unsafe {
            gl::glGenTextures(1, &mut texture as *mut _);
            gl::glActiveTexture(gl::GL_TEXTURE0);
            gl::glBindTexture(gl::GL_TEXTURE_2D, texture);
            gl::glPixelStorei(gl::GL_UNPACK_ALIGNMENT, 1); // miniquad always uses row alignment of 1

            // do we need this?
            gl::glTexParameteri(
                gl::GL_TEXTURE_2D,
                gl::GL_TEXTURE_SWIZZLE_A,
                gl::GL_ALPHA as _,
            );

            gl::glTexImage2D(
                gl::GL_TEXTURE_2D,
                0,
                gl::GL_RGBA as i32,
                1i32,
                1i32,
                0,
                gl::GL_RGBA,
                gl::GL_UNSIGNED_BYTE,
                (&[255, 255, 255, 255]).as_ptr() as *const _,
            );

            gl::glTexParameteri(
                gl::GL_TEXTURE_2D,
                gl::GL_TEXTURE_WRAP_S,
                gl::GL_CLAMP_TO_EDGE as i32,
            );
            gl::glTexParameteri(
                gl::GL_TEXTURE_2D,
                gl::GL_TEXTURE_WRAP_T,
                gl::GL_CLAMP_TO_EDGE as i32,
            );
            gl::glTexParameteri(
                gl::GL_TEXTURE_2D,
                gl::GL_TEXTURE_MIN_FILTER,
                gl::GL_LINEAR as i32,
            );
            gl::glTexParameteri(
                gl::GL_TEXTURE_2D,
                gl::GL_TEXTURE_MAG_FILTER,
                gl::GL_LINEAR as i32,
            );
        }

        let vertex_shader = unsafe {
            let shader = gl::glCreateShader(gl::GL_VERTEX_SHADER);
            assert!(shader != 0);
            let cstring = std::ffi::CString::new(GL_VERTEX).unwrap();
            let csource = [cstring];
            gl::glShaderSource(shader, 1, csource.as_ptr() as *const _, std::ptr::null());
            gl::glCompileShader(shader);

            let mut is_compiled = 0;
            gl::glGetShaderiv(shader, gl::GL_COMPILE_STATUS, &mut is_compiled as *mut _);
            assert!(is_compiled != 0);
            shader
        };
        let fragment_shader = unsafe {
            let shader = gl::glCreateShader(gl::GL_FRAGMENT_SHADER);
            assert!(shader != 0);
            let cstring = std::ffi::CString::new(GL_FRAGMENT).unwrap();
            let csource = [cstring];
            gl::glShaderSource(shader, 1, csource.as_ptr() as *const _, std::ptr::null());
            gl::glCompileShader(shader);

            let mut is_compiled = 0;
            gl::glGetShaderiv(shader, gl::GL_COMPILE_STATUS, &mut is_compiled as *mut _);
            assert!(is_compiled != 0);
            shader
        };

        let shader_program = unsafe {
            let program = gl::glCreateProgram();
            gl::glAttachShader(program, vertex_shader);
            gl::glAttachShader(program, fragment_shader);
            gl::glLinkProgram(program);

            // delete no longer used shaders
            gl::glDetachShader(program, vertex_shader);
            gl::glDeleteShader(vertex_shader);
            gl::glDeleteShader(fragment_shader);

            let mut link_status = 0;
            gl::glGetProgramiv(program, gl::GL_LINK_STATUS, &mut link_status as *mut _);
            assert!(link_status != 0);
            gl::glUseProgram(program);
            program
        };

        let cname = std::ffi::CString::new("tex").unwrap();
        let tex_loc = unsafe { gl::glGetUniformLocation(shader_program, cname.as_ptr()) };
        assert!(tex_loc != -1);
        let cname = std::ffi::CString::new("Projection").unwrap();
        let proj_loc = unsafe { gl::glGetUniformLocation(shader_program, cname.as_ptr()) };
        assert!(proj_loc != -1);
        let cname = std::ffi::CString::new("Model").unwrap();
        let model_loc = unsafe { gl::glGetUniformLocation(shader_program, cname.as_ptr()) };
        assert!(model_loc != -1);

        unsafe {
            let cname = std::ffi::CString::new("in_pos").unwrap();
            let loc = gl::glGetAttribLocation(shader_program, cname.as_ptr() as *const _);
            assert!(loc != -1);
            let cname = std::ffi::CString::new("in_color").unwrap();
            let loc = gl::glGetAttribLocation(shader_program, cname.as_ptr() as *const _);
            assert!(loc != -1);
            let cname = std::ffi::CString::new("in_uv").unwrap();
            let loc = gl::glGetAttribLocation(shader_program, cname.as_ptr() as *const _);
            assert!(loc != -1);
        };

        let mut fb: gl::GLuint = 0;
        unsafe {
            gl::glGetIntegerv(gl::GL_FRAMEBUFFER_BINDING, &mut fb as *mut _ as *mut _);
        }

        Stage {
            ctx,
            buff: None,
            vert: None,
            fb,
            shader_program,
            texture,
            tex_loc,
            proj_loc,
            model_loc,
        }
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {
        let elapsed = START.elapsed().as_secs();

        if self.vert.is_none() && elapsed >= 4 {
            debug!("LFG! vert");
            let verts = [
                Vertex {
                    pos: [0.0, 0.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    pos: [100.0, 0.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    pos: [0.0, 100.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
            ];

            assert_eq!(std::mem::size_of_val(&verts), 96);
            assert_eq!(std::mem::size_of::<Vertex>(), 32);

            let mut gl_buf: u32 = 0;
            unsafe {
                gl::glGenBuffers(1, &mut gl_buf as *mut _);
                gl::glBindBuffer(gl::GL_ARRAY_BUFFER, gl_buf);
                gl::glBufferData(
                    gl::GL_ARRAY_BUFFER,
                    96,
                    std::ptr::null() as *const _,
                    gl::GL_STATIC_DRAW,
                );
                gl::glBufferSubData(gl::GL_ARRAY_BUFFER, 0, 96, (&verts).as_ptr() as _);
            }
            self.vert = Some(gl_buf);
        }

        if self.buff.is_none() {
            debug!("LFG! idx");
            let idx: [u16; 3] = [0, 2, 1];

            assert_eq!(std::mem::size_of_val(&idx), 6);

            let mut gl_buf: u32 = 0;
            unsafe {
                gl::glGenBuffers(1, &mut gl_buf as *mut _);
                gl::glBindBuffer(gl::GL_ELEMENT_ARRAY_BUFFER, gl_buf);
                gl::glBufferData(
                    gl::GL_ELEMENT_ARRAY_BUFFER,
                    6,
                    std::ptr::null() as *const _,
                    gl::GL_STATIC_DRAW,
                );
                gl::glBufferSubData(gl::GL_ELEMENT_ARRAY_BUFFER, 0, 6, (&idx).as_ptr() as _);
            }
            self.buff = Some(gl_buf);
        }
    }

    fn draw(&mut self) {
        //self.ctx.begin_default_pass(PassAction::clear_color(0., 0., 0., 1.));
        let (screen_w, screen_h) = miniquad::window::screen_size();
        let framebuffer = self.fb;
        unsafe {
            gl::glBindFramebuffer(gl::GL_FRAMEBUFFER, framebuffer);
            gl::glViewport(0, 0, screen_w as i32, screen_h as i32);
            gl::glScissor(0, 0, screen_w as i32, screen_h as i32);

            gl::glClearColor(0., 0., 0., 1.);
            gl::glClearDepthf(1.);
            gl::glClear(gl::GL_COLOR_BUFFER_BIT);
        }

        //self.ctx.apply_pipeline(&self.pipeline);
        unsafe {
            gl::glUseProgram(self.shader_program);
            gl::glEnable(gl::GL_SCISSOR_TEST);
            gl::glDisable(gl::GL_DEPTH_TEST);
            gl::glFrontFace(gl::GL_CCW);
            gl::glDisable(gl::GL_CULL_FACE);
            gl::glDisable(gl::GL_BLEND);
            gl::glDisable(gl::GL_STENCIL_TEST);
            gl::glColorMask(true as _, true as _, true as _, true as _);
        }

        let (screen_w, screen_h) = miniquad::window::screen_size();
        // This will make the top left (0, 0) and the bottom right (1, 1)
        // Default is (-1, 1) -> (1, -1)
        let proj = glam::Mat4::from_translation(glam::Vec3::new(-1., 1., 0.))
            * glam::Mat4::from_scale(glam::Vec3::new(2., -2., 1.));
        let model = glam::Mat4::from_translation(glam::Vec3::new(0., 0., 0.))
            * glam::Mat4::from_scale(glam::Vec3::new(1. / screen_w, 1. / screen_h, 1.));

        let mut uniforms_data = [0u8; 128];
        let data: [u8; 64] = unsafe { std::mem::transmute_copy(&proj) };
        uniforms_data[0..64].copy_from_slice(&data);
        let data: [u8; 64] = unsafe { std::mem::transmute_copy(&model) };
        uniforms_data[64..].copy_from_slice(&data);
        assert_eq!(128, 2 * UniformType::Mat4.size());
        //self.ctx.apply_uniforms_from_bytes(uniforms_data.as_ptr(), uniforms_data.len());
        let uniform_ptr = uniforms_data.as_ptr();
        unsafe {
            let data = (uniform_ptr as *const f32).add(0);
            gl::glUniformMatrix4fv(self.proj_loc, 1, 0, data);
            let data = (uniform_ptr as *const f32).add(16);
            gl::glUniformMatrix4fv(self.model_loc, 1, 0, data);
        }

        if let Some(buff) = self.buff {
            if let Some(vert) = self.vert {
                // apply_bindings
                unsafe {
                    gl::glActiveTexture(gl::GL_TEXTURE0);
                    gl::glBindTexture(gl::GL_TEXTURE_2D, self.texture);
                    // shader image loc
                    //gl::glUniform1i(self.loc_tex, 0i32);

                    gl::glBindBuffer(gl::GL_ELEMENT_ARRAY_BUFFER, buff);
                    gl::glBindBuffer(gl::GL_ARRAY_BUFFER, vert);

                    // pos
                    gl::glVertexAttribPointer(
                        0,
                        2,
                        gl::GL_FLOAT,
                        gl::GL_FALSE as u8,
                        32,
                        0 as *mut _,
                    );
                    gl::glEnableVertexAttribArray(0);
                    // color
                    gl::glVertexAttribPointer(
                        1,
                        4,
                        gl::GL_FLOAT,
                        gl::GL_FALSE as u8,
                        32,
                        8 as *mut _,
                    );
                    gl::glEnableVertexAttribArray(1);
                    // uv
                    gl::glVertexAttribPointer(
                        2,
                        2,
                        gl::GL_FLOAT,
                        gl::GL_FALSE as u8,
                        32,
                        24 as *mut _,
                    );
                    gl::glEnableVertexAttribArray(2);
                }

                // ctx.draw()
                unsafe {
                    gl::glDrawElementsInstanced(
                        gl::GL_TRIANGLES,
                        3,
                        gl::GL_UNSIGNED_SHORT,
                        std::ptr::null_mut(),
                        1,
                    );
                }
            }
        }

        //self.ctx.commit_frame();
        unsafe {
            gl::glBindBuffer(gl::GL_ARRAY_BUFFER, 0);
            gl::glBindBuffer(gl::GL_ELEMENT_ARRAY_BUFFER, 0);
        }
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
            let cfg = AndroidConfig::default()
                .with_max_level(level)
                .with_tag("mqapp");
            Box::new(Self {
                logger: AndroidLogger::new(cfg),
                level,
                config,
            })
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
