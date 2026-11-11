pub mod vec3simd;

use std::time::SystemTime;

use glam::{Vec3, vec3};
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};
use wide::{f32x16, i32x16, u32x16};

use crate::vec3simd::Vec3Simd;

const WIDTH: usize = 320;
const HEIGHT: usize = 180;

fn hit_sphere(
    center: Vec3Simd,
    radius: f32x16,
    ray_origin: Vec3Simd,
    ray_direction: Vec3Simd,
) -> f32x16 {
    let oc = center - ray_origin;

    let a = ray_direction.length_squared();
    let h = ray_direction.dot(oc);
    let c = oc.length_squared() - radius * radius;
    let discriminant = h * h - a * c;

    discriminant
        .is_sign_negative()
        .bitselect(f32x16::splat(-1.0), (h - discriminant.sqrt()) / a)
}

#[allow(unused)]
fn hit_cube(
    min: Vec3Simd,
    max: Vec3Simd,
    ray_origin: Vec3Simd,
    inv_ray_direction: Vec3Simd,
) -> (f32x16, f32x16) {
    let mut tmin = f32x16::splat(0.0);
    let mut tmax = f32x16::splat(f32::MAX);

    for i in 0..3 {
        let ray_origin_componenent = ray_origin.component(i);
        let dir_inv_component = inv_ray_direction.component(i);
        let t1 = (min.component(i) - ray_origin_componenent) * dir_inv_component;
        let t2 = (max.component(i) - ray_origin_componenent) * dir_inv_component;

        tmin = tmin.max(t1.min(t2)); // max(tmin, min(t1, t2)));
        tmax = tmax.min(t1.max(t2)); // min(tmax, max(t1, t2)));
    }

    (tmin, tmax)
}

fn hit_cube_with_normal(
    min: Vec3Simd,
    max: Vec3Simd,
    ray_origin: Vec3Simd,
    inv_ray_direction: Vec3Simd,
) -> (f32x16, f32x16, i32x16) {
    // Use NEG_INFINITY so the first valid t_near always wins an axis
    let mut tmin = f32x16::splat(f32::NEG_INFINITY);
    let mut tmax = f32x16::splat(f32::MAX);
    let mut hit_axis = i32x16::splat(0);

    for i in 0..3 {
        let ray_origin_componenent = ray_origin.component(i);
        let dir_inv_component = inv_ray_direction.component(i);
        let t1 = (min.component(i) - ray_origin_componenent) * dir_inv_component;
        let t2 = (max.component(i) - ray_origin_componenent) * dir_inv_component;

        let t_near = t1.min(t2);
        let t_far = t1.max(t2);

        // Identify lanes where this axis becomes the new latest entry
        let is_new_max = t_near.simd_gt(tmin);

        tmin = is_new_max.select(t_near, tmin);
        hit_axis = is_new_max.select(i32x16::splat(i as i32), hit_axis);
        tmax = tmax.min(t_far);
    }

    // Clamp tmin to 0.0 after axis tracking so forward-hit checks stay identical
    tmin = tmin.max(f32x16::splat(0.0));

    (tmin, tmax, hit_axis)
}

#[derive(Debug)]
struct Camera {
    origin: Vec3,
    target: Vec3,
    fov_y: f32,
    aspect_ratio: f32,
    #[allow(unused)]
    forward: Vec3,
    #[allow(unused)]
    right: Vec3,
    #[allow(unused)]
    up: Vec3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    pixel00_loc: Vec3,
}

impl Camera {
    fn new(position: Vec3, target: Vec3, fov_y: f32, aspect_ratio: f32) -> Self {
        let mut camera = Camera {
            origin: position,
            target,
            fov_y,
            aspect_ratio,
            forward: Vec3::ZERO,
            right: Vec3::ZERO,
            up: Vec3::ZERO,
            pixel_delta_u: Vec3::ZERO,
            pixel_delta_v: Vec3::ZERO,
            pixel00_loc: Vec3::ZERO,
        };

        camera.update();

        camera
    }

    fn update(&mut self) {
        self.forward = (self.target - self.origin).normalize();
        self.right = self.forward.cross(Vec3::Y).normalize();
        self.up = self.right.cross(self.forward).normalize();

        let viewport_height = 2.0 * (self.fov_y * 0.5).tan();
        let viewport_width = viewport_height * self.aspect_ratio;
        let focal_length = 1.0;

        let viewport_u = self.right * viewport_width;
        let viewport_v = self.up * -viewport_height;

        self.pixel_delta_u = viewport_u / WIDTH as f32;
        self.pixel_delta_v = viewport_v / HEIGHT as f32;

        let viewport_upper_left =
            self.origin + self.forward * focal_length - viewport_u * 0.5 - viewport_v * 0.5;
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);
    }
}

fn main() {
    let mut buffer = vec![0u32; WIDTH * HEIGHT];

    let mut window = Window::new(
        "Toy Voxel Engine",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            scale_mode: ScaleMode::Stretch,
            scale: Scale::X4,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to create the window");

    window.set_target_fps(60);

    let mut time_since_last_fps = SystemTime::now();

    let mut camera = Camera::new(
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, 0.0),
        90.0_f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
    );

    let background_color = Vec3Simd::from(vec3(0.5, 0.7, 1.0));

    let sphere_center = Vec3Simd::from(vec3(-0.5, 0.0, 0.0));
    let sphere_radius = f32x16::splat(0.5);

    let cube_center = Vec3Simd::from(vec3(0.5, 0.0, 0.0));
    let cube_min = cube_center + Vec3Simd::from(vec3(-0.33, -0.33, -0.33));
    let cube_max = cube_center + Vec3Simd::from(vec3(0.33, 0.33, 0.33));

    let local_x = i32x16::new([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    let local_y = i32x16::new([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut rot_y = 0.0f32;
        let mut fwd = 0.0f32;
        let mut rot_x = 0.0f32;

        window.get_keys().iter().for_each(|key| match key {
            Key::W => fwd += 0.01,
            Key::S => fwd -= 0.01,
            Key::A => rot_y -= 0.1,
            Key::D => rot_y += 0.1,
            Key::Space => rot_x -= 0.1,
            Key::LeftShift => rot_x += 0.1,
            _ => (),
        });

        // Move camera along its forward axis
        camera.origin += camera.forward * fwd;

        // Rotate camera around target
        let mut pos = camera.origin - camera.target;

        pos = pos.rotate_y(rot_y);

        let dir = pos.normalize();
        let current_pitch = dir.y.clamp(-1.0, 1.0).asin();
        let max_pitch = std::f32::consts::FRAC_PI_2 - 30.0_f32.to_radians();
        let min_pitch = -max_pitch;
        let target_pitch = (current_pitch - rot_x).clamp(min_pitch, max_pitch);
        let clamped_rot_x = current_pitch - target_pitch;
        pos = pos.rotate_axis(camera.right, clamped_rot_x);

        camera.origin = pos + camera.target;

        camera.update();

        // Process image in blocks of 4x4 pixels
        for chunk_y in 0..(HEIGHT / 4) {
            for chunk_x in 0..(WIDTH / 4) {
                let x = i32x16::splat((chunk_x as i32) * 4) + local_x;
                let y = i32x16::splat((chunk_y as i32) * 4) + local_y;

                let x_f32 = f32x16::from_i32x16(x);
                let y_f32 = f32x16::from_i32x16(y);

                let pixel_center = Vec3Simd::from(camera.pixel00_loc)
                    + x_f32 * Vec3Simd::from(camera.pixel_delta_u)
                    + y_f32 * Vec3Simd::from(camera.pixel_delta_v);

                let ray_direction = (pixel_center - Vec3Simd::from(camera.origin)).normalize();
                let inv_ray_direction = f32x16::splat(1.0) / ray_direction;

                let grad = 0.5 * ray_direction.y + 1.0;

                let mut color = (1.0 - grad) * 1.0 + grad * background_color;

                let mut closest_t = f32x16::splat(f32::MAX);

                // Sphere
                let t = hit_sphere(
                    sphere_center,
                    sphere_radius,
                    Vec3Simd::from(camera.origin),
                    ray_direction,
                );
                let sphere_hit_normal =
                    (Vec3Simd::from(camera.origin) + ray_direction * t - sphere_center).normalize();
                color = Vec3Simd::bitselect_scalar(
                    t.is_sign_positive(),
                    f32x16::splat(0.5) * (sphere_hit_normal + f32x16::splat(1.0)),
                    color,
                );

                closest_t = t.is_sign_positive().bitselect(t, closest_t);

                // Cube
                let (tmin, tmax, hit_axis) = hit_cube_with_normal(
                    cube_min,
                    cube_max,
                    Vec3Simd::from(camera.origin),
                    inv_ray_direction,
                );
                let cube_hit_normal = Vec3Simd::new(
                    -ray_direction.x.signum() * hit_axis.simd_eq(i32x16::splat(0)).round_float(),
                    -ray_direction.y.signum() * hit_axis.simd_eq(i32x16::splat(1)).round_float(),
                    -ray_direction.z.signum() * hit_axis.simd_eq(i32x16::splat(2)).round_float(),
                );
                color = Vec3Simd::bitselect_scalar(
                    tmin.simd_lt(closest_t) & tmin.simd_le(tmax) & tmax.is_sign_positive(),
                    f32x16::splat(0.5) * (cube_hit_normal + f32x16::splat(1.0)),
                    color,
                );

                color.clamp_scalar(f32x16::splat(0.0), f32x16::splat(1.0));

                let color_r_int: u32x16 = bytemuck::cast((color.x * 255.0f32).fast_round_int());
                let color_g_int: u32x16 = bytemuck::cast((color.y * 255.0f32).fast_round_int());
                let color_b_int: u32x16 = bytemuck::cast((color.z * 255.0f32).fast_round_int());

                // Encode rgb color to single u32 (0RGB format)
                let color: u32x16 = (color_r_int << 16) | (color_g_int << 8) | color_b_int;

                let color_array: [u32; 16] = color.into();

                for i in 0..4 {
                    let dst_index = (chunk_y * 4 + i) * WIDTH + chunk_x * 4;
                    let src_index = i * 4;
                    buffer[dst_index..(dst_index + 4)]
                        .copy_from_slice(&color_array[src_index..(src_index + 4)]);
                }
            }
        }

        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();

        if time_since_last_fps.elapsed().unwrap().as_millis() >= 1000 {
            println!("fps: {}", window.get_fps());
            time_since_last_fps = SystemTime::now();
        }
    }
}
