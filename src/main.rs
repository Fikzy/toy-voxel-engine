pub mod vec3simd;

use std::time::SystemTime;

use glam::{IVec3, Vec3, vec3};
use minifb::{Key, KeyRepeat, Scale, ScaleMode, Window, WindowOptions};
use wide::{f32x16, i32x16, u32x16};

#[cfg(all(target_arch = "x86_64"))]
use core::arch::x86_64::*;

use crate::vec3simd::{IVec3Simd, Vec3Simd};

const WIDTH: usize = 320;
const HEIGHT: usize = 180;
const SCALE: Scale = Scale::X4;

const MAX_ITERATONS: usize = 100;

fn hit_aabb(
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

        tmin = tmin.max(t1.min(t2)); // max(tmin, min(t1, t2));
        tmax = tmax.min(t1.max(t2)); // min(tmax, max(t1, t2));
    }

    (tmin, tmax)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn gather_u32x16_avx2(data: &[u32], indices: u32x16, mask: u32x16) -> u32x16 {
    unsafe {
        // Reinterpret the 16-lane vectors as pairs of 256-bit AVX2 registers
        let [idx_lo, idx_hi]: [__m256i; 2] = bytemuck::cast(indices);
        let [mask_lo, mask_hi]: [__m256i; 2] = bytemuck::cast(mask);

        let fallback = _mm256_setzero_si256();
        let base_ptr = data.as_ptr() as *const i32;

        // Gather lower 8 lanes and upper 8 lanes (SCALE = 4 bytes per u32)
        let res_lo = _mm256_mask_i32gather_epi32::<4>(fallback, base_ptr, idx_lo, mask_lo);
        let res_hi = _mm256_mask_i32gather_epi32::<4>(fallback, base_ptr, idx_hi, mask_hi);

        return bytemuck::cast([res_lo, res_hi]);
    }
}

#[inline(always)]
fn gather_u32x16_scalar(data: &[u32], indices: u32x16, mask: u32x16) -> u32x16 {
    // Portable scalar fallback for non-AVX2 targets (ARM NEON, SSE, etc.)
    let idx_arr = indices.as_array();
    let mask_arr = mask.as_array();
    let mut out = [0u32; 16];

    for lane in 0..16 {
        if mask_arr[lane] != 0 {
            out[lane] = data[idx_arr[lane] as usize];
        }
    }
    u32x16::new(out)
}

#[inline(always)]
fn gather_u32x16<const USE_AVX2: bool>(data: &[u32], indices: u32x16, mask: u32x16) -> u32x16 {
    if USE_AVX2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            gather_u32x16_avx2(data, indices, mask)
        }
        #[cfg(not(target_arch = "x86_64"))]
        unreachable!()
    } else {
        gather_u32x16_scalar(data, indices, mask)
    }
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

    let use_avx2: bool = std::is_x86_feature_detected!("avx2");

    let mut window = Window::new(
        "Toy Voxel Engine",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            scale_mode: ScaleMode::Stretch,
            scale: SCALE,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to create the window");

    window.set_target_fps(60);

    let mut time_since_last_metrics = SystemTime::now();
    let mut average_frame_time = 0.0f64;

    let mut heat_overlay = false;

    let mut camera = Camera::new(
        vec3(-2.0, 3.0, -2.0),
        vec3(4.0, 4.0, 4.0),
        90.0_f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
    );

    let background_color = Vec3Simd::from(vec3(0.5, 0.7, 1.0));

    const VOLUME_EDGE: i32 = 8;
    let volume_center = Vec3Simd::from(vec3(0.0, 0.0, 0.0));
    let volume_min = volume_center + Vec3Simd::from(vec3(0.0, 0.0, 0.0));
    let volume_max = volume_center
        + Vec3Simd::from(vec3(
            VOLUME_EDGE as f32,
            VOLUME_EDGE as f32,
            VOLUME_EDGE as f32,
        ));
    let mut volume: [u32; VOLUME_EDGE.pow(3) as usize] = [0; VOLUME_EDGE.pow(3) as usize]; // 4x4x4 volume
    for i in 0..VOLUME_EDGE {
        for j in 0..VOLUME_EDGE {
            for k in 0..VOLUME_EDGE {
                let local_coord = Vec3::new(i as f32, j as f32, k as f32) + 0.5;
                let centered_coord = local_coord - (VOLUME_EDGE / 2) as f32;
                if centered_coord.length() >= (VOLUME_EDGE / 2) as f32 {
                    volume[(i * VOLUME_EDGE.pow(2) + j * VOLUME_EDGE + k) as usize] = u32::MAX;
                }
            }
        }
    }

    let dir_light_start = vec3(2.0, 2.0, 2.0);
    let dir_light_end = vec3(1.0, 1.0, 1.0);
    let light_dir = (dir_light_end - dir_light_start).normalize();

    let local_x = i32x16::new([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    let local_y = i32x16::new([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame_start = SystemTime::now();

        window
            .get_keys_pressed(KeyRepeat::No)
            .iter()
            .for_each(|key| match key {
                // Miscs
                Key::F1 => heat_overlay = !heat_overlay,
                _ => (),
            });
        {}

        let mut rot_y = 0.0f32;
        let mut fwd = 0.0f32;
        let mut rot_x = 0.0f32;

        window.get_keys().iter().for_each(|key| match key {
            // Camera controls
            Key::W => fwd += 0.1,
            Key::S => fwd -= 0.1,
            Key::A => rot_y -= 0.1,
            Key::D => rot_y += 0.1,
            Key::Space => rot_x -= 0.1,
            Key::LeftShift => rot_x += 0.1,
            _ => (),
        });

        if let Some(scroll) = window.get_scroll_wheel() {
            fwd += scroll.1 / 12.0;
        }

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

        let patches_to_dbg = vec![
            // 1 / 4
            // (WIDTH / 16, HEIGHT / 16),
            // 1 / 2
            // (WIDTH / 8, HEIGHT / 8),
        ];

        // Process image in blocks of 4x4 pixels
        for chunk_y in 0..(HEIGHT / 4) {
            for chunk_x in 0..(WIDTH / 4) {
                let x = i32x16::splat((chunk_x as i32) * 4) + local_x;
                let y = i32x16::splat((chunk_y as i32) * 4) + local_y;

                let x_f32 = f32x16::from_i32x16(x);
                let y_f32 = f32x16::from_i32x16(y);

                let to_dbg = patches_to_dbg.contains(&(chunk_x, chunk_y));

                let pixel_center = Vec3Simd::from(camera.pixel00_loc)
                    + x_f32 * Vec3Simd::from(camera.pixel_delta_u)
                    + y_f32 * Vec3Simd::from(camera.pixel_delta_v);

                let ray_direction = (pixel_center - Vec3Simd::from(camera.origin)).normalize();
                let inv_ray_direction = f32x16::splat(1.0) / ray_direction;

                let grad = 0.5 * ray_direction.y + 1.0;

                let mut color = (1.0 - grad) * 1.0 + grad * background_color;

                // Volume
                let (tmin, tmax) = hit_aabb(
                    volume_min,
                    volume_max,
                    Vec3Simd::from(camera.origin),
                    inv_ray_direction,
                );

                let t_entry = Vec3Simd::from(camera.origin) + tmin * ray_direction;

                // let intersection_coord = Vec3Simd::from(camera.origin) + tmin * ray_direction;
                // let mut voxel = IVec3Simd::from(intersection_coord.fast_trunc_int());

                let mut voxel = IVec3Simd::from(camera.origin.floor().as_ivec3());

                // if to_dbg {
                //     dbg!(tmin, tmax);
                //     dbg!(intersection_coord);
                // }

                let step = ray_direction.signum().fast_round_int();

                let t_delta = Vec3Simd::new(
                    ray_direction
                        .x
                        .simd_ne(0.0)
                        .select((1.0 / ray_direction.x).abs(), f32x16::splat(1e30f32)),
                    ray_direction
                        .y
                        .simd_ne(0.0)
                        .select((1.0 / ray_direction.y).abs(), f32x16::splat(1e30f32)),
                    ray_direction
                        .z
                        .simd_ne(0.0)
                        .select((1.0 / ray_direction.z).abs(), f32x16::splat(1e30f32)),
                );

                let mut t_max = Vec3Simd::new(
                    ray_direction.x.is_sign_positive().select(
                        ((voxel.x + 1).round_float() - camera.origin.x) * t_delta.x,
                        (camera.origin.x - voxel.x.round_float()) * t_delta.x,
                    ),
                    ray_direction.y.is_sign_positive().select(
                        ((voxel.y + 1).round_float() - camera.origin.y) * t_delta.y,
                        (camera.origin.y - voxel.y.round_float()) * t_delta.y,
                    ),
                    ray_direction.z.is_sign_positive().select(
                        ((voxel.z + 1).round_float() - camera.origin.z) * t_delta.z,
                        (camera.origin.z - voxel.z.round_float()) * t_delta.z,
                    ),
                );

                // if to_dbg {
                //     dbg!(voxel, step, t_delta, t_max);
                // }

                let mut active_mask =
                    bytemuck::cast::<f32x16, i32x16>(tmin.simd_le(tmax) & tmax.is_sign_positive());
                // let mut active_mask = i32x16::splat(-1);
                let mut hit_data = u32x16::splat(0);
                // let mut t = f32x16::splat(0.0);
                let mut hit_normal = IVec3Simd::from(IVec3::ZERO);

                let mut last_step_x = bytemuck::cast::<f32x16, i32x16>(
                    t_entry.x.simd_ge(t_entry.y) & t_entry.x.simd_ge(t_entry.z),
                );
                let mut last_step_y =
                    !last_step_x & bytemuck::cast::<f32x16, i32x16>(t_entry.y.simd_ge(t_entry.z));
                let mut last_step_z = !last_step_x & !last_step_y;

                let mut i = 0;
                let mut iterations = i32x16::splat(0);
                while i < MAX_ITERATONS {
                    // active_mask &= in_bounds;

                    let indices = voxel.x
                        + voxel.y * i32x16::splat(VOLUME_EDGE)
                        + voxel.z * i32x16::splat(VOLUME_EDGE.pow(2));

                    let in_volume_bounds = voxel.x.simd_ge(0)
                        & voxel.x.simd_lt(VOLUME_EDGE)
                        & voxel.y.simd_ge(0)
                        & voxel.y.simd_lt(VOLUME_EDGE)
                        & voxel.z.simd_ge(0)
                        & voxel.z.simd_lt(VOLUME_EDGE);

                    let sampled = if use_avx2 {
                        gather_u32x16::<true>(
                            &volume,
                            bytemuck::cast(indices),
                            bytemuck::cast(in_volume_bounds & active_mask),
                        )
                    } else {
                        gather_u32x16::<false>(
                            &volume,
                            bytemuck::cast(indices),
                            bytemuck::cast(in_volume_bounds & active_mask),
                        )
                    };

                    let hit_mask =
                        bytemuck::cast::<u32x16, i32x16>(sampled).simd_ne(0) & active_mask;

                    if hit_mask.any() {
                        hit_data =
                            bytemuck::cast::<i32x16, u32x16>(hit_mask).select(sampled, hit_data);
                        // Stop advancing rays that hit a solid voxel
                        active_mask &= !hit_mask;
                    }

                    if to_dbg {
                        dbg!(voxel);
                        // dbg!(in_bounds);
                        dbg!(active_mask, indices, sampled, hit_mask, hit_data);
                    }

                    if active_mask.none() {
                        break;
                    }

                    let mask_x = t_max.x.simd_lt(t_max.y) & t_max.x.simd_lt(t_max.z);
                    let mask_y = !mask_x & t_max.y.simd_lt(t_max.z);
                    let mask_z = !mask_x & !mask_y;

                    let i_mask_x = bytemuck::cast::<f32x16, i32x16>(mask_x);
                    let i_mask_y = bytemuck::cast::<f32x16, i32x16>(mask_y);
                    let i_mask_z = bytemuck::cast::<f32x16, i32x16>(mask_z);

                    // if to_dbg {
                    //     dbg!(mask_x, mask_y, mask_z);
                    // }

                    voxel.x = i_mask_x.select(voxel.x + step.x, voxel.x);
                    t_max.x = mask_x.select(t_max.x + t_delta.x, t_max.x);

                    voxel.y = i_mask_y.select(voxel.y + step.y, voxel.y);
                    t_max.y = mask_y.select(t_max.y + t_delta.y, t_max.y);

                    voxel.z = i_mask_z.select(voxel.z + step.z, voxel.z);
                    t_max.z = mask_z.select(t_max.z + t_delta.z, t_max.z);

                    i += 1;
                    iterations = active_mask.simd_ne(0).select(iterations + 1, iterations);
                }

                if to_dbg {
                    dbg!(iterations);
                }

                let hit_normal = Vec3Simd::new(
                    last_step_x.select(-step.x, i32x16::splat(0)).round_float(),
                    last_step_y.select(-step.y, i32x16::splat(0)).round_float(),
                    last_step_z.select(-step.z, i32x16::splat(0)).round_float(),
                );

                let hit_mask = hit_data.simd_ne(0);

                let hit_color_r: u32x16 = (hit_data & (255 << 16)) >> 16; // 0R00
                let hit_color_g: u32x16 = (hit_data & (255 << 8)) >> 8; // 00G0
                let hit_color_b: u32x16 = hit_data & 255; // 000B

                let hit_color = Vec3Simd::new(
                    hit_color_r.cast_signed().round_float() / 255.0f32,
                    hit_color_g.cast_signed().round_float() / 255.0f32,
                    hit_color_b.cast_signed().round_float() / 255.0f32,
                );

                let diffuse = hit_normal
                    .dot(Vec3Simd::from(light_dir))
                    .max(f32x16::splat(0.0));

                color = Vec3Simd::select_scalar(bytemuck::cast(hit_mask), hit_color, color);

                if heat_overlay {
                    let heat_color = Vec3Simd::new(
                        iterations.round_float() / 100 as f32,
                        f32x16::ZERO,
                        f32x16::ZERO,
                    );

                    // Alpha blend "heat" (number of iterations per ray) with background
                    let alpha = 0.3f32;
                    let blend_color =
                        f32x16::splat(alpha) * heat_color + (1.0 - f32x16::splat(alpha)) * color;
                    color = Vec3Simd::select_scalar(bytemuck::cast(hit_mask), blend_color, color);
                }

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

        // Exponential Moving Average frame time
        average_frame_time +=
            (frame_start.elapsed().unwrap().as_millis() as f64 - average_frame_time) as f64 * 0.03;

        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();

        if time_since_last_metrics.elapsed().unwrap().as_millis() >= 1000 {
            println!(
                "fps: {:>3}, frame: {:.3}ms",
                window.get_fps(),
                average_frame_time
            );
            time_since_last_metrics = SystemTime::now();
        }

        // break;
    }
}
