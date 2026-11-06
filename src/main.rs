use std::time::SystemTime;

use glam::{Vec3, vec3};
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};
use wide::{f32x16, i32x16, u32x16};

const WIDTH: usize = 320;
const HEIGHT: usize = 180;

fn from_f32_rgb(r: f32, g: f32, b: f32) -> u32 {
    let (r, g, b) = ((r * 255.0) as u32, (g * 255.0) as u32, (b * 255.0) as u32);
    (r << 16) | (g << 8) | b
}

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Ray {
        Self { origin, direction }
    }

    fn at(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }
}

fn hit_sphere(center: Vec3, radius: f32, ray: &Ray) -> f32 {
    let oc = center - ray.origin;

    let a = ray.direction.length_squared();
    let h = ray.direction.dot(oc);
    let c = oc.length_squared() - radius * radius;
    let discriminant = h * h - a * c;

    if discriminant < 0.0 {
        -1.0
    } else {
        (h - discriminant.sqrt()) / a
    }
}

fn hit_cube(min: Vec3, max: Vec3, ray: &Ray) -> (f32, f32) {
    let dir_inv = 1.0 / ray.direction;

    let mut tmin = 0.0f32;
    let mut tmax = f32::MAX;

    for i in 0..3 {
        let t1 = (min.to_array()[i] - ray.origin.to_array()[i]) * dir_inv.to_array()[i];
        let t2 = (max.to_array()[i] - ray.origin.to_array()[i]) * dir_inv.to_array()[i];

        tmin = tmin.max(t1.min(t2).min(tmax)); // max(tmin, min(min(t1, t2), tmax));
        tmax = tmax.min(t1.max(t2).max(tmin)); // min(tmax, max(max(t1, t2), tmin));
    }

    (tmin, tmax)
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

    fn ray_for_pixel(&self, x: usize, y: usize) -> Ray {
        let pixel_center =
            self.pixel00_loc + x as f32 * self.pixel_delta_u + y as f32 * self.pixel_delta_v;
        let direction = (pixel_center - self.origin).normalize();
        Ray::new(self.origin, direction)
    }
}

fn main() {
    let mut buffer = vec![0u32; WIDTH * HEIGHT];

    let mut window = Window::new(
        "Voxel Engine",
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

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut rot_y = 0.0f32;
        let mut fwd = 0.0f32;

        window.get_keys().iter().for_each(|key| match key {
            Key::W => fwd += 0.01,
            Key::S => fwd -= 0.01,
            Key::A => rot_y -= 0.1,
            Key::D => rot_y += 0.1,
            _ => (),
        });

        // Move camera along its forward axis
        camera.origin += camera.forward * fwd;

        // Rotate camera around target on Y axis
        let mut pos = camera.origin - camera.target;
        pos = vec3(
            pos.x * rot_y.cos() + pos.z * rot_y.sin(),
            pos.y,
            -pos.x * rot_y.sin() + pos.z * rot_y.cos(),
        );
        camera.origin = pos + camera.target;

        camera.update();

        let pixel00_loc_x = f32x16::splat(camera.pixel00_loc.x);
        let pixel00_loc_y = f32x16::splat(camera.pixel00_loc.y);
        let pixel00_loc_z = f32x16::splat(camera.pixel00_loc.z);

        let delta_ux = f32x16::splat(camera.pixel_delta_u.x);
        let delta_uy = f32x16::splat(camera.pixel_delta_u.y);
        let delta_uz = f32x16::splat(camera.pixel_delta_u.z);

        let delta_vx = f32x16::splat(camera.pixel_delta_v.x);
        let delta_vy = f32x16::splat(camera.pixel_delta_v.y);
        let delta_vz = f32x16::splat(camera.pixel_delta_v.z);

        let origin_x = f32x16::splat(camera.origin.x);
        let origin_y = f32x16::splat(camera.origin.y);
        let origin_z = f32x16::splat(camera.origin.z);

        let local_x = i32x16::new([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
        let local_y = i32x16::new([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);

        // Process image in blocks of 4x4 pixels
        for chunk_y in 0..(HEIGHT / 4) {
            for chunk_x in 0..(WIDTH / 4) {
                let x = i32x16::splat((chunk_x as i32) * 4) + local_x;
                let y = i32x16::splat((chunk_y as i32) * 4) + local_y;

                let x_f32 = f32x16::from_i32x16(x);
                let y_f32 = f32x16::from_i32x16(y);

                let pixel_center_x = pixel00_loc_x + (x_f32 * delta_ux) + (y_f32 * delta_vx);
                let pixel_center_y = pixel00_loc_y + (x_f32 * delta_uy) + (y_f32 * delta_vy);
                let pixel_center_z = pixel00_loc_z + (x_f32 * delta_uz) + (y_f32 * delta_vz);

                let ray_direction_diff_x = pixel_center_x - origin_x;
                let ray_direction_diff_y = pixel_center_y - origin_y;
                let ray_direction_diff_z = pixel_center_z - origin_z;

                let norm = (ray_direction_diff_x * ray_direction_diff_x
                    + ray_direction_diff_y * ray_direction_diff_y
                    + ray_direction_diff_z * ray_direction_diff_z)
                    .sqrt();

                let ray_direction_x = ray_direction_diff_x / norm;
                let ray_direction_y = ray_direction_diff_y / norm;
                let ray_direction_z = ray_direction_diff_z / norm;

                let grad = 0.5 * ray_direction_y + 1.0;

                let color_r = (1.0 - grad) * 1.0 + grad * 0.5;
                let color_g = (1.0 - grad) * 1.0 + grad * 0.7;
                let color_b = (1.0 - grad) * 1.0 + grad * 1.0;

                let color_r_int: u32x16 = bytemuck::cast((color_r * 255.0f32).fast_round_int());
                let color_g_int: u32x16 = bytemuck::cast((color_g * 255.0f32).fast_round_int());
                let color_b_int: u32x16 = bytemuck::cast((color_b * 255.0f32).fast_round_int());

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
