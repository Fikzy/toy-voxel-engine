use glam::{Vec3, usizevec2, vec3};
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};

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
        Ray { origin, direction }
    }

    fn at(self, t: f32) -> Vec3 {
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

    window.set_target_fps(30);

    let window_size = usizevec2(WIDTH, HEIGHT);

    // Camera
    let focal_length = 1.0f32;
    let viewport_height = 2.0f32;
    let viewport_width = viewport_height * window_size.x as f32 / window_size.y as f32;
    let camera_center = vec3(0.0, 0.0, 0.0);

    // println!("{} {}", viewport_width, viewport_height);

    let viewport_u = vec3(viewport_width, 0.0, 0.0);
    let viewport_v = vec3(0.0, -viewport_height, 0.0);

    let pixel_delta_u = viewport_u / window_size.x as f32;
    let pixel_delta_v = viewport_v / window_size.y as f32;

    let viewport_upper_left =
        camera_center - vec3(0.0, 0.0, focal_length) - viewport_u / 2.0 - viewport_v / 2.0;
    let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut x = 0;
        let mut y = 0;
        for pixel in buffer.iter_mut() {
            let pixel_center =
                pixel00_loc + (x as f32 * pixel_delta_u) + (y as f32 * pixel_delta_v);
            let ray_direction = (pixel_center - camera_center).normalize();
            let ray = Ray::new(camera_center, ray_direction);

            let grad = 0.5 * ray.direction.y + 1.0;
            let mut color = (1.0 - grad) * vec3(1.0, 1.0, 1.0) + grad * vec3(0.5, 0.7, 1.0);

            let sphere_center = vec3(0.0, 0.0, -1.0);
            let t = hit_sphere(sphere_center, 0.5, &ray);
            if t > 0.0 {
                let n = (ray.at(t) - sphere_center).normalize();
                color = 0.5 * vec3(n.x + 1.0, n.y + 1.0, n.z + 1.0);
            }

            *pixel = from_f32_rgb(color.x, color.y, color.z);

            if x >= window_size.x - 1 {
                x = 0;
                y += 1;
            } else {
                x += 1;
            }
        }

        window
            .update_with_buffer(&buffer, window_size.x, window_size.y)
            .unwrap();
    }
}
