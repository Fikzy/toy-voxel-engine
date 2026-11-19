use glam::{IVec3, UVec3, Vec3};
use std::ops::{Add, Div, Mul, Neg, Sub};
use wide::{f32x16, i32x16, u32x16};

macro_rules! impl_vec3_simd_base {
    ($struct_name:ident, $simd_ty:ty, $glam_ty:ty) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $struct_name {
            pub x: $simd_ty,
            pub y: $simd_ty,
            pub z: $simd_ty,
        }

        impl $struct_name {
            #[inline]
            pub const fn new(x: $simd_ty, y: $simd_ty, z: $simd_ty) -> Self {
                Self { x, y, z }
            }

            #[inline]
            pub fn dot(self, rhs: Self) -> $simd_ty {
                (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
            }

            #[inline]
            pub fn length_squared(self) -> $simd_ty {
                self.dot(self)
            }

            #[inline]
            pub fn clamp_scalar(self, min: $simd_ty, max: $simd_ty) -> Self {
                Self::new(
                    self.x.max(min).min(max),
                    self.y.max(min).min(max),
                    self.z.max(min).min(max),
                )
            }

            #[inline]
            pub fn to_array(self) -> [$simd_ty; 3] {
                [self.x, self.y, self.z]
            }

            #[inline]
            pub fn component(self, i: usize) -> $simd_ty {
                match i {
                    0 => self.x,
                    1 => self.y,
                    2 => self.z,
                    _ => panic!("Out of bounds"),
                }
            }

            #[inline]
            pub fn set_component(&mut self, i: usize, elem: $simd_ty) {
                match i {
                    0 => self.x = elem,
                    1 => self.y = elem,
                    2 => self.z = elem,
                    _ => panic!("Out of bounds"),
                }
            }
        }

        // Implement binary operators: Vec3 + Vec3, Vec3 + Scalar, Scalar + Vec3
        impl_vec3_op!($struct_name, $simd_ty, Add, add);
        impl_vec3_op!($struct_name, $simd_ty, Sub, sub);
        impl_vec3_op!($struct_name, $simd_ty, Mul, mul);
        impl_vec3_op!($struct_name, $simd_ty, Div, div);

        impl From<$glam_ty> for $struct_name {
            #[inline]
            fn from(v: $glam_ty) -> Self {
                Self::new(
                    <$simd_ty>::from(v.x),
                    <$simd_ty>::from(v.y),
                    <$simd_ty>::from(v.z),
                )
            }
        }
    };
}

macro_rules! impl_vec3_op {
    ($struct_name:ident, $simd_ty:ty, $trait:ident, $method:ident) => {
        impl $trait for $struct_name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                Self::new(
                    self.x.$method(rhs.x),
                    self.y.$method(rhs.y),
                    self.z.$method(rhs.z),
                )
            }
        }

        impl $trait<$simd_ty> for $struct_name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: $simd_ty) -> Self {
                Self::new(
                    self.x.$method(rhs),
                    self.y.$method(rhs),
                    self.z.$method(rhs),
                )
            }
        }

        impl $trait<$struct_name> for $simd_ty {
            type Output = $struct_name;
            #[inline]
            fn $method(self, rhs: $struct_name) -> $struct_name {
                <$struct_name>::new(
                    self.$method(rhs.x),
                    self.$method(rhs.y),
                    self.$method(rhs.z),
                )
            }
        }
    };
}

macro_rules! impl_vec3_simd_signed {
    ($struct_name:ident) => {
        impl $struct_name {
            #[inline]
            pub fn abs(self) -> Self {
                Self::new(self.x.abs(), self.y.abs(), self.z.abs())
            }

            #[inline]
            pub fn signum(self) -> Self {
                Self::new(self.x.signum(), self.y.signum(), self.z.signum())
            }
        }

        impl Neg for $struct_name {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self::Output {
                Self::new(-self.x, -self.y, -self.z)
            }
        }
    };
}

macro_rules! impl_vec3_simd_float {
    ($struct_name:ident, $simd_ty:ty) => {
        impl $struct_name {
            #[inline]
            pub fn is_finite(self) -> bool {
                self.x.is_finite().all() && self.y.is_finite().all() && self.z.is_finite().all()
            }

            #[inline]
            pub fn length(self) -> $simd_ty {
                self.dot(self).sqrt()
            }

            #[inline]
            pub fn normalize(self) -> Self {
                let normalized = self / self.length();
                debug_assert!(normalized.is_finite());
                normalized
            }

            #[inline]
            pub fn select_scalar(mask: $simd_ty, if_true: Self, if_false: Self) -> Self {
                Self::new(
                    mask.select(if_true.x, if_false.x),
                    mask.select(if_true.y, if_false.y),
                    mask.select(if_true.z, if_false.z),
                )
            }

            #[inline]
            pub fn floor(self) -> $struct_name {
                $struct_name::new(self.x.floor(), self.y.floor(), self.z.floor())
            }

            #[inline]
            pub fn fast_round_int(self) -> IVec3Simd {
                IVec3Simd::new(
                    self.x.fast_round_int(),
                    self.y.fast_round_int(),
                    self.z.fast_round_int(),
                )
            }

            #[inline]
            pub fn fast_trunc_int(self) -> IVec3Simd {
                IVec3Simd::new(
                    self.x.fast_trunc_int(),
                    self.y.fast_trunc_int(),
                    self.z.fast_trunc_int(),
                )
            }
        }
    };
}

// Float vector (Base + Signed + Float)
impl_vec3_simd_base!(Vec3Simd, f32x16, Vec3);
impl_vec3_simd_signed!(Vec3Simd);
impl_vec3_simd_float!(Vec3Simd, f32x16);

// Signed integer vector (Base + Signed)
impl_vec3_simd_base!(IVec3Simd, i32x16, IVec3);
impl_vec3_simd_signed!(IVec3Simd);

// Unsigned integer vector (Base only)
impl_vec3_simd_base!(UVec3Simd, u32x16, UVec3);
