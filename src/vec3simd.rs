use std::ops::{Add, Div, Mul, Sub};

use glam::Vec3;

pub type FloatSimd = wide::f32x16;

#[derive(Clone, Copy, Debug)]
pub struct Vec3Simd {
    pub x: FloatSimd,
    pub y: FloatSimd,
    pub z: FloatSimd,
}

impl Vec3Simd {
    #[inline]
    pub const fn new(x: FloatSimd, y: FloatSimd, z: FloatSimd) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite().all() && self.y.is_finite().all() && self.z.is_finite().all()
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> FloatSimd {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    #[inline]
    pub fn length(self) -> FloatSimd {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn length_squared(self) -> FloatSimd {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> FloatSimd {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Vec3Simd {
        let normalized = self / self.length();
        debug_assert!(normalized.is_finite());
        normalized
    }

    #[inline]
    pub fn select_scalar(mask: FloatSimd, if_true: Self, if_false: Self) -> Self {
        Self::new(
            mask.select(if_true.x, if_false.x),
            mask.select(if_true.y, if_false.y),
            mask.select(if_true.z, if_false.z),
        )
    }

    #[inline]
    pub fn bitselect_scalar(mask: FloatSimd, if_true: Self, if_false: Self) -> Self {
        Self::new(
            mask.bitselect(if_true.x, if_false.x),
            mask.bitselect(if_true.y, if_false.y),
            mask.bitselect(if_true.z, if_false.z),
        )
    }

    #[inline]
    pub fn clamp_scalar(self, min: FloatSimd, max: FloatSimd) -> Self {
        Self::new(
            self.x.clamp(min, max),
            self.y.clamp(min, max),
            self.z.clamp(min, max),
        )
    }

    #[inline]
    pub fn to_array(self) -> [FloatSimd; 3] {
        [self.x, self.y, self.z]
    }

    #[inline]
    pub fn component(self, i: usize) -> FloatSimd {
        match i {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("Out of bounds"),
        }
    }
}

/// Vec3Simd arithmetic operations

impl Add for Vec3Simd {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<FloatSimd> for Vec3Simd {
    type Output = Self;
    #[inline]
    fn add(self, rhs: FloatSimd) -> Self {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl Sub for Vec3Simd {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub<FloatSimd> for Vec3Simd {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: FloatSimd) -> Self {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl Mul for Vec3Simd {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Mul<FloatSimd> for Vec3Simd {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: FloatSimd) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div for Vec3Simd {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Div<FloatSimd> for Vec3Simd {
    type Output = Self;
    #[inline]
    fn div(self, rhs: FloatSimd) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

/// FloatSimd arithmetic operations with Vec3Simd

impl Add<Vec3Simd> for FloatSimd {
    type Output = Vec3Simd;
    #[inline]
    fn add(self, rhs: Vec3Simd) -> Self::Output {
        Vec3Simd::new(self + rhs.x, self + rhs.y, self + rhs.z)
    }
}
impl Sub<Vec3Simd> for FloatSimd {
    type Output = Vec3Simd;
    #[inline]
    fn sub(self, rhs: Vec3Simd) -> Self::Output {
        Vec3Simd::new(self - rhs.x, self - rhs.y, self - rhs.z)
    }
}
impl Mul<Vec3Simd> for FloatSimd {
    type Output = Vec3Simd;
    #[inline]
    fn mul(self, rhs: Vec3Simd) -> Self::Output {
        Vec3Simd::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}
impl Div<Vec3Simd> for FloatSimd {
    type Output = Vec3Simd;
    #[inline]
    fn div(self, rhs: Vec3Simd) -> Self::Output {
        Vec3Simd::new(self / rhs.x, self / rhs.y, self / rhs.z)
    }
}

impl From<Vec3> for Vec3Simd {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self::new(
            FloatSimd::from(v.x),
            FloatSimd::from(v.y),
            FloatSimd::from(v.z),
        )
    }
}
