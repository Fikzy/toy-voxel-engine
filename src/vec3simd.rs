use std::ops::{Add, Div, Mul, Sub};

use glam::Vec3;
use wide::f32x16;

#[derive(Clone, Copy)]
pub struct Vec3Simd<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

// "Mock" Simd trait
pub trait Simd {
    fn sqrt(self) -> Self;
}

impl Simd for f32x16 {
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

// For left-hand operations (f32x16, Vec3Simd)
impl Add<Vec3Simd<f32x16>> for f32x16 {
    type Output = Vec3Simd<f32x16>;
    #[inline]
    fn add(self, rhs: Vec3Simd<f32x16>) -> Self::Output {
        Vec3Simd::new(self + rhs.x, self + rhs.y, self + rhs.z)
    }
}
impl Sub<Vec3Simd<f32x16>> for f32x16 {
    type Output = Vec3Simd<f32x16>;
    #[inline]
    fn sub(self, rhs: Vec3Simd<f32x16>) -> Self::Output {
        Vec3Simd::new(self - rhs.x, self - rhs.y, self - rhs.z)
    }
}
impl Mul<Vec3Simd<f32x16>> for f32x16 {
    type Output = Vec3Simd<f32x16>;
    #[inline]
    fn mul(self, rhs: Vec3Simd<f32x16>) -> Self::Output {
        Vec3Simd::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}
impl Div<Vec3Simd<f32x16>> for f32x16 {
    type Output = Vec3Simd<f32x16>;
    #[inline]
    fn div(self, rhs: Vec3Simd<f32x16>) -> Self::Output {
        Vec3Simd::new(self / rhs.x, self / rhs.y, self / rhs.z)
    }
}

impl<T> Vec3Simd<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn norm(self) -> T
    where
        T: Add<Output = T> + Mul<Output = T> + Simd + Copy,
    {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Vec3Simd<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Div<Output = T> + Simd + Copy,
    {
        self / self.norm()
    }
}

impl<T> From<Vec3> for Vec3Simd<T>
where
    T: From<f32>,
{
    #[inline]
    fn from(v: Vec3) -> Self {
        Self::new(T::from(v.x), T::from(v.y), T::from(v.z))
    }
}

impl<T> Add for Vec3Simd<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl<T> Add<T> for Vec3Simd<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl<T> Sub for Vec3Simd<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl<T> Sub<T> for Vec3Simd<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl<T> Mul for Vec3Simd<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl<T> Mul<T> for Vec3Simd<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl<T> Div for Vec3Simd<T>
where
    T: Div<Output = T> + Copy,
{
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl<T> Div<T> for Vec3Simd<T>
where
    T: Div<Output = T> + Copy,
{
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}
