// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::cmp;

use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};
use image::{
    imageops::FilterType, DynamicImage, GenericImageView as _, GrayAlphaImage, GrayImage,
    ImageBuffer, Rgb32FImage, RgbImage, Rgba32FImage, RgbaImage,
};
use ndarray::{s, Array, ArrayD, Axis, ShapeError};
use ort::TensorValueType;

use crate::Variant;

/// Re-normalize a floating point value (either scalar or array) from the range [0,1] to the range
/// [-1, 1].
macro_rules! convert_from_0_1_to_neg1_1 {
    ($f:expr) => {
        $f * 2. - 1.
    };
}

/// Re-normalize a floating point value (either scalar or array) from the range [-1, 1] to the
/// range [0, 1].
macro_rules! convert_from_neg1_1_to_0_1 {
    ($f:expr) => {
        ($f + 1.) / 2.
    };
}

pub(super) struct ModelImage(pub(super) u32, pub(super) Variant, pub(super) DynamicImage);

/// The error type for the `image_processing` module.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Something went wrong during inference.
    #[error("onnx error: {0}")]
    Ort(#[from] ort::Error),

    /// We were unable to make an `ndarray::Array` of the requested shape.
    #[error("shape error: {0}")]
    Shape(#[from] ShapeError),

    /// The input array has an unexpected shape.
    #[error("invalid shape")]
    InvalidShape,

    // We could not create an `ImageBuffer` with the requested array.
    #[error("invalid image")]
    Image,

    /// We were unable to resize the input image.
    #[error("resize error: {0}")]
    Resize(#[from] fast_image_resize::ResizeError),
}

impl TryFrom<ModelImage> for ort::Value<TensorValueType<f32>> {
    type Error = Error;

    fn try_from(ModelImage(size, variant, img): ModelImage) -> Result<Self, Self::Error> {
        let (w, h, xpos, ypos) = center_crop_size_and_offset(variant, &img);

        let options = ResizeOptions::new()
            .crop(xpos as f64, ypos as f64, w as f64, h as f64)
            .resize_alg(ResizeAlg::Interpolation(
                fast_image_resize::FilterType::Bilinear,
            ));
        let modified_img = resize_img(&img, size, size, options)?;

        let img = modified_img.into_rgb32f().into_vec();
        let array = Array::from(img);

        // The `image` crate normalizes to `[0,1]`. Trustmark wants images normalized to `[-1,1]`.
        let array = convert_from_0_1_to_neg1_1!(array);

        let mut array = array
            .to_shape([size as usize, size as usize, 3])?
            .insert_axis(Axis(3))
            .reversed_axes();
        array.swap_axes(2, 3);
        assert_eq!(array.shape(), &[1, 3, size as usize, size as usize]);
        Ok(ort::Value::from_array(&array)?)
    }
}

impl TryFrom<(u32, Variant, ArrayD<f32>)> for ModelImage {
    type Error = Error;

    fn try_from(
        (size, variant, mut array): (u32, Variant, ArrayD<f32>),
    ) -> Result<Self, Self::Error> {
        let &[1, 3, height, width] = &array.shape().to_owned()[..] else {
            return Err(Error::InvalidShape);
        };
        array.swap_axes(2, 3);
        let array = array.reversed_axes().remove_axis(Axis(3));
        let array = array.to_shape([width * height * 3])?;

        // The `image` crate normalizes to `[0,1]`. Trustmark wants images normalized to `[-1,1]`.
        let array = convert_from_neg1_1_to_0_1!(array);

        let image = Rgb32FImage::from_vec(width as u32, height as u32, array.to_vec())
            .ok_or(Error::Image)?;

        Ok(Self(size, variant, image.into()))
    }
}

/// Apply `residual` to the `input` in place.
///
/// This function computes each required residual pixel on demand by bilinearly interpolating from
/// the source residual image, avoiding the allocation of a full-size upscaled residual.
pub(super) fn apply_residual_in_place(input: &mut DynamicImage, residual: DynamicImage) {
    let (w, h) = input.dimensions();
    let residual_rgba = residual.into_rgba32f();
    match input {
        DynamicImage::ImageRgba32F(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                for c in 0..3 {
                    let base = convert_from_0_1_to_neg1_1!(target_pixel[c]);
                    target_pixel[c] =
                        convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[c], 1.0));
                }
            });
        }
        DynamicImage::ImageRgb32F(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                for c in 0..3 {
                    let base = convert_from_0_1_to_neg1_1!(target_pixel[c]);
                    target_pixel[c] =
                        convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[c], 1.0));
                }
            });
        }
        DynamicImage::ImageLuma8(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                let base = convert_from_0_1_to_neg1_1!(target_pixel[0] as f32 / u8::MAX as f32);
                let output = convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[0], 1.0));
                target_pixel[0] = (output * u8::MAX as f32) as u8;
            });
        }
        DynamicImage::ImageLumaA8(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                let base = convert_from_0_1_to_neg1_1!(target_pixel[0] as f32 / u8::MAX as f32);
                let output = convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[0], 1.0));
                target_pixel[0] = (output * u8::MAX as f32) as u8;
            });
        }
        DynamicImage::ImageRgb8(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                for c in 0..3 {
                    let base = convert_from_0_1_to_neg1_1!(target_pixel[c] as f32 / u8::MAX as f32);
                    let output =
                        convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[c], 1.0));
                    target_pixel[c] = (output * u8::MAX as f32) as u8;
                }
            });
        }
        DynamicImage::ImageRgba8(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                for c in 0..3 {
                    let base = convert_from_0_1_to_neg1_1!(target_pixel[c] as f32 / u8::MAX as f32);
                    let output =
                        convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[c], 1.0));
                    target_pixel[c] = (output * u8::MAX as f32) as u8;
                }
            });
        }
        DynamicImage::ImageLuma16(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                let base = convert_from_0_1_to_neg1_1!(target_pixel[0] as f32 / u16::MAX as f32);
                let output = convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[0], 1.0));
                target_pixel[0] = (output * u16::MAX as f32) as u16;
            });
        }
        DynamicImage::ImageLumaA16(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                let base = convert_from_0_1_to_neg1_1!(target_pixel[0] as f32 / u16::MAX as f32);
                let output = convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[0], 1.0));
                target_pixel[0] = (output * u16::MAX as f32) as u16;
            });
        }
        DynamicImage::ImageRgb16(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                for c in 0..3 {
                    let base =
                        convert_from_0_1_to_neg1_1!(target_pixel[c] as f32 / u16::MAX as f32);
                    let output =
                        convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[c], 1.0));
                    target_pixel[c] = (output * u16::MAX as f32) as u16;
                }
            });
        }
        DynamicImage::ImageRgba16(target) => {
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = target.get_pixel_mut(x, y);
                for c in 0..3 {
                    let base =
                        convert_from_0_1_to_neg1_1!(target_pixel[c] as f32 / u16::MAX as f32);
                    let output =
                        convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[c], 1.0));
                    target_pixel[c] = (output * u16::MAX as f32) as u16;
                }
            });
        }
        // DynamicImage is non-exhaustive; keep a forward-compatible fallback.
        image => {
            let mut rgba_target = image.to_rgba32f();
            for_each_interpolated_residual(w, h, residual_rgba, |x, y, residual_pixel| {
                let target_pixel = rgba_target.get_pixel_mut(x, y);
                for c in 0..3 {
                    let base = convert_from_0_1_to_neg1_1!(target_pixel[c]);
                    target_pixel[c] =
                        convert_from_neg1_1_to_0_1!(f32::min(base + residual_pixel[c], 1.0));
                }
            });
            *image = DynamicImage::ImageRgba32F(rgba_target);
        }
    }
}

fn for_each_interpolated_residual<F>(
    width: u32,
    height: u32,
    mut residual_rgba: Rgba32FImage,
    mut apply_residual: F,
) where
    F: FnMut(u32, u32, [f32; 4]),
{
    let (rw, rh) = residual_rgba.dimensions();
    if rw > width || rh > height {
        let target_w = rw.min(width);
        let target_h = rh.min(height);
        residual_rgba = DynamicImage::ImageRgba32F(residual_rgba)
            .resize_exact(target_w, target_h, FilterType::Triangle)
            .into_rgba32f();
    }

    let (rw, rh) = residual_rgba.dimensions();
    let x_scale = rw as f32 / width as f32;
    let y_scale = rh as f32 / height as f32;
    let src_x_start = (0.5 * x_scale) - 0.5;
    let src_y_start = (0.5 * y_scale) - 0.5;

    let mut cached_neighborhood: Option<(f32, f32)> = None;
    let mut cached_pixels = [[0.0; 4]; 4];

    let mut src_y = src_y_start;
    for y in 0..height {
        let src_y_clamped = src_y.clamp(0.0, (rh - 1) as f32);
        let mut src_x = src_x_start;
        for x in 0..width {
            let src_x_clamped = src_x.clamp(0.0, (rw - 1) as f32);
            let x0f = src_x_clamped.floor();
            let y0f = src_y_clamped.floor();
            let wx = src_x_clamped - x0f;
            let wy = src_y_clamped - y0f;

            let neighborhood = (x0f, y0f);
            if cached_neighborhood != Some(neighborhood) {
                cached_neighborhood = Some(neighborhood);
                let x0 = x0f as u32;
                let y0 = y0f as u32;
                let x1 = (x0 + 1).min(rw - 1);
                let y1 = (y0 + 1).min(rh - 1);
                cached_pixels[0] = residual_rgba.get_pixel(x0, y0).0;
                cached_pixels[1] = residual_rgba.get_pixel(x1, y0).0;
                cached_pixels[2] = residual_rgba.get_pixel(x0, y1).0;
                cached_pixels[3] = residual_rgba.get_pixel(x1, y1).0;
            }

            let mut interpolated = [0.0; 4];
            for channel in 0..4 {
                let top =
                    (cached_pixels[0][channel] * (1.0 - wx)) + (cached_pixels[1][channel] * wx);
                let bottom =
                    (cached_pixels[2][channel] * (1.0 - wx)) + (cached_pixels[3][channel] * wx);
                interpolated[channel] =
                    convert_from_0_1_to_neg1_1!((top * (1.0 - wy)) + (bottom * wy));
            }

            apply_residual(x, y, interpolated);
            src_x += x_scale;
        }
        src_y += y_scale;
    }
}

/// Return the size and offset of the "center-cropped" image.
///
/// Returns `(width, height, xpos, ypos)` for the square to crop.
///
/// For long-skinny images or short-wide images, we want to crop a square image with side length of
/// the shorter side out of the center of the image for the model.
fn center_crop_size_and_offset(variant: Variant, img: &DynamicImage) -> (u32, u32, u32, u32) {
    let (width, height) = img.dimensions();

    if height > width * 2 || width > height * 2 || variant == Variant::P {
        let m = cmp::min(height, width);
        let offset = (cmp::max(height, width) - m) / 2;

        let xpos;
        let ypos;
        if height > width {
            xpos = 0;
            ypos = offset;
        } else {
            ypos = 0;
            xpos = offset;
        }

        (m, m, xpos, ypos)
    } else {
        (width, height, 0, 0)
    }
}

/// Returns a new `DynamicImage`, resized to `width` by `height` with the specified `ResizeOptions`.
fn resize_img(
    img: &DynamicImage,
    width: u32,
    height: u32,
    options: ResizeOptions,
) -> Result<DynamicImage, Error> {
    let mut modified_img = match img {
        DynamicImage::ImageLuma8(_) => DynamicImage::ImageLuma8(GrayImage::new(width, height)),
        DynamicImage::ImageLumaA8(_) => {
            DynamicImage::ImageLumaA8(GrayAlphaImage::new(width, height))
        }
        DynamicImage::ImageRgb8(_) => DynamicImage::ImageRgb8(RgbImage::new(width, height)),
        DynamicImage::ImageRgba8(_) => DynamicImage::ImageRgba8(RgbaImage::new(width, height)),
        DynamicImage::ImageLuma16(_) => DynamicImage::ImageLuma16(ImageBuffer::new(width, height)),
        DynamicImage::ImageLumaA16(_) => {
            DynamicImage::ImageLumaA16(ImageBuffer::new(width, height))
        }
        DynamicImage::ImageRgb16(_) => DynamicImage::ImageRgb16(ImageBuffer::new(width, height)),
        DynamicImage::ImageRgba16(_) => DynamicImage::ImageRgba16(ImageBuffer::new(width, height)),
        DynamicImage::ImageRgb32F(_) => DynamicImage::ImageRgb32F(Rgb32FImage::new(width, height)),
        DynamicImage::ImageRgba32F(_) => {
            DynamicImage::ImageRgba32F(Rgba32FImage::new(width, height))
        }
        // Technically unreachable, but we error for safety.
        _ => return Err(Error::Image),
    };
    Resizer::new().resize(img, &mut modified_img, &options)?;

    Ok(modified_img)
}

/// Applies the mean padding boundary artifact mitigation.
///
/// Center cropped images have a vertical line problem along the boundary of the residual. This
/// transformation makes this boundary less visible.
pub(super) fn remove_boundary_artifact(
    mut residual: ArrayD<f32>,
    (width, height): (usize, usize),
    _variant: Variant,
) -> ArrayD<f32> {
    // We're going to replace the border of the residual with the mean and also pad the non-center
    // areas with the mean value.
    let channel_means: Vec<f32> = (0_usize..3)
        .map(|i| residual.slice(s![.., i, .., ..]).mean().unwrap())
        .collect();

    // We want one dimension of the output to be 256 and we we want the aspect ratio of the output
    // to match the input image.
    let other_dim;
    let mut mean_padded: ndarray::Array4<f32> = if width > height {
        other_dim = ((width as f32 / height as f32) * 256.0) as usize;
        ndarray::Array4::zeros([1, 3, 256_usize, other_dim])
    } else {
        other_dim = ((height as f32 / width as f32) * 256.0) as usize;
        ndarray::Array4::zeros([1, 3, other_dim, 256])
    };

    // This softens the transition between the residual area and the rest of the image.
    let border = 2;
    for (i, mean) in channel_means.iter().enumerate() {
        residual.slice_mut(s![0, i, ..border, ..]).fill(*mean);
        residual.slice_mut(s![0, i, -border.., ..]).fill(*mean);
        residual.slice_mut(s![0, i, .., -border..]).fill(*mean);
        residual.slice_mut(s![0, i, .., ..border]).fill(*mean);
        mean_padded.slice_mut(s![0, i, .., ..]).fill(*mean);
    }

    if width > height {
        let leftover = (other_dim - 256) / 2;
        mean_padded
            .slice_mut(s![.., .., .., leftover..(leftover + 256)])
            .assign(&residual);
    } else {
        let leftover = (other_dim - 256) / 2;
        mean_padded
            .slice_mut(s![.., .., leftover..(leftover + 256), ..])
            .assign(&residual);
    }

    mean_padded.into_dyn()
}

#[cfg(test)]
mod tests {
    use image::{imageops, Pixel as _};
    use ndarray::Array4;

    use super::*;

    #[test]
    fn renormalize_from_0_1() {
        assert_eq!(convert_from_0_1_to_neg1_1!(0.), -1.);
        assert_eq!(convert_from_0_1_to_neg1_1!(0.5), 0.);
        assert_eq!(convert_from_0_1_to_neg1_1!(0.99), 0.98);
    }

    #[test]
    fn renormalize_from_neg1_1() {
        assert_eq!(convert_from_neg1_1_to_0_1!(-1.), 0.);
        assert_eq!(convert_from_neg1_1_to_0_1!(0.5), 0.75);
        assert_eq!(convert_from_neg1_1_to_0_1!(-0.1), 0.45);
    }

    #[test]
    fn normal_image() {
        let image = DynamicImage::new(100, 110, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::Q, &image),
            (100, 110, 0, 0)
        );
    }

    #[test]
    fn skinny_image() {
        let image = DynamicImage::new(10, 100, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::Q, &image),
            (10, 10, 0, 45)
        );
    }

    #[test]
    fn wide_image() {
        let image = DynamicImage::new(101, 10, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::Q, &image),
            (10, 10, 45, 0)
        );
    }

    #[test]
    fn always_crop_p() {
        let image = DynamicImage::new(100, 110, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::P, &image),
            (100, 100, 0, 5)
        );
    }

    #[test]
    fn remove_boundary_artifact_tall() {
        let residual: Array4<f32> = Array4::ones([1, 3, 256, 256]);
        let width = 256;
        let height = 298;

        let output = remove_boundary_artifact(residual.into_dyn(), (width, height), Variant::P);

        assert_eq!(output.shape(), &[1, 3, 298, 256]);
    }

    fn apply_residual_reference(input: DynamicImage, residual: DynamicImage) -> DynamicImage {
        let has_alpha = input.color().has_alpha();
        let (w, h) = input.dimensions();

        let applied = {
            let input = input.clone().into_rgba32f();
            let mut target = input.clone();

            let residual = residual.resize_exact(w, h, FilterType::Triangle);
            let residual = residual.into_rgba32f();

            for ((target, residual), original) in target
                .pixels_mut()
                .zip(residual.pixels())
                .zip(input.pixels())
            {
                target.apply2(residual, |x, y| {
                    let x = convert_from_0_1_to_neg1_1!(x);
                    let y = convert_from_0_1_to_neg1_1!(y);

                    convert_from_neg1_1_to_0_1!(f32::min(x + y, 1.0))
                });
                target[3] = original[3];
            }

            target
        };

        if has_alpha {
            let mut input = input.into_rgba32f();
            imageops::replace(&mut input, &applied, 0, 0);
            input.into()
        } else {
            let mut input = input.into_rgb32f();
            let applied = DynamicImage::ImageRgba32F(applied).into_rgb32f();
            imageops::replace(&mut input, &applied, 0, 0);
            input.into()
        }
    }

    fn apply_residual_reference_with_min_resize(
        input: DynamicImage,
        residual: DynamicImage,
    ) -> DynamicImage {
        let (w, h) = input.dimensions();
        let (rw, rh) = residual.dimensions();
        let residual = if rw > w || rh > h {
            residual.resize_exact(rw.min(w), rh.min(h), FilterType::Triangle)
        } else {
            residual
        };
        apply_residual_reference(input, residual)
    }

    #[test]
    fn apply_residual_in_place_matches_reference() {
        let mut input = Rgba32FImage::new(23, 17);
        for (x, y, pixel) in input.enumerate_pixels_mut() {
            let xf = x as f32 / 23.0;
            let yf = y as f32 / 17.0;
            *pixel = image::Rgba([xf, yf, (xf + yf) / 2.0, 1.0 - (xf * yf * 0.5)]);
        }
        let input = DynamicImage::ImageRgba32F(input);

        let mut residual = Rgb32FImage::new(11, 9);
        for (x, y, pixel) in residual.enumerate_pixels_mut() {
            let xf = x as f32 / 11.0;
            let yf = y as f32 / 9.0;
            *pixel = image::Rgb([0.2 + (xf * 0.4), 0.3 + (yf * 0.2), 0.1 + ((xf + yf) * 0.2)]);
        }
        let residual = DynamicImage::ImageRgb32F(residual);

        let expected = apply_residual_reference(input.clone(), residual.clone()).into_rgba32f();

        let mut actual_in_place = input.clone();
        apply_residual_in_place(&mut actual_in_place, residual);
        let actual_in_place = actual_in_place.into_rgba32f();

        for (actual, expected) in actual_in_place.pixels().zip(expected.pixels()) {
            for c in 0..4 {
                assert!((actual[c] - expected[c]).abs() <= f32::EPSILON);
            }
        }
    }

    #[test]
    fn apply_residual_in_place_matches_reference_with_larger_residual() {
        let mut input = Rgba32FImage::new(23, 17);
        for (x, y, pixel) in input.enumerate_pixels_mut() {
            let xf = x as f32 / 23.0;
            let yf = y as f32 / 17.0;
            *pixel = image::Rgba([xf, yf, (xf + yf) / 2.0, 1.0 - (xf * yf * 0.5)]);
        }
        let input = DynamicImage::ImageRgba32F(input);

        let mut residual = Rgb32FImage::new(47, 31);
        for (x, y, pixel) in residual.enumerate_pixels_mut() {
            let xf = x as f32 / 47.0;
            let yf = y as f32 / 31.0;
            *pixel = image::Rgb([0.2 + (xf * 0.4), 0.3 + (yf * 0.2), 0.1 + ((xf + yf) * 0.2)]);
        }
        let residual = DynamicImage::ImageRgb32F(residual);

        let expected = apply_residual_reference(input.clone(), residual.clone()).into_rgba32f();

        let mut actual_in_place = input.clone();
        apply_residual_in_place(&mut actual_in_place, residual);
        let actual_in_place = actual_in_place.into_rgba32f();

        for (actual, expected) in actual_in_place.pixels().zip(expected.pixels()) {
            for c in 0..4 {
                assert!((actual[c] - expected[c]).abs() <= f32::EPSILON);
            }
        }
    }

    #[test]
    fn apply_residual_in_place_matches_reference_with_mixed_residual_dims() {
        let mut input = Rgba32FImage::new(23, 17);
        for (x, y, pixel) in input.enumerate_pixels_mut() {
            let xf = x as f32 / 23.0;
            let yf = y as f32 / 17.0;
            *pixel = image::Rgba([xf, yf, (xf + yf) / 2.0, 1.0 - (xf * yf * 0.5)]);
        }
        let input = DynamicImage::ImageRgba32F(input);

        // Wider than input, but shorter than input.
        let mut residual = Rgb32FImage::new(47, 11);
        for (x, y, pixel) in residual.enumerate_pixels_mut() {
            let xf = x as f32 / 47.0;
            let yf = y as f32 / 11.0;
            *pixel = image::Rgb([0.2 + (xf * 0.4), 0.3 + (yf * 0.2), 0.1 + ((xf + yf) * 0.2)]);
        }
        let residual = DynamicImage::ImageRgb32F(residual);

        let expected = apply_residual_reference_with_min_resize(input.clone(), residual.clone())
            .into_rgba32f();

        let mut actual_in_place = input.clone();
        apply_residual_in_place(&mut actual_in_place, residual);
        let actual_in_place = actual_in_place.into_rgba32f();

        for (actual, expected) in actual_in_place.pixels().zip(expected.pixels()) {
            for c in 0..4 {
                assert!((actual[c] - expected[c]).abs() <= f32::EPSILON);
            }
        }
    }
}
