// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::{fs::OpenOptions, path::PathBuf};

use clap::{Parser, Subcommand};
use image::{codecs::jpeg::JpegEncoder, ImageFormat};
use rand::{distributions::Standard, prelude::Distribution as _};
use trustmark::{Trustmark, Variant, Version};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long)]
    models: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Encode a watermark into an image
    Encode {
        /// The image to encode.
        #[arg(short)]
        input: PathBuf,
        /// The path to save the watermarked image.
        #[arg(short)]
        output: PathBuf,
        /// The watermark to encode. Defaults to random if not specified.
        #[arg(short, long)]
        watermark: Option<String>,
        /// The BCH version to encode with. Defaults to BchSuper.
        #[arg(long)]
        version: Option<Version>,
        /// The model variant to encode with.
        #[arg(long)]
        variant: Option<Variant>,
        /// If the requested output is JPEG, the quality to use for encoding.
        #[arg(long)]
        quality: Option<u8>,
    },
    /// Decode a watermark from an image
    Decode {
        #[arg(short)]
        input: PathBuf,
        /// The model variant to decode with.
        #[arg(long)]
        variant: Option<Variant>,
    },
}

impl Command {
    /// Extract the version to use from this `Command`.
    fn get_version(&self) -> Version {
        match self {
            Command::Encode {
                version: Some(version),
                ..
            } => *version,
            _ => Version::Bch5,
        }
    }

    /// Extract the variant to use from this `Command`.
    fn get_variant(&self) -> Variant {
        match self {
            Command::Encode {
                variant: Some(variant),
                ..
            } => *variant,
            Command::Decode {
                variant: Some(variant),
                ..
            } => *variant,
            _ => Variant::Q,
        }
    }
}

/// Generate a random watermark with as many bits as specified by `bits`.
///
/// # Example
///
/// ```rust
/// # fn main() {
/// println!("{}", gen_watermark(4)); // 1010
/// # }
/// ```
fn gen_watermark(bits: usize) -> String {
    let mut rng = rand::thread_rng();
    let v: Vec<bool> = Standard.sample_iter(&mut rng).take(bits).collect();
    v.into_iter()
        .map(|bit| if bit { '1' } else { '0' })
        .collect()
}

fn main() {
    let args = Args::parse();
    let tm = Trustmark::new(
        &args.models,
        args.command.get_variant(),
        args.command.get_version(),
    )
    .unwrap();
    match args.command {
        Command::Encode {
            input,
            output,
            watermark,
            version,
            quality,
            ..
        } => {
            let mut input = image::open(input).unwrap();
            let watermark = watermark.unwrap_or_else(|| {
                gen_watermark(version.unwrap_or(Version::Bch5).data_bits().into())
            });
            tm.encode_in_place(watermark, &mut input, 0.95).unwrap();

            let format = ImageFormat::from_path(&output).unwrap();
            match format {
                // JPEG encoding can make visual artifacts worse, so we encode with a higher
                // quality than the default (or the quality requested by the user).
                ImageFormat::Jpeg => {
                    let quality = quality.unwrap_or(90);
                    let mut writer = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&output)
                        .unwrap();
                    let encoder = JpegEncoder::new_with_quality(&mut writer, quality);
                    input.to_rgb8().write_with_encoder(encoder).unwrap();
                }
                _ => {
                    input.to_rgba8().save(&output).unwrap();
                }
            }
        }
        Command::Decode { input, .. } => {
            let input = image::open(input).unwrap();
            match tm.decode(input) {
                Ok(decoded) => println!("Found watermark: {decoded}"),
                Err(trustmark::Error::CorruptWatermark) => {
                    println!("Corrupt or missing watermark")
                }
                err => panic!("{err:?}"),
            }
        }
    }
}
