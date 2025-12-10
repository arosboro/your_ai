extern crate cmake;

use cmake::Config;
use std::{env, path::PathBuf};

fn build_and_link_mlx_c() {
    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // Force ARM64 on macOS using toolchain file
    #[cfg(target_os = "macos")]
    {
        let toolchain_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("darwin-arm64.cmake");
        config.define("CMAKE_TOOLCHAIN_FILE", toolchain_path.to_str().unwrap());
        config.define("CMAKE_OSX_ARCHITECTURES", "arm64");
        config.define("CMAKE_SYSTEM_PROCESSOR", "arm64");
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", "14.0");

        // Set SDK path to ensure proper symbol resolution
        if let Ok(sdk_path) = std::process::Command::new("xcrun")
            .args(["--show-sdk-path"])
            .output()
        {
            if sdk_path.status.success() {
                let sdk_str = String::from_utf8_lossy(&sdk_path.stdout).trim().to_string();
                config.define("CMAKE_OSX_SYSROOT", &sdk_str);
            }
        }
    }

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Add SDK root for linker to resolve weak symbols like __isPlatformVersionAtLeast
    #[cfg(target_os = "macos")]
    {
        if let Ok(sdk_path) = std::process::Command::new("xcrun")
            .args(["--show-sdk-path"])
            .output()
        {
            if sdk_path.status.success() {
                let sdk_str = String::from_utf8_lossy(&sdk_path.stdout).trim().to_string();
                println!("cargo:rustc-link-arg=-isysroot");
                println!("cargo:rustc-link-arg={}", sdk_str);
            }
        }
    }
}

fn main() {
    build_and_link_mlx_c();

    // Set libclang path for bindgen on macOS
    #[cfg(target_os = "macos")]
    {
        env::set_var("LIBCLANG_PATH", "/Library/Developer/CommandLineTools/usr/lib");
    }

    // generate bindings
    let rust_target = bindgen::RustTarget::stable(1, 82)
        .unwrap_or_else(|_| bindgen::RustTarget::nightly());
    let bindings = bindgen::Builder::default()
        .rust_target(rust_target)
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
