fn main() {
    use std::env;
    use std::path::PathBuf;

    let cuda_home = env::var("CUDA_HOME").unwrap();
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);

    let mut cupti_include: Option<PathBuf> = None;
    let mut cupti_libdir: Option<PathBuf> = None;
    for (inc, lib) in [
        (
            format!("{}/extras/CUPTI/include", cuda_home),
            format!("{}/extras/CUPTI/lib64", cuda_home),
        ),
        (
            format!("{}/targets/x86_64-linux/include", cuda_home),
            format!("{}/targets/x86_64-linux/lib", cuda_home),
        ),
        (
            format!("{}/include", cuda_home),
            format!("{}/lib64", cuda_home),
        ),
    ] {
        let inc_path = PathBuf::from(&inc);
        let lib_path = PathBuf::from(&lib);
        if inc_path.join("cupti.h").exists() {
            cupti_include = Some(inc_path);
            cupti_libdir = Some(lib_path);
            break;
        }
    }
    let cupti_include = cupti_include
        .expect("could not find cupti.h under CUDA_HOME — install CUPTI or set CUDA_HOME");
    let cupti_libdir = cupti_libdir.unwrap();

    cc::Build::new()
        .file("src/cupti_shim.c")
        .include(&cupti_include)
        .include(format!("{}/include", cuda_home))
        .flag("-Wno-unused-parameter")
        .std("c11")
        .compile("phantora_cupti_shim");

    println!("cargo:rustc-link-search=native={}", cupti_libdir.display());
    println!("cargo:rerun-if-changed=src/cupti_shim.c");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
