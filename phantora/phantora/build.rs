fn main() {
    use std::env;

    let cuda_home = env::var("CUDA_HOME").unwrap();
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
}
