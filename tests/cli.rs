//! CLI integration tests (no extra dependencies: Cargo exposes the built
//! binary path via `CARGO_BIN_EXE_vanish`).

use std::process::Command;

fn run(args: &[&str]) -> (String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_vanish"))
        .args(args)
        .output()
        .expect("binary runs");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn cli_golden_outputs() {
    let (out, code) = run(&["rung", "--p", "3457", "--s", "32", "--r", "16", "--q", "2"]);
    assert_eq!(code, 0);
    assert!(out.contains("bucket = 422"), "got: {out}");

    let (out, code) = run(&[
        "decompose",
        "--p",
        "77569",
        "--s",
        "32",
        "--r",
        "16",
        "--lam",
        "0",
    ]);
    assert_eq!(code, 0);
    assert!(out.contains("bucket(0) = 30598"), "got: {out}");

    let (out, code) = run(&[
        "attack",
        "--n",
        "2097152",
        "--k",
        "1048576",
        "--list-bits",
        "57.93",
    ]);
    assert_eq!(code, 0);
    assert!(out.contains("0.492188"), "antipodal baseline: {out}");
    assert!(out.contains("0.484375"), "ladder improvement: {out}");
}

#[test]
fn cli_error_handling() {
    let out = Command::new(env!("CARGO_BIN_EXE_vanish"))
        .args(["nonsense"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let out = Command::new(env!("CARGO_BIN_EXE_vanish"))
        .args([
            "bucket", "--p", "3458", "--s", "32", "--r", "16", "--lam", "0",
        ])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(1),
        "composite p -> clean error exit"
    );
}
