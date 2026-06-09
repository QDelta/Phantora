//! Performance database: persist the simulator's kernel-timing caches to
//! human-readable CSV so preset simulations can run without a GPU.
//!
//! Record mode (`--record-perf-db <dir>`) profiles on a real GPU as usual and
//! dumps the populated caches here; replay mode (`--perf-db <dir>`) loads them
//! and answers timing queries from the DB, never touching the GPU.
//!
//! On disk the DB is a directory of CSV files (one per timing table) plus a
//! `manifest.md`. CSV is used so the committed DB renders as a table and diffs
//! cleanly on GitHub. The `compute.csv` `key` column is the `TorchCallInfo`
//! serialized as JSON, then compacted (see `compact_value`): tensor objects
//! `{shape,dtype}` become `[code,[dims]]`, and runs of identical tensors in a
//! list collapse to `{"R":[k,[period]]}`. It still round-trips losslessly via
//! serde, and stays plain JSON so the Python `bench.py` reads it with the stdlib
//! (it applies the same expand). `load` also accepts the older verbose form.

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Duration;

use cuda_call::CudaMemcpyKind;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::torch_call::TorchCallInfo;

pub const SCHEMA_VERSION: u32 = 1;

/// Key for the (currently uncached) flash-attention timing, mirroring the
/// arguments of `CudaEstimator::flash_attn`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FlashAttnKey {
    pub is_fwd: bool,
    pub is_bf16: bool,
    pub batch_size: i32,
    pub seqlen_q: i32,
    pub seqlen_k: i32,
    pub num_heads: i32,
    pub num_heads_k: i32,
    pub head_size: i32,
    pub window_size_left: i32,
    pub window_size_right: i32,
    pub is_causal: bool,
}

/// All timing tables, with `Duration` values. Persisted as CSV.
#[derive(Default)]
pub struct PerfDb {
    pub gpu_name: String,
    pub compute: HashMap<TorchCallInfo, Duration>,
    pub sequence: BTreeMap<u64, Vec<Duration>>,
    pub memcpy: HashMap<(CudaMemcpyKind, usize), Duration>,
    pub flash_attn: HashMap<FlashAttnKey, Duration>,
}

fn memcpy_kind_str(kind: CudaMemcpyKind) -> &'static str {
    match kind {
        CudaMemcpyKind::HostToHost => "HostToHost",
        CudaMemcpyKind::HostToDevice => "HostToDevice",
        CudaMemcpyKind::PinnedHostToDevice => "PinnedHostToDevice",
        CudaMemcpyKind::DeviceToHost => "DeviceToHost",
        CudaMemcpyKind::DeviceToPinnedHost => "DeviceToPinnedHost",
        CudaMemcpyKind::DeviceToDevice => "DeviceToDevice",
    }
}

fn memcpy_kind_from_str(s: &str) -> Option<CudaMemcpyKind> {
    Some(match s {
        "HostToHost" => CudaMemcpyKind::HostToHost,
        "HostToDevice" => CudaMemcpyKind::HostToDevice,
        "PinnedHostToDevice" => CudaMemcpyKind::PinnedHostToDevice,
        "DeviceToHost" => CudaMemcpyKind::DeviceToHost,
        "DeviceToPinnedHost" => CudaMemcpyKind::DeviceToPinnedHost,
        "DeviceToDevice" => CudaMemcpyKind::DeviceToDevice,
        _ => return None,
    })
}

/// `tch::Kind` serialized name <-> short code used by the compact tensor form
/// `[code,[dims]]`. Tensors whose dtype is not listed here keep the verbose
/// `{shape,dtype}` object, so the mapping stays bijective and round-trips.
const DTYPE_CODES: &[(&str, &str)] = &[
    ("Float", "f32"),
    ("Double", "f64"),
    ("Half", "f16"),
    ("BFloat16", "bf16"),
    ("Bool", "b8"),
    ("Int", "i32"),
    ("Int64", "i64"),
    ("Int16", "i16"),
    ("Int8", "i8"),
    ("Uint8", "u8"),
];

fn dtype_to_code(name: &str) -> Option<&'static str> {
    DTYPE_CODES.iter().find(|(n, _)| *n == name).map(|(_, c)| *c)
}

fn dtype_from_code(code: &str) -> Option<&'static str> {
    DTYPE_CODES.iter().find(|(_, c)| *c == code).map(|(n, _)| *n)
}

/// If `items` is a single period repeated k>1 times, return `(k, period)`.
fn find_period(items: &[Value]) -> Option<(usize, Vec<Value>)> {
    let n = items.len();
    for p in 1..=n / 2 {
        if n % p == 0 && (0..n).all(|i| items[i] == items[i % p]) {
            return Some((n / p, items[..p].to_vec()));
        }
    }
    None
}

/// Compact a `TorchCallInfo` JSON value for storage. Lossless; `expand_value`
/// inverts it. Two shorthands: a tensor object `{"shape":S,"dtype":"Float"}`
/// becomes `["f32",S]`, and an all-tensor list that is a repeated period becomes
/// `{"R":[k,[period]]}` (the foreach optimizer ops list 1000s of tensors with a
/// tiny period). Note: relies on no `TorchCallInfo` variant producing a 2-tuple
/// `[<string>, <array>]` body (it would alias a compact tensor) -- guarded by the
/// round-trip test.
fn compact_value(v: Value) -> Value {
    match v {
        Value::Object(map) => {
            if map.len() == 2 {
                if let (Some(shape @ Value::Array(_)), Some(Value::String(dt))) =
                    (map.get("shape"), map.get("dtype"))
                {
                    if let Some(code) = dtype_to_code(dt) {
                        return Value::Array(vec![Value::String(code.to_string()), shape.clone()]);
                    }
                }
            }
            Value::Object(map.into_iter().map(|(k, val)| (k, compact_value(val))).collect())
        }
        Value::Array(arr) => {
            let items: Vec<Value> = arr.into_iter().map(compact_value).collect();
            let all_tensors = items.len() > 1
                && items.iter().all(|it| {
                    matches!(it, Value::Array(a) if a.len() == 2 && a[0].is_string() && a[1].is_array())
                });
            if all_tensors {
                if let Some((k, period)) = find_period(&items) {
                    return serde_json::json!({ "R": [k, period] });
                }
            }
            Value::Array(items)
        }
        other => other,
    }
}

/// Inverse of `compact_value`. Also tolerates the legacy verbose form (an old
/// committed DB whose tensors are still `{shape,dtype}` objects loads unchanged).
fn expand_value(v: Value) -> Value {
    match v {
        Value::Object(map) => {
            if map.len() == 1 {
                if let Some(Value::Array(rarr)) = map.get("R") {
                    if let (Some(k), Some(period)) =
                        (rarr.first().and_then(Value::as_u64), rarr.get(1).and_then(Value::as_array))
                    {
                        let mut out = Vec::with_capacity(k as usize * period.len());
                        for _ in 0..k {
                            for e in period {
                                out.push(expand_value(e.clone()));
                            }
                        }
                        return Value::Array(out);
                    }
                }
            }
            Value::Object(map.into_iter().map(|(k, val)| (k, expand_value(val))).collect())
        }
        Value::Array(arr) => {
            let tensor_dtype = if arr.len() == 2 && arr[1].is_array() {
                arr[0].as_str().and_then(dtype_from_code)
            } else {
                None
            };
            if let Some(name) = tensor_dtype {
                let shape = arr.into_iter().nth(1).unwrap();
                return serde_json::json!({ "shape": shape, "dtype": name });
            }
            Value::Array(arr.into_iter().map(expand_value).collect())
        }
        other => other,
    }
}

impl PerfDb {
    /// Load a DB from `dir`. Missing table files are treated as empty; a missing
    /// directory is an error (the caller asked to replay a non-existent DB).
    pub fn load(dir: &Path) -> Result<Self, Box<dyn Error>> {
        if !dir.is_dir() {
            return Err(format!("perf-db directory not found: {}", dir.display()).into());
        }
        let mut db = PerfDb::default();

        let compute_path = dir.join("compute.csv");
        if compute_path.exists() {
            let mut r = csv::Reader::from_path(&compute_path)?;
            for rec in r.records() {
                let rec = rec?;
                // New layout: key,nanos. Legacy layout: op,key,nanos.
                let (key_str, nanos_str) = if rec.len() >= 3 {
                    (&rec[1], &rec[2])
                } else {
                    (&rec[0], &rec[1])
                };
                let value: Value = serde_json::from_str(key_str)?;
                let key: TorchCallInfo = serde_json::from_value(expand_value(value))?;
                db.compute.insert(key, Duration::from_nanos(nanos_str.parse()?));
            }
        }

        let memcpy_path = dir.join("memcpy.csv");
        if memcpy_path.exists() {
            let mut r = csv::Reader::from_path(&memcpy_path)?;
            for rec in r.records() {
                let rec = rec?;
                // columns: kind, size_bytes, nanos
                let kind = memcpy_kind_from_str(&rec[0])
                    .ok_or_else(|| format!("unknown memcpy kind: {}", &rec[0]))?;
                let size: usize = rec[1].parse()?;
                db.memcpy
                    .insert((kind, size), Duration::from_nanos(rec[2].parse()?));
            }
        }

        let flash_path = dir.join("flash_attn.csv");
        if flash_path.exists() {
            let mut r = csv::Reader::from_path(&flash_path)?;
            for rec in r.records() {
                let rec = rec?;
                // columns: is_fwd,is_bf16,batch,seqlen_q,seqlen_k,num_heads,num_heads_k,head_size,win_left,win_right,is_causal,nanos
                let key = FlashAttnKey {
                    is_fwd: rec[0].parse()?,
                    is_bf16: rec[1].parse()?,
                    batch_size: rec[2].parse()?,
                    seqlen_q: rec[3].parse()?,
                    seqlen_k: rec[4].parse()?,
                    num_heads: rec[5].parse()?,
                    num_heads_k: rec[6].parse()?,
                    head_size: rec[7].parse()?,
                    window_size_left: rec[8].parse()?,
                    window_size_right: rec[9].parse()?,
                    is_causal: rec[10].parse()?,
                };
                db.flash_attn
                    .insert(key, Duration::from_nanos(rec[11].parse()?));
            }
        }

        let seq_path = dir.join("sequence.csv");
        if seq_path.exists() {
            let mut r = csv::Reader::from_path(&seq_path)?;
            for rec in r.records() {
                let rec = rec?;
                // columns: seq_hash, nanos (";"-joined)
                let seq_hash: u64 = rec[0].parse()?;
                let durs = rec[1]
                    .split(';')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u64>().map(Duration::from_nanos))
                    .collect::<Result<Vec<_>, _>>()?;
                db.sequence.insert(seq_hash, durs);
            }
        }

        let manifest = dir.join("manifest.md");
        if let Ok(text) = fs::read_to_string(&manifest) {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("- **GPU:** ") {
                    db.gpu_name = rest.trim().to_string();
                }
            }
        }

        Ok(db)
    }

    /// Write the DB to `dir` as CSV tables + a markdown manifest, creating `dir`.
    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(dir)?;

        // compute.csv (key = compacted TorchCallInfo JSON; see compact_value)
        {
            let mut w = csv::Writer::from_path(dir.join("compute.csv"))?;
            w.write_record(["key", "nanos"])?;
            let mut rows: Vec<(String, u128)> = self
                .compute
                .iter()
                .map(|(k, v)| {
                    let compact = compact_value(serde_json::to_value(k).unwrap());
                    (serde_json::to_string(&compact).unwrap(), v.as_nanos())
                })
                .collect();
            rows.sort(); // stable, diff-friendly ordering
            for (key, nanos) in rows {
                w.write_record([key, nanos.to_string()])?;
            }
            w.flush()?;
        }

        // memcpy.csv
        {
            let mut w = csv::Writer::from_path(dir.join("memcpy.csv"))?;
            w.write_record(["kind", "size_bytes", "nanos"])?;
            let mut rows: Vec<(&str, usize, u128)> = self
                .memcpy
                .iter()
                .map(|((kind, size), v)| (memcpy_kind_str(*kind), *size, v.as_nanos()))
                .collect();
            rows.sort();
            for (kind, size, nanos) in rows {
                w.write_record([kind.to_string(), size.to_string(), nanos.to_string()])?;
            }
            w.flush()?;
        }

        // flash_attn.csv (omitted entirely when empty -- e.g. presets whose
        // attention is simulated via ordinary compute ops; load() tolerates the
        // missing file. Remove any stale copy so a re-record cleans up.)
        let flash_path = dir.join("flash_attn.csv");
        if self.flash_attn.is_empty() {
            let _ = fs::remove_file(&flash_path);
        } else {
            let mut w = csv::Writer::from_path(&flash_path)?;
            w.write_record([
                "is_fwd",
                "is_bf16",
                "batch",
                "seqlen_q",
                "seqlen_k",
                "num_heads",
                "num_heads_k",
                "head_size",
                "win_left",
                "win_right",
                "is_causal",
                "nanos",
            ])?;
            let mut keys: Vec<&FlashAttnKey> = self.flash_attn.keys().collect();
            keys.sort_by_key(|k| {
                (
                    k.is_fwd,
                    k.batch_size,
                    k.seqlen_q,
                    k.seqlen_k,
                    k.num_heads,
                    k.head_size,
                )
            });
            for k in keys {
                let nanos = self.flash_attn[k].as_nanos();
                w.write_record([
                    k.is_fwd.to_string(),
                    k.is_bf16.to_string(),
                    k.batch_size.to_string(),
                    k.seqlen_q.to_string(),
                    k.seqlen_k.to_string(),
                    k.num_heads.to_string(),
                    k.num_heads_k.to_string(),
                    k.head_size.to_string(),
                    k.window_size_left.to_string(),
                    k.window_size_right.to_string(),
                    k.is_causal.to_string(),
                    nanos.to_string(),
                ])?;
            }
            w.flush()?;
        }

        // sequence.csv (omitted entirely when empty -- perf-db modes force
        // single-op timing, so it is empty for the recorded presets; load()
        // tolerates the missing file. Remove any stale copy.)
        let seq_path = dir.join("sequence.csv");
        if self.sequence.is_empty() {
            let _ = fs::remove_file(&seq_path);
        } else {
            let mut w = csv::Writer::from_path(&seq_path)?;
            w.write_record(["seq_hash", "nanos"])?;
            for (hash, durs) in &self.sequence {
                let nanos = durs
                    .iter()
                    .map(|d| d.as_nanos().to_string())
                    .collect::<Vec<_>>()
                    .join(";");
                w.write_record([hash.to_string(), nanos])?;
            }
            w.flush()?;
        }

        // manifest.md
        let manifest = format!(
            "# Phantora performance database\n\n\
             - **GPU:** {}\n\
             - **Schema version:** {}\n\
             - **compute entries:** {}\n\
             - **sequence entries:** {}\n\
             - **memcpy entries:** {}\n\
             - **flash_attn entries:** {}\n\n\
             Recorded with `--record-perf-db <dir>`. Replay with `--perf-db <dir>` (no GPU \
             required). Each `*.csv` is one timing table (values in nanoseconds); the \
             `compute.csv` `key` is the `TorchCallInfo` as compacted JSON (tensors \
             as `[code,[dims]]`, repeated-tensor lists as `{{\"R\":[k,[period]]}}`).\n",
            self.gpu_name,
            SCHEMA_VERSION,
            self.compute.len(),
            self.sequence.len(),
            self.memcpy.len(),
            self.flash_attn.len(),
        );
        fs::write(dir.join("manifest.md"), manifest)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch_call::TensorInfo;
    use tch::Kind;

    fn ti(shape: &[i64], dtype: Kind) -> TensorInfo {
        TensorInfo {
            shape: shape.to_vec(),
            dtype,
        }
    }

    #[test]
    fn perf_db_csv_round_trip() {
        let mut db = PerfDb::default();
        db.gpu_name = "NVIDIA L40S".to_string();

        // A spread of TorchCallInfo shapes: tuple, Option, scalar, and struct variants.
        db.compute.insert(
            TorchCallInfo::MM(ti(&[1024, 1024], Kind::Float), ti(&[1024, 1024], Kind::Float)),
            Duration::from_nanos(1_234_000),
        );
        db.compute.insert(
            TorchCallInfo::Linear(
                ti(&[8, 1024, 4096], Kind::BFloat16),
                ti(&[4096, 4096], Kind::BFloat16),
                Some(ti(&[4096], Kind::BFloat16)),
            ),
            Duration::from_nanos(5_678_000),
        );
        db.compute.insert(
            TorchCallInfo::Softmax(ti(&[1, 64, 1024, 1024], Kind::Float), -1),
            Duration::from_nanos(42_000),
        );
        db.compute.insert(
            TorchCallInfo::SDPA {
                q: ti(&[2, 32, 1024, 128], Kind::BFloat16),
                k: ti(&[2, 32, 1024, 128], Kind::BFloat16),
                v: ti(&[2, 32, 1024, 128], Kind::BFloat16),
                causal: true,
                gqa: false,
            },
            Duration::from_nanos(9_000),
        );

        // Foreach op: interleaved periodic tensor lists exercise the {"R":...}
        // run-length shorthand and must round-trip in exact order.
        db.compute.insert(
            TorchCallInfo::ForeachAddCMul_(
                vec![
                    ti(&[1572864], Kind::Float),
                    ti(&[3145728], Kind::Float),
                    ti(&[1572864], Kind::Float),
                    ti(&[3145728], Kind::Float),
                ],
                vec![ti(&[1572864], Kind::Float), ti(&[3145728], Kind::Float)],
                vec![ti(&[16], Kind::Float)],
            ),
            Duration::from_nanos(28_800),
        );

        db.memcpy
            .insert((CudaMemcpyKind::HostToDevice, 1_048_576), Duration::from_nanos(45_600));
        db.memcpy
            .insert((CudaMemcpyKind::DeviceToDevice, 4096), Duration::from_nanos(1_200));

        db.flash_attn.insert(
            FlashAttnKey {
                is_fwd: true,
                is_bf16: true,
                batch_size: 8,
                seqlen_q: 4096,
                seqlen_k: 4096,
                num_heads: 32,
                num_heads_k: 32,
                head_size: 128,
                window_size_left: -1,
                window_size_right: -1,
                is_causal: true,
            },
            Duration::from_nanos(890_000),
        );

        db.sequence.insert(
            0x1234_5678_9abc_def0,
            vec![Duration::from_nanos(10), Duration::from_nanos(20), Duration::from_nanos(30)],
        );

        let dir = std::env::temp_dir().join(format!("phantora_perfdb_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        db.save(&dir).expect("save");
        let loaded = PerfDb::load(&dir).expect("load");
        fs::remove_dir_all(&dir).ok();

        assert_eq!(loaded.gpu_name, db.gpu_name);
        assert_eq!(loaded.compute, db.compute);
        assert_eq!(loaded.memcpy, db.memcpy);
        assert_eq!(loaded.flash_attn, db.flash_attn);
        assert_eq!(loaded.sequence, db.sequence);
    }

    // Re-save an existing DB through the current codec (e.g. to migrate it to a
    // new compact format) and verify it reloads identically. Ignored by default;
    // run with `PERFDB_DIR=<dir> cargo test -p phantora migrate_perf_db -- --ignored`.
    #[test]
    #[ignore]
    fn migrate_perf_db() {
        let dir = std::path::PathBuf::from(std::env::var("PERFDB_DIR").expect("set PERFDB_DIR"));
        let before = PerfDb::load(&dir).expect("load");
        before.save(&dir).expect("save");
        let after = PerfDb::load(&dir).expect("reload");
        assert_eq!(before.compute, after.compute);
        assert_eq!(before.memcpy, after.memcpy);
        assert_eq!(before.flash_attn, after.flash_attn);
        assert_eq!(before.sequence, after.sequence);
        assert_eq!(before.gpu_name, after.gpu_name);
        eprintln!("migrated {} compute entries in {}", after.compute.len(), dir.display());
    }
}
