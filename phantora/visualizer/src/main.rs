use std::fs;
use std::io;
use std::io::{Read, Write};
use std::path;
use std::sync::OnceLock;

use phantora::event_queue::{Action, EventId};
use phantora::simulator::ComputeMeta;
use phantora::timeline::Record;

use clap::Parser;
use fnv::FnvHashMap as HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Parser)]
#[command(
    name = "Phantora Timeline Visualizer",
    about = "Phantora Timeline Visualizer"
)]
pub struct Args {
    /// Path to a file to collect event timeline trace for visualization.
    #[arg(short, long)]
    pub timeline_file: path::PathBuf,

    /// Output file path
    #[arg(short, long)]
    pub output: path::PathBuf,
}

pub fn get_args() -> &'static Args {
    static ARGS: OnceLock<Args> = OnceLock::new();

    let args: &Args = ARGS.get_or_init(|| {
        let args = Args::parse();
        args
    });

    args
}

struct Logger<T> {
    last_log: Option<T>,
}

impl<T: PartialEq + std::fmt::Display> Logger<T> {
    fn new() -> Self {
        Logger { last_log: None }
    }

    fn log(&mut self, prefix: &str, val: T, suffix: &str) {
        if self.last_log.as_ref() != Some(&val) {
            eprint!("{}{}{}", prefix, val, suffix);
            self.last_log = Some(val);
        }
    }
}

struct TimelineReader {
    reader: io::BufReader<fs::File>,
    timeline_pos: u64,
    timeline_size: u64,
}

impl TimelineReader {
    fn new<P: AsRef<path::Path>>(path: P) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let timeline_size = file.metadata()?.len();
        let reader = io::BufReader::new(file);
        Ok(TimelineReader {
            reader,
            timeline_pos: 0,
            timeline_size,
        })
    }

    fn next_record(&mut self) -> io::Result<Record> {
        let mut len_buf = [0u8; 8];
        self.reader.read_exact(&mut len_buf)?;
        let len = u64::from_be_bytes(len_buf) as usize;
        let mut buf = Vec::with_capacity(len);
        unsafe { buf.set_len(len) };
        self.reader.read_exact(&mut buf)?;
        self.timeline_pos += 8 + len as u64;
        Ok(bincode::deserialize(&buf).unwrap())
    }

    fn percent(&self) -> u64 {
        self.timeline_pos * 100 / self.timeline_size
    }
}

// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
// {
//   "name": "myName",
//   "cat": "category,list",
//   "ph": "B",
//   "ts": 12345,
//   "pid": 123,
//   "tid": 456,
//   "args": {
//     "someArg": 1,
//     "anotherArg": {
//       "value": "my value"
//     }
//   }
// }

mod chrome {
    use super::*;

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, strum::Display)]
    pub enum Phase {
        B,
        E,
        X,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Event {
        pub name: String,
        pub cat: String,
        pub ph: Phase,
        pub ts: i64,
        pub dur: i64,
        pub pid: u32,
        pub tid: u32,
        pub args: Action,
    }
}

#[derive(Debug, Clone)]
pub struct Span {
    pub id: EventId,
    pub start: i64,
    pub end: i64,
    pub action: Action,
}

impl std::hash::Hash for Span {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Default)]
pub struct Timeline {
    db: HashMap<EventId, Span>,
    min_ts: i64,
}

impl Timeline {
    fn new() -> Self {
        Timeline {
            db: Default::default(),
            min_ts: i64::max_value(),
        }
    }

    fn min_ts(&self) -> i64 {
        self.min_ts
    }

    fn update(&mut self, rec: &Record) {
        match rec {
            Record::Action { id, action } => {
                let id = *id;
                let present = self
                    .db
                    .insert(
                        id,
                        Span {
                            id,
                            start: 0,
                            end: 0,
                            action: action.clone(),
                        },
                    )
                    .is_none();
                assert!(present);
            }
            &Record::Span { id, start, end } => {
                self.min_ts = self.min_ts.min(start);
                self.db.entry(id).and_modify(|e| {
                    e.start = start;
                    e.end = end;
                });
                // .or_insert_with(|| panic!("all span has an action"));
                if !self.db.contains_key(&id) {
                    assert_eq!(start, end, "id: {}", id);
                }
            }
        }
    }
}

fn main() {
    let mut timeline = Timeline::new();
    let timeline_file = &get_args().timeline_file;
    let mut logger = Logger::new();
    for step in [0, 1] {
        let mut reader = TimelineReader::new(timeline_file).unwrap();
        loop {
            let result = reader.next_record();
            match result {
                Ok(rec) => {
                    if step == 0 && matches!(rec, Record::Action { .. }) {
                        timeline.update(&rec);
                        logger.log(
                            "\rvisualizer reading actions: ",
                            reader.percent().to_string(),
                            "%",
                        );
                    }
                    if step == 1 && matches!(rec, Record::Span { .. }) {
                        timeline.update(&rec);
                        logger.log(
                            "\rvisualizer reading spans: ",
                            reader.percent().to_string(),
                            "%",
                        );
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(err) => panic!("error: {}", err),
            }
        }
        eprint!("\n");
    }

    // create Events
    let mut host_rank = HashMap::default();
    let mut host_rank_cnt = 0;
    let mut cnt = 0;
    let total_events = timeline.db.len();
    let mut f = fs::File::create(&get_args().output).unwrap();
    f.write(b"[\n").unwrap();
    for (_id, span) in timeline.db.iter() {
        if span.start == 0 {
            // panic!("all action has a span: {:?}", span);
        }

        let ev = match &span.action {
            Action::Computation(node_id, _dura, meta) => {
                let name = match meta {
                    ComputeMeta::Cuda(cuda_call) => cuda_call.to_string(),
                    ComputeMeta::Torch(torch_call) => torch_call.info.to_string(),
                };
                let cat = "Computation".to_owned();
                let ph = chrome::Phase::X;
                let ts = span.start - timeline.min_ts();
                let dur = span.end - span.start;
                let pid = node_id.0.pid;
                let tid = node_id.0.pid;
                let args = span.action.clone();
                chrome::Event {
                    name,
                    cat,
                    ph,
                    ts,
                    dur,
                    pid,
                    tid,
                    args,
                }
            }
            Action::Communication(flow, meta) => {
                let stream = meta.get_cuda_stream().unwrap();
                let name = meta.to_string();
                let cat = "Communication".to_owned();
                let ph = chrome::Phase::X;
                let ts = span.start - timeline.min_ts();
                let dur = span.end - span.start;
                let pid = *host_rank.entry(flow.src.clone()).or_insert_with(|| {
                    host_rank_cnt += 1;
                    host_rank_cnt - 1
                });
                let tid = (((stream.device as u64) << 32) | (stream.id as u64)) as _;
                let args = span.action.clone();
                chrome::Event {
                    name,
                    cat,
                    ph,
                    ts,
                    dur,
                    pid,
                    tid,
                    args,
                }
            }
        };

        if cnt > 0 {
            f.write(b",\n").unwrap();
        }
        serde_json::to_writer(&mut f, &ev).unwrap();
        cnt += 1;
        logger.log(
            "\rvisualizer processing events: ",
            format!("{:.1}", cnt as f64 * 100.0 / total_events as f64),
            "%",
        );
    }
    eprint!("\n");
    f.write(b"\n]").unwrap();
    f.flush().unwrap();
}
