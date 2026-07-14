use std::error::Error;
use std::path;
use std::sync::OnceLock;

use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(name = "Phantora Simulator", about = "Phantora Simulator")]
pub struct Args {
    /// The configuration file for the network simulator.
    #[arg(short = 'c', long = "netconfig")]
    pub net_config: path::PathBuf,

    /// Path to a file to collect event timeline trace for visualization.
    #[arg(long)]
    pub timeline_file: Option<path::PathBuf>,

    /// Core indices that are available for the applications, like "2,4-6,8" for cores 2,4,5,6,8.
    #[arg(long, value_parser = parse_cores)]
    pub available_cores: Option<std::vec::Vec<usize>>,

    /// disable sequence of calls optimization
    // Accept both spellings: clap's `long` default would kebab-case the field to
    // `--disable-sequence-call`, but the natural/documented spelling is the
    // underscore form. Make the underscore canonical and alias the kebab so
    // neither is rejected as an unexpected argument.
    #[arg(
        long = "disable_sequence_call",
        alias = "disable-sequence-call",
        action = clap::ArgAction::SetTrue
    )]
    pub disable_sequence_call: bool,

    /// Replay kernel timings from a performance database directory instead of
    /// profiling on a GPU. Skips all GPU/CUPTI init, so preset simulations run on
    /// a GPU-less machine. A missing (op, shape) is NOT fatal: it is charged zero
    /// time and written to <dir>.missing, which makes that run's numbers invalid
    /// (see the warning on exit) but lets one run discover every shape to profile.
    #[arg(long, conflicts_with = "record_perf_db")]
    pub perf_db: Option<path::PathBuf>,

    /// Record kernel timings to a performance database directory: profile on the
    /// GPU as usual, then write/merge the timing tables (CSV) on exit.
    #[arg(long)]
    pub record_perf_db: Option<path::PathBuf>,
}

fn parse_cores(s: &str) -> Result<Vec<usize>, Box<dyn Error + Send + Sync>> {
    let mut cores = vec![];
    for interval in s.split(',') {
        match interval.split('-').collect::<Vec<_>>().as_slice() {
            [c] => cores.push(c.parse()?),
            [start, end] => {
                let start = start.parse()?;
                let end = end.parse()?;
                for c in start..=end {
                    cores.push(c);
                }
            }
            _ => return Err(format!("Invalid core range {}", interval).into()),
        }
    }
    Ok(cores)
}

pub fn get_args() -> &'static Args {
    static ARGS: OnceLock<Args> = OnceLock::new();

    let args: &Args = ARGS.get_or_init(|| {
        let args = Args::parse();
        println!("args: {:#?}", args);
        args
    });

    args
}
