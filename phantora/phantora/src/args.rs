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
    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub disable_sequence_call: bool,
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
