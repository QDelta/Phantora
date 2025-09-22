use std::path;

use serde::{Deserialize, Serialize};

use crate::architecture::{
    build_arbitrary_cluster, build_fatree_fake, build_twolayer_multipath_cluster, TopoArgs,
};
use crate::bandwidth::BandwidthTrait;
use crate::cluster::{Cluster, Topology};
use crate::simulator::SimulatorSetting;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Output path of the figure
    #[serde(default)]
    pub directory: Option<path::PathBuf>,

    /// Simulator settings
    pub simulator: SimulatorSetting,

    /// A list of hostnames, the process of host_mapping[i] will be mapped
    /// to "host_i" in the network topology.
    pub host_mapping: Vec<String>,

    /// Topology settings
    pub topology: TopoArgs,
}

pub fn read_config<T: serde::de::DeserializeOwned, P: AsRef<path::Path>>(path: P) -> T {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("fail to open file");
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    toml::from_str(&content).expect("parse failed")
}

pub fn build_cloud(setting: &Config) -> Cluster {
    match setting.topology {
        TopoArgs::FatTree {
            nports,
            bandwidth,
            oversub_ratio,
        } => {
            let cluster = build_fatree_fake(nports, bandwidth.gbps(), oversub_ratio);
            assert_eq!(cluster.num_hosts(), nports * nports * nports / 4);
            cluster
        }
        TopoArgs::Arbitrary {
            nracks,
            rack_size,
            host_bw,
            rack_bw,
        } => build_arbitrary_cluster(nracks, rack_size, host_bw.gbps(), rack_bw.gbps()),
        TopoArgs::TwoLayerMultiPath {
            nspines,
            nracks,
            rack_size,
            host_bw,
            rack_uplink_port_bw,
            ..
        } => build_twolayer_multipath_cluster(
            nspines,
            nracks,
            rack_size,
            host_bw.gbps(),
            rack_uplink_port_bw.gbps(),
        ),
    }
}
