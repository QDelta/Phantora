use crate::bandwidth::Bandwidth;
use crate::cluster::{Cluster, Node, NodeType};
use crate::LoadBalancerType;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
#[serde(tag = "type", content = "args")]
pub enum TopoArgs {
    /// FatTree, parameters include the number of ports of each switch, bandwidth, and oversubscription ratio
    FatTree {
        /// Set the the number of ports
        nports: usize,
        /// Bandwidth of a host, in Gbps
        bandwidth: f64,
        /// Oversubscription ratio
        oversub_ratio: f64,
    },

    /// 'Arbitrary' cluster, parameters include the number of racks and rack_size, host_bw, and rack_bw
    Arbitrary {
        /// Specify the number of racks
        nracks: usize,
        /// Specify the number of hosts within one rack
        rack_size: usize,
        /// Bandwidth of a host, in Gbps
        host_bw: f64,
        /// Bandwidth of a ToR switch, in Gbps
        rack_bw: f64,
    },

    /// Two-layer (number of switch layers) Multipath, where the two layer of switches are fully
    /// connected
    TwoLayerMultiPath {
        /// Specify the number of spine switches
        nspines: usize,
        /// Specify the number of racks (or leaf switches)
        nracks: usize,
        /// Specify the number of hosts within one rack
        rack_size: usize,
        /// Bandwidth of a host, in Gbps
        host_bw: f64,
        /// Bandwidth of each link between a spine and a leaf, in Gbps.
        rack_uplink_port_bw: f64,
        /// Type of network load balancer
        load_balancer_type: LoadBalancerType,
    },
}

impl TopoArgs {
    pub fn by_arbitrary_topo(&self) -> TopoArgs {
        match self {
            a @ Self::Arbitrary { .. } => a.clone(),
            &Self::FatTree {
                nports,
                bandwidth,
                oversub_ratio,
            } => {
                assert!(
                    nports % 2 == 0,
                    "the number of ports of a switch is required to be even for FatTree"
                );

                let k = nports;
                let num_pods = k;
                let _num_cores = k * k / 4;
                let num_aggs_in_pod = k / 2;
                let _num_aggs = num_pods * num_aggs_in_pod;
                let num_edges_in_pod = k / 2;
                let num_edges = num_pods * num_edges_in_pod;
                let num_hosts_under_edge = k / 2;
                let _num_hosts = num_edges * num_hosts_under_edge;
                Self::Arbitrary {
                    nracks: num_edges,
                    rack_size: num_hosts_under_edge,
                    host_bw: bandwidth,
                    rack_bw: bandwidth * num_hosts_under_edge as f64 / oversub_ratio,
                }
            }
            &Self::TwoLayerMultiPath {
                nspines,
                nracks,
                rack_size,
                host_bw,
                rack_uplink_port_bw,
                ..
            } => Self::Arbitrary {
                nracks,
                rack_size,
                host_bw,
                rack_bw: rack_uplink_port_bw * nspines as f64,
            },
        }
    }

    /// Number of spines/roots switch in the cluster
    pub fn nroots(&self) -> usize {
        match self {
            &Self::Arbitrary { .. } | &Self::FatTree { .. } => 1,
            &Self::TwoLayerMultiPath { nspines, .. } => nspines,
        }
    }

    /// Number of racks in the cluster
    pub fn nracks(&self) -> usize {
        match self {
            &Self::Arbitrary { nracks, .. } => nracks,
            &Self::FatTree { .. } => self.by_arbitrary_topo().nracks(),
            &Self::TwoLayerMultiPath { .. } => self.by_arbitrary_topo().nracks(),
        }
    }

    /// Number of hosts in a rack
    pub fn rack_size(&self) -> usize {
        match self {
            &Self::Arbitrary { rack_size, .. } => rack_size,
            &Self::FatTree { .. } => self.by_arbitrary_topo().rack_size(),
            &Self::TwoLayerMultiPath { .. } => self.by_arbitrary_topo().rack_size(),
        }
    }

    /// Host outbound/inbound bandwidth
    pub fn host_bw(&self) -> f64 {
        match self {
            &Self::Arbitrary { host_bw, .. } => host_bw,
            &Self::FatTree { .. } => self.by_arbitrary_topo().host_bw(),
            &Self::TwoLayerMultiPath { .. } => self.by_arbitrary_topo().host_bw(),
        }
    }

    /// Aggregated rack outbound/inbound bandwidth
    pub fn rack_bw(&self) -> f64 {
        match self {
            &Self::Arbitrary { rack_bw, .. } => rack_bw,
            &Self::FatTree { bandwidth, .. } => bandwidth,
            &Self::TwoLayerMultiPath { .. } => self.by_arbitrary_topo().rack_bw(),
        }
    }
}

impl std::fmt::Display for TopoArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopoArgs::FatTree {
                nports,
                bandwidth,
                oversub_ratio,
            } => write!(f, "fattree_{}_{}g_{:.2}", nports, bandwidth, oversub_ratio),
            TopoArgs::Arbitrary {
                nracks,
                rack_size,
                host_bw,
                rack_bw,
            } => write!(
                f,
                "arbitrary_{}_{}_{}g_{}g",
                nracks, rack_size, host_bw, rack_bw,
            ),
            TopoArgs::TwoLayerMultiPath {
                nspines,
                nracks,
                rack_size,
                host_bw,
                rack_uplink_port_bw,
                load_balancer_type,
            } => write!(
                f,
                "twolayermultipath_{}_{}_{}_{}g_{}g_{:?}",
                nspines, nracks, rack_size, host_bw, rack_uplink_port_bw, load_balancer_type,
            ),
        }
    }
}

pub fn build_fatree_fake(nports: usize, bw: Bandwidth, oversub_ratio: f64) -> Cluster {
    assert!(
        nports % 2 == 0,
        "the number of ports of a switch is required to be even for FatTree"
    );

    let k = nports;
    let num_pods = k;
    let _num_cores = k * k / 4;
    let num_aggs_in_pod = k / 2;
    let _num_aggs = num_pods * num_aggs_in_pod;
    let num_edges_in_pod = k / 2;
    let num_edges = num_pods * num_edges_in_pod;
    let num_hosts_under_edge = k / 2;
    let _num_hosts = num_edges * num_hosts_under_edge;

    build_arbitrary_cluster(
        num_edges,
        num_hosts_under_edge,
        bw,
        bw * num_hosts_under_edge / oversub_ratio,
    )
}

pub fn build_arbitrary_cluster(
    nracks: usize,
    rack_size: usize,
    host_bw: Bandwidth,
    rack_bw: Bandwidth,
) -> Cluster {
    let mut cluster = Cluster::new();
    let cloud = Node::new("cloud", 1, NodeType::Switch);
    cluster.add_node(cloud);

    let mut host_id = 0;
    for i in 0..nracks {
        let tor_name = format!("tor_{}", i);
        let tor = Node::new(&tor_name, 2, NodeType::Switch);
        cluster.add_node(tor);
        cluster.add_link_by_name("cloud", &tor_name, rack_bw);

        for j in host_id..host_id + rack_size {
            let host_name = format!("host_{}", j);
            let host = Node::new(&host_name, 3, NodeType::Host);
            cluster.add_node(host);
            cluster.add_link_by_name(&tor_name, &host_name, host_bw);
        }

        host_id += rack_size;
    }

    cluster
}

pub fn build_twolayer_multipath_cluster(
    nspines: usize,
    nracks: usize,
    rack_size: usize,
    host_bw: Bandwidth,
    rack_uplink_port_bw: Bandwidth,
) -> Cluster {
    let mut cluster = Cluster::new();
    let mut host_cnt = 0;

    for i in 0..nspines {
        let spine_name = format!("spine_{}", i);
        let spine = Node::new(&spine_name, 1, NodeType::Switch);
        cluster.add_node(spine);
    }

    for i in 0..nracks {
        // we use tor_{} here because this assumption is used across this codebase.
        // use some other name may break the code unexpected
        let tor_name = format!("tor_{}", i);
        let tor = Node::new(&tor_name, 2, NodeType::Switch);
        cluster.add_node(tor);

        for j in 0..nspines {
            cluster.add_link_by_name(&format!("spine_{}", j), &tor_name, rack_uplink_port_bw);
        }

        for j in host_cnt..host_cnt + rack_size {
            let host_name = format!("host_{}", j);
            let host = Node::new(&host_name, 3, NodeType::Host);
            cluster.add_node(host);
            cluster.add_link_by_name(&tor_name, &host_name, host_bw);
        }

        host_cnt += rack_size;
    }

    cluster
}
