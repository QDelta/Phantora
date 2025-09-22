#[cfg(test)]
use netsim::bandwidth::{Bandwidth, BandwidthTrait};
use netsim::cluster::{Cluster, Node, NodeType};
use netsim::simulator::{Executor, Simulator};
use netsim::{Flow, Trace, TraceRecord};

#[allow(unused)]
type Layer<'a> = &'a [Node];

#[allow(unused)]
trait Network {
    fn add_layer(&mut self, layer: Layer);
    fn fully_connect(&mut self, layer1: Layer, layer2: Layer, bw: Bandwidth);
}

impl Network for Cluster {
    fn add_layer(&mut self, layer: Layer) {
        for n in layer {
            self.add_node(n.clone());
        }
    }

    fn fully_connect(&mut self, layer1: Layer, layer2: Layer, bw: Bandwidth) {
        for n1 in layer1 {
            for n2 in layer2 {
                self.add_link_by_name(&n1.name, &n2.name, bw);
            }
        }
    }
}

fn _build_fatree(nports: usize, bw: Bandwidth) -> Cluster {
    assert!(
        nports % 2 == 0,
        "the number of ports of a switch is required to be even"
    );
    let k = nports;
    let num_pods = k;
    let num_cores = k * k / 4;
    let num_aggs_in_pod = k / 2;
    let _num_aggs = num_pods * num_aggs_in_pod;
    let num_edges_in_pod = k / 2;
    let num_edges = num_pods * num_edges_in_pod;
    let num_hosts_under_edge = k / 2;
    let _num_hosts = num_edges * num_hosts_under_edge;

    let mut cluster = Cluster::new();

    // create core layer switches
    let cores: Vec<Node> = (0..num_cores)
        .map(|i| Node::new(&format!("core_{}", i), 1, NodeType::Switch))
        .collect();

    cluster.add_layer(&cores);

    let mut host_id = 0;
    // create each pods individually
    for i in 0..num_pods {
        // create aggregation layer switches
        let start = i * num_pods;
        let aggs: Vec<Node> = (start..start + num_aggs_in_pod)
            .map(|i| Node::new(&format!("agg_{}", i), 2, NodeType::Switch))
            .collect();

        // create edge layer switches
        let edges: Vec<Node> = (start..start + num_edges_in_pod)
            .map(|i| Node::new(&format!("edge_{}", i), 3, NodeType::Switch))
            .collect();

        cluster.add_layer(&aggs);
        cluster.add_layer(&edges);

        // fully connect
        cluster.fully_connect(&aggs, &edges, bw);

        for a in 0..aggs.len() {
            let agg = &aggs[a];
            for c in (0..cores.len()).step_by(k / 2) {
                let core = &cores[c];
                cluster.add_link_by_name(&core.name, &agg.name, bw);
            }
        }

        for sw in edges {
            // create hosts
            let hosts: Vec<Node> = (host_id..host_id + num_hosts_under_edge)
                .map(|i| Node::new(&format!("host_{}", i), 4, NodeType::Host))
                .collect();
            host_id += num_hosts_under_edge;

            cluster.add_layer(&hosts);
            cluster.fully_connect(&[sw], &hosts, bw);
        }
    }

    cluster
}

fn build_fatree_fake(nports: usize, bw: Bandwidth) -> Cluster {
    assert!(
        nports % 2 == 0,
        "the number of ports of a switch is required to be even"
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

    let mut cluster = Cluster::new();
    let cloud = Node::new(&format!("cloud"), 1, NodeType::Switch);
    cluster.add_node(cloud);

    let mut host_id = 0;
    for i in 0..num_edges {
        let tor_name = format!("tor_{}", i);
        let tor = Node::new(&tor_name, 2, NodeType::Switch);
        cluster.add_node(tor);
        cluster.add_link_by_name("cloud", &tor_name, bw * num_hosts_under_edge);

        for j in host_id..host_id + num_hosts_under_edge {
            let host_name = format!("host_{}", j);
            let host = Node::new(&host_name, 3, NodeType::Host);
            cluster.add_node(host);
            cluster.add_link_by_name(&tor_name, &host_name, bw);
        }

        host_id += num_hosts_under_edge;
    }

    cluster
}

fn alltoall_trace(num_hosts: usize, flow_size: usize) -> Trace {
    let mut trace = Trace::new();

    for i in 0..num_hosts {
        for j in 0..num_hosts {
            if i == j {
                continue;
            }
            let src = format!("host_{}", i);
            let dst = format!("host_{}", j);
            let flow = Flow::new(flow_size, &src, &dst, None);
            let rec = TraceRecord::new(0, flow, None);
            trace.add_record(rec);
        }
    }

    trace
}

#[test]
fn alltoall() {
    env_logger::init();

    let nports = 12;
    let cluster = build_fatree_fake(nports, 100.gbps());

    let nhosts = nports * nports * nports / 4;
    let trace = alltoall_trace(nhosts, 1_000_000);

    let mut simulator = Simulator::new(cluster);
    let _output = simulator.run_with_trace(trace);
    // println!("{:?}", output);
}
