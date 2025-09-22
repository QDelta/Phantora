#[cfg(test)]
use netsim::bandwidth::BandwidthTrait;
use netsim::cluster::{Cluster, Node, NodeType};
use netsim::simulator::{Executor, Simulator};
use netsim::{Flow, Trace, TraceRecord};

#[test]
fn toy1() {
    env_logger::init();

    let nodes = vec![
        ("a1", 3),
        ("a2", 3),
        ("a3", 3),
        ("a5", 3),
        ("a6", 3),
        ("vs1", 2),
        ("vs2", 2),
        ("cloud", 1),
    ]
    .into_iter()
    .map(|(n, depth)| {
        Node::new(
            n,
            depth,
            if depth == 3 {
                NodeType::Host
            } else {
                NodeType::Switch
            },
        )
    })
    .collect();

    let mut cluster = Cluster::from_nodes(nodes);

    vec![
        ("a1", "vs1", 20.gbps()),
        ("a2", "vs1", 10.gbps()),
        ("a3", "vs1", 9.gbps()),
        ("a5", "vs2", 10.gbps()),
        ("a6", "vs2", 5.gbps()),
        ("vs1", "cloud", 35.gbps()),
        ("vs2", "cloud", 15.gbps()),
    ]
    .into_iter()
    .for_each(|args| cluster.add_link_by_name(args.1, args.0, args.2));

    let mut trace = Trace::new();
    let records: Vec<TraceRecord> = vec![
        (0, 1e6 as usize, "a1", "a5"),
        (0, 1e6 as usize, "a2", "a6"),
        (1000000, 1e6 as usize, "a2", "a3"),
    ]
    .into_iter()
    .map(|args| TraceRecord::new(args.0, Flow::new(args.1, args.2, args.3, None), None))
    .collect();
    records.into_iter().for_each(|r| trace.add_record(r));

    let mut simulator = Simulator::new(cluster);
    let output = simulator.run_with_trace(trace);
    println!("{:#?}", output);
    assert_eq!(output.recs[0].dura, Some(800_000));
    assert_eq!(output.recs[1].dura, Some(1_600_000));
    assert_eq!(
        output.recs[2].dura,
        Some(((600. + (10. / 16.) * 1600. * (5. / 9.)) * 1e3f64).round() as u64)
    );
}
