use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use fnv::FnvHashMap as HashMap;
use lazy_static::lazy_static;
use petgraph::{
    dot::Dot,
    graph::{EdgeIndex, EdgeIndices, Graph, NodeIndex},
};
use serde::{Deserialize, Serialize};

use crate::bandwidth::Bandwidth;
use crate::Flow;
use crate::LoadBalancer;

lazy_static! {
    static ref LINK_ID: AtomicUsize = AtomicUsize::new(0);
}

pub type LinkIx = EdgeIndex;
pub type NodeIx = NodeIndex;
pub type LinkIxIter = EdgeIndices;

use std::ops::{Index, IndexMut};

pub trait TopologyMultiPath: Topology {
    fn get_uplinks(&self, ix: NodeIx) -> std::slice::Iter<'_, LinkIx>;
    // fn find_all_links(&self, ix: NodeIx, iy: NodeIx) -> std::slice::Iter<LinkIx>;
    fn resolve_route_multipath(
        &self,
        flow: &Flow,
        load_balancer: &mut Option<Box<dyn LoadBalancer>>,
    ) -> Route;
}

pub trait Topology:
    Index<NodeIx, Output = Node> + Index<LinkIx, Output = Link> + IndexMut<LinkIx> + IndexMut<NodeIx>
{
    fn get_node_index(&self, name: &str) -> NodeIx;
    fn get_target(&self, ix: LinkIx) -> NodeIx;
    fn get_source(&self, ix: LinkIx) -> NodeIx;
    fn get_uplink(&self, ix: NodeIx) -> LinkIx;
    fn get_downlinks(&self, ix: NodeIx) -> std::slice::Iter<'_, LinkIx>;
    fn get_reverse_link(&self, ix: LinkIx) -> LinkIx;
    fn all_links(&self) -> EdgeIndices;
    fn find_link(&self, ix: NodeIx, iy: NodeIx) -> Option<LinkIx>;
    fn resolve_route(&self, src: &str, dst: &str, hint: &RouteHint) -> Route;
    fn num_hosts(&self) -> usize;
    fn num_switches(&self) -> usize;
    fn num_racks(&self) -> usize;
    /// do network translation
    fn translate(&self, vname: &str) -> String;
    fn to_dot(&self) -> Dot<'_, &Graph<Node, Link>>;
}

// We want to compare two topologies without some degree of tolerance.
// it is inappropriate to implement this comparison as PartialEq.
impl PartialEq for &dyn Topology {
    fn eq(&self, other: &&dyn Topology) -> bool {
        use crate::bandwidth::BandwidthTrait;

        let dump: HashMap<LinkIx, Link> = self
            .all_links()
            .map(|link_ix| (link_ix, self[link_ix].clone()))
            .collect();

        other.all_links().all(|link_ix| {
            let l1 = other[link_ix].clone();
            if let Some(l2) = dump.get(&link_ix) {
                let f = l1.bandwidth + 1.gbps() >= l2.bandwidth
                    && l2.bandwidth + 1.gbps() >= l1.bandwidth;
                if !f {
                    log::debug!("l1: {}, l2: {}", l1, l2);
                }
                f
            } else {
                false
            }
        })
    }
}

pub trait TopologyClone: Topology {
    fn into_box(&self) -> Box<dyn TopologyClone + '_>;
}

impl<T> TopologyClone for T
where
    T: Clone + Topology,
{
    fn into_box(&self) -> Box<dyn TopologyClone + '_> {
        Box::new(self.clone())
    }
}

impl std::fmt::Debug for &dyn Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.to_dot())
    }
}

impl std::fmt::Debug for Box<dyn Topology> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.to_dot())
    }
}

/// The network topology and hardware configuration of the cluster.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Cluster {
    graph: Graph<Node, Link>,
    node_map: HashMap<String, NodeIndex>,
    num_hosts: usize,
    num_racks: usize,
}

impl Cluster {
    #[inline]
    pub fn new() -> Self {
        Default::default()
    }

    pub fn refresh_node_map(&mut self) {
        let mut node_map = HashMap::default();
        for link_ix in self.all_links() {
            let dst = self.get_target(link_ix);
            let dst_name = self.graph[dst].name.clone();
            log::debug!("dst_name: {}", dst_name);
            *node_map.entry(dst_name).or_insert(dst) = dst;
        }
        self.node_map = node_map;
    }

    pub fn from_nodes(nodes: Vec<Node>) -> Self {
        let mut g = Graph::new();
        let mut node_map = HashMap::default();
        let num_hosts = nodes.iter().filter(|n| n.is_host()).count();
        let num_racks = nodes.iter().filter(|n| n.is_tor()).count();
        nodes.into_iter().for_each(|n| {
            let name = n.name.clone();
            let node_idx = g.add_node(n);
            let old = node_map.insert(name.clone(), node_idx);
            assert!(old.is_none(), "repeated key: {}", name);
        });
        Cluster {
            graph: g,
            node_map,
            num_hosts,
            num_racks,
        }
    }

    #[inline]
    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        if node.is_host() {
            self.num_hosts += 1;
        }
        if node.is_tor() {
            self.num_racks += 1;
        }
        let name = node.name.clone();
        let node_idx = self.graph.add_node(node);
        let old = self.node_map.insert(name.clone(), node_idx);
        assert!(old.is_none(), "repeated key: {}", name);
        node_idx
    }

    #[inline]
    pub fn add_link_by_name(&mut self, parent: &str, child: &str, bw: Bandwidth) {
        let &pnode = self
            .node_map
            .get(parent)
            .unwrap_or_else(|| panic!("cannot find node with name: {}", parent));
        let &cnode = self
            .node_map
            .get(child)
            .unwrap_or_else(|| panic!("cannot find node with name: {}", child));

        let l1 = self.graph.add_edge(pnode, cnode, Link::new(bw));
        let l2 = self.graph.add_edge(cnode, pnode, Link::new(bw));

        self.graph[pnode].children.push(l1);
        // assert!(self.graph[cnode].parent.is_empty());
        self.graph[cnode].parent.push(l2);
    }
}

impl std::ops::Index<NodeIx> for Cluster {
    type Output = Node;
    fn index(&self, index: NodeIx) -> &Self::Output {
        &self.graph[index]
    }
}

impl std::ops::Index<LinkIx> for Cluster {
    type Output = Link;
    fn index(&self, index: LinkIx) -> &Self::Output {
        &self.graph[index]
    }
}

impl std::ops::IndexMut<LinkIx> for Cluster {
    fn index_mut(&mut self, index: LinkIx) -> &mut Self::Output {
        &mut self.graph[index]
    }
}

impl IndexMut<NodeIx> for Cluster {
    fn index_mut(&mut self, index: NodeIx) -> &mut Self::Output {
        &mut self.graph[index]
    }
}

impl TopologyMultiPath for Cluster {
    fn get_uplinks(&self, ix: NodeIx) -> std::slice::Iter<'_, LinkIx> {
        self.graph[ix].parent.iter()
    }

    fn resolve_route_multipath(
        &self,
        flow: &Flow,
        load_balancer: &mut Option<Box<dyn LoadBalancer>>,
    ) -> Route {
        let Some(load_balancer) = load_balancer else {
            return self.resolve_route(
                &flow.src,
                &flow.dst,
                &RouteHint::VirtAddr(flow.vsrc.as_deref(), flow.vdst.as_deref()),
            );
        };

        let g = &self.graph;
        let src_id = self.node_map[&flow.src];
        let dst_id = self.node_map[&flow.dst];

        log::debug!("searching route from {} to {}", flow.src, flow.dst);
        log::trace!("src_node: {:?}, dst_node: {:?}", g[src_id], g[dst_id]);
        assert_eq!(g[src_id].depth, g[dst_id].depth);
        let mut depth = g[src_id].depth;

        if src_id == dst_id {
            return self.resolve_route(
                &flow.src,
                &flow.dst,
                &RouteHint::VirtAddr(flow.vsrc.as_deref(), flow.vdst.as_deref()),
            );
        }

        // path1: x --> root-level switch
        let mut path1 = Vec::new();
        // path2: reverse(y to the root-level switch in path1)
        let mut path2 = Vec::new();

        let mut x = src_id;
        // all the nodes at a certain layer that y can reach
        let mut ys = vec![vec![dst_id]];
        let mut ys_from = vec![vec![]];

        // first pass: find the LCA
        while !ys.last().unwrap().contains(&x) && depth > 1 {
            let choice = match g[x].parent.len() {
                0 => {
                    panic!("something is wrong, check your cluster topology")
                }
                _ => {
                    // has multiple paths
                    // choose a path with the load balancer
                    let hash = load_balancer.compute_hash(flow);
                    (hash % g[x].parent.len() as u64) as usize
                }
            };
            let parx = g[x].parent[choice as usize];
            x = g.raw_edges()[parx.index()].target();
            path1.push(parx);
            depth -= 1;
            // y goes one level up (like a BFS order)
            let mut ys_next = vec![];
            let mut ys_from_next = vec![];
            for &y in ys.last().unwrap() {
                ys_next.extend(
                    g[y].parent
                        .iter()
                        .map(|pary| g.raw_edges()[pary.index()].target()),
                );
                // ys_from_next.extend(std::iter::repeat(y).take(g[y].parent.len()));
                ys_from_next.extend(g[y].parent.iter());
            }
            ys.push(ys_next);
            ys_from.push(ys_from_next);
        }

        assert!(
            ys.last().unwrap().contains(&x),
            "route from {} to {} not found",
            flow.src,
            flow.dst
        );

        // second pass: generate the subpath from the common ancestor to y
        while x != dst_id && !ys.is_empty() {
            let ys_last = ys.pop().unwrap();
            let ys_from_last = ys_from.pop().unwrap();
            assert!(
                ys_last.len() == ys_from_last.len()
                    || (ys_from_last.len() == 0 && ys_last.len() == 1 && ys_last[0] == dst_id)
            );
            if let Some(pos) = ys_last.iter().position(|&y| y == x) {
                let parx = ys_from_last[pos];
                x = self.get_source(parx);
                path2.push(self.get_reverse_link(parx));
            } else {
                panic!("Something is wrong, check your topology")
            }
        }

        path1.append(&mut path2);
        let route = Route {
            from: src_id,
            to: dst_id,
            path: path1,
        };

        log::trace!(
            "find a route from {} to {}, route; {:#?}",
            flow.src,
            flow.dst,
            route
        );

        route
    }
}

impl Topology for Cluster {
    fn get_node_index(&self, name: &str) -> NodeIx {
        let &id = self
            .node_map
            .get(name)
            .unwrap_or_else(|| panic!("cannot find node with name: {}", name));
        id
    }

    fn get_target(&self, ix: LinkIx) -> NodeIx {
        self.graph.raw_edges()[ix.index()].target()
    }

    fn get_source(&self, ix: LinkIx) -> NodeIx {
        self.graph.raw_edges()[ix.index()].source()
    }

    fn get_uplink(&self, ix: NodeIx) -> LinkIx {
        *self.graph[ix]
            .parent
            .get(0)
            .unwrap_or_else(|| panic!("invalid index: {:?}", ix))
    }

    fn get_downlinks(&self, ix: NodeIx) -> std::slice::Iter<'_, LinkIx> {
        self.graph[ix].children.iter()
    }

    fn get_reverse_link(&self, ix: LinkIx) -> LinkIx {
        LinkIx::new(ix.index() ^ 1)
    }

    fn all_links(&self) -> EdgeIndices {
        self.graph.edge_indices()
    }

    fn find_link(&self, ix: NodeIx, iy: NodeIx) -> Option<LinkIx> {
        self.graph.find_edge(ix, iy)
    }

    /// Tenants' are segregated, so there must be no flow between two different tenants.
    /// There two some special cases that should be noted.
    /// When psrc == pdst && vsrc == vdst, the flow is intra-Vm flow, return empty path.
    /// When psrc == pdst && vsrc != vdst, the flow is inter-VM but the VM colocates, let the flow goes througth the access port of the switch.
    /// Otherwise, resolve the route as normal.
    fn resolve_route(&self, src: &str, dst: &str, hint: &RouteHint) -> Route {
        let g = &self.graph;

        log::debug!("searching route from {} to {}", src, dst);
        let src_id = self.node_map[src];
        let dst_id = self.node_map[dst];

        log::trace!("src_node: {:?}, dst_node: {:?}", g[src_id], g[dst_id]);
        assert_eq!(g[src_id].depth, g[dst_id].depth);
        let mut depth = g[src_id].depth;

        if src_id == dst_id {
            match hint {
                RouteHint::VirtAddr(Some(vsrc), Some(vdst)) if vsrc != vdst => {
                    // psrc == pdst && vsrc != vdst
                    let mut path = Vec::with_capacity(2);
                    let x = src_id;
                    let parx = g[x].parent[0];
                    path.push(parx);
                    path.push(self.get_reverse_link(parx));
                    log::trace!("find a route from {}[{}] to {}[{}]", src, vsrc, dst, vdst);
                    return Route {
                        from: src_id,
                        to: dst_id,
                        path,
                    };
                }
                RouteHint::VirtAddr(_, _) => {}
            }
        }

        let mut path1 = Vec::new();
        let mut path2 = Vec::new();

        let mut x = src_id;
        let mut y = dst_id;
        while x != y && depth > 1 {
            let parx = g[x].parent[0];
            let pary = g[y].parent[0];
            x = g.raw_edges()[parx.index()].target();
            y = g.raw_edges()[pary.index()].target();
            depth -= 1;
            path1.push(parx);
            path2.push(self.get_reverse_link(pary));
        }

        assert!(x == y, "route from {} to {} not found", src, dst);
        path2.reverse();
        path1.append(&mut path2);
        let route = Route {
            from: src_id,
            to: dst_id,
            path: path1,
        };

        log::trace!("find a route from {} to {}, route; {:#?}", src, dst, route);

        route
    }

    fn num_hosts(&self) -> usize {
        self.num_hosts
    }

    fn num_switches(&self) -> usize {
        self.graph.node_count() - self.num_hosts
    }

    fn num_racks(&self) -> usize {
        self.num_racks
    }

    fn translate(&self, vname: &str) -> String {
        vname.to_owned()
    }

    fn to_dot(&self) -> Dot<'_, &Graph<Node, Link>> {
        Dot::with_config(&self.graph, &[])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    id: usize,
    pub bandwidth: Bandwidth,
    // latency: Duration,
    // drop_rate: f64,
}

impl Link {
    #[inline]
    pub fn new(bw: Bandwidth) -> Link {
        Link {
            id: LINK_ID.fetch_add(1, SeqCst),
            bandwidth: bw,
        }
    }
}

impl std::fmt::Display for Link {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.bandwidth)
    }
}

impl std::cmp::PartialEq for Link {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Link {}

impl std::hash::Hash for Link {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NodeType {
    Host,
    Switch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub depth: usize, // 1: core, 2: agg, 3: edge, 4: host
    pub node_type: NodeType,
    // this gives us faster route search
    parent: Vec<EdgeIndex>,
    children: Vec<EdgeIndex>,
}

impl Node {
    #[inline]
    pub fn new(name: &str, depth: usize, node_type: NodeType) -> Node {
        Node {
            name: name.to_owned(),
            depth,
            node_type,
            parent: Vec::new(),
            children: Vec::new(),
        }
    }

    #[inline]
    pub fn is_host(&self) -> bool {
        matches!(self.node_type, NodeType::Host)
    }

    #[inline]
    pub fn is_tor(&self) -> bool {
        !self.is_host() && self.name.starts_with("tor_")
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A route is a network path that a flow goes through
#[derive(Debug, Clone)]
pub struct Route {
    pub from: NodeIndex,
    pub to: NodeIndex,
    pub path: Vec<LinkIx>,
}

#[derive(Debug, Clone)]
pub enum RouteHint<'a> {
    // vsrc, vdst
    VirtAddr(Option<&'a str>, Option<&'a str>),
}

impl<'a> Default for RouteHint<'a> {
    fn default() -> Self {
        RouteHint::VirtAddr(None, None)
    }
}

// helper functions
pub mod helpers {
    use super::*;

    #[inline]
    pub fn get_up_bw(vc: &dyn Topology, host_id: usize) -> Bandwidth {
        let host_name = format!("host_{}", host_id);
        let host_ix = vc.get_node_index(&host_name);
        vc[vc.get_uplink(host_ix)].bandwidth
    }

    #[inline]
    pub fn get_down_bw(vc: &dyn Topology, host_id: usize) -> Bandwidth {
        let host_name = format!("host_{}", host_id);
        let host_ix = vc.get_node_index(&host_name);
        vc[vc.get_reverse_link(vc.get_uplink(host_ix))].bandwidth
    }

    #[inline]
    pub fn get_rack_up_bw(vc: &dyn Topology, rack_id: usize) -> Bandwidth {
        let host_name = format!("tor_{}", rack_id);
        let tor_ix = vc.get_node_index(&host_name);
        vc[vc.get_uplink(tor_ix)].bandwidth
    }

    #[inline]
    pub fn get_rack_down_bw(vc: &dyn Topology, rack_id: usize) -> Bandwidth {
        let host_name = format!("tor_{}", rack_id);
        let tor_ix = vc.get_node_index(&host_name);
        vc[vc.get_reverse_link(vc.get_uplink(tor_ix))].bandwidth
    }

    #[inline]
    pub fn get_rack_ix(vc: &dyn Topology, host_id: usize) -> NodeIx {
        let host_name = format!("host_{}", host_id);
        let host_ix = vc.get_node_index(&host_name);
        let rack_ix = vc.get_target(vc.get_uplink(host_ix));
        rack_ix
    }

    #[inline]
    pub fn get_host_id(vc: &dyn Topology, host_ix: NodeIx) -> usize {
        vc[host_ix]
            .name
            .strip_prefix("host_")
            .unwrap()
            .parse()
            .unwrap()
    }

    #[inline]
    pub fn get_rack_id(vc: &dyn Topology, rack_ix: NodeIx) -> usize {
        vc[rack_ix]
            .name
            .strip_prefix("tor_")
            .unwrap()
            .parse()
            .unwrap()
    }
}
