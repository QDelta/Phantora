use cuda_call::{HostId, NcclDatatype};
use netsim::Flow;

pub(crate) type Trace = Vec<Flow>;

pub(crate) trait NcclOps {
    fn bcast(
        &mut self,
        root: i32,
        // [(host_id, device_id)]
        ranks: &[(HostId, i32)],
        count: usize,
        dtype: NcclDatatype,
    ) -> Trace;
    // fn reduce(
    //     &mut self,
    //     root: i32,
    //     ranks: &[(u32, i32)],
    //     count: usize,
    //     dtype: NcclDatatype,
    // ) -> Trace;
    fn allreduce(&mut self, ranks: &[(HostId, i32)], count: usize, dtype: NcclDatatype) -> Trace;
    fn allgather(&mut self, ranks: &[(HostId, i32)], count: usize, dtype: NcclDatatype) -> Trace;
    fn reduce_scatter(
        &mut self,
        ranks: &[(HostId, i32)],
        count: usize,
        dtype: NcclDatatype,
    ) -> Trace;
}

pub(crate) struct SimpleRing {
    #[allow(unused)]
    num_rings: u32,
}

impl Default for SimpleRing {
    fn default() -> Self {
        SimpleRing { num_rings: 1 }
    }
}

impl SimpleRing {
    #[allow(unused)]
    fn new(num_rings: u32) -> Self {
        SimpleRing { num_rings }
    }
}

impl NcclOps for SimpleRing {
    fn bcast(
        &mut self,
        root: i32,
        ranks: &[(HostId, i32)],
        count: usize,
        dtype: NcclDatatype,
    ) -> Trace {
        let mut trace = Trace::new();
        let n = ranks.len();
        let size = count * dtype.size();
        for i in 1..n {
            let src = &ranks[root as usize].0.hostname;
            let dst = &ranks[(root as usize + i) % n].0.hostname;
            trace.push(Flow::new(size, src, dst, None));
        }
        trace
    }

    fn allreduce(&mut self, ranks: &[(HostId, i32)], count: usize, dtype: NcclDatatype) -> Trace {
        let mut trace = Trace::new();
        let n = ranks.len();
        let size = count * dtype.size() * 2 * (n - 1) / n;
        for i in 0..n {
            let src = &ranks[i].0.hostname;
            let dst = &ranks[(i + 1) % n].0.hostname;
            trace.push(Flow::new(size, src, dst, None));
        }
        trace
    }

    fn allgather(&mut self, ranks: &[(HostId, i32)], count: usize, dtype: NcclDatatype) -> Trace {
        let mut trace = Trace::new();
        let n = ranks.len();
        let send_size = count * dtype.size() * n;
        let size = send_size / n * (n - 1);
        for i in 0..n {
            let src = &ranks[i].0.hostname;
            let dst = &ranks[(i + 1) % n].0.hostname;
            trace.push(Flow::new(size, src, dst, None));
        }
        trace
    }

    fn reduce_scatter(
        &mut self,
        ranks: &[(HostId, i32)],
        count: usize,
        dtype: NcclDatatype,
    ) -> Trace {
        let mut trace = Trace::new();
        let n = ranks.len();
        let send_size = count * dtype.size() * n;
        let size = send_size / n * (n - 1);
        for i in 0..n {
            let src = &ranks[i].0.hostname;
            let dst = &ranks[(i + 1) % n].0.hostname;
            trace.push(Flow::new(size, src, dst, None));
        }
        trace
    }
}
