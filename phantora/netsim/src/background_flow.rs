use crate::{
    app::{AppEvent, AppEventKind, Application},
    simulator::{Event, Events},
    Duration, Flow, Timestamp, Trace, TraceRecord,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundFlowPattern {
    Alltoall,
    PlinkProbe,
}

/// BackgroundFlow App creates a certain traffic pattern that will last for a certain time in a cluster.
/// It keeps generating flows with `msg_size` and stops when dur_ms has passed.
/// It also has a generic type T as the output type. The output is just a stub to fit in AppGroup<T>.
#[derive(Debug)]
pub struct BackgroundFlowApp<T> {
    nhosts: usize,
    /// running for dur_ms milliseconds
    dur_ms: Duration,
    pattern: BackgroundFlowPattern,
    /// default msg_size is 1MB
    msg_size: usize,
    output_stub: T,

    // state
    stopped: bool,
    remaining_flows: usize,
}

impl<T> BackgroundFlowApp<T> {
    pub fn new(
        nhosts: usize,
        dur_ms: Duration,
        pattern: BackgroundFlowPattern,
        msg_size: Option<usize>,
        output_stub: T,
    ) -> Self {
        BackgroundFlowApp {
            nhosts,
            dur_ms,
            pattern,
            msg_size: msg_size.unwrap_or(1_000_000),
            output_stub,
            stopped: false,
            remaining_flows: 0,
        }
    }

    fn on_event_alltoall(&mut self, event: AppEvent) -> Events {
        match event.event {
            AppEventKind::AppStart => {
                let mut trace = Trace::new();
                let n = self.nhosts;
                let vnames: Vec<_> = (0..n)
                    .map(|i| {
                        let vname = format!("host_{}", i);
                        vname
                    })
                    .collect();

                #[allow(clippy::needless_range_loop)]
                for i in 0..n {
                    let sname = &vnames[i];
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let dname = &vnames[j];
                        let flow = Flow::new(self.msg_size, sname, dname, None);
                        let rec = TraceRecord::new(0, flow, None);
                        trace.add_record(rec);
                    }
                }

                self.remaining_flows = trace.recs.len();
                Event::FlowArrive(trace.recs).into()
            }
            AppEventKind::FlowComplete(recs) => {
                self.remaining_flows -= recs.len();
                let cur_ts = event.ts;
                if cur_ts > self.dur_ms * 2 * 1_000_000 {
                    log::warn!("background flow was expect to finish in {} ms, however it has been running for {} ms", self.dur_ms, cur_ts / 1_000_000);
                }

                assert!(!self.stopped || self.stopped && cur_ts >= self.dur_ms * 1_000_000);

                if cur_ts < self.dur_ms * 1_000_000 {
                    // start new flows
                    let mut new_flows = Trace::new();
                    for r in &recs {
                        let flow = Flow::new(r.flow.bytes, &r.flow.src, &r.flow.dst, r.flow.token);
                        let rec = TraceRecord::new(cur_ts, flow, None);
                        new_flows.add_record(rec);
                    }

                    self.remaining_flows += recs.len();
                    Event::FlowArrive(new_flows.recs).into()
                } else {
                    self.stopped = true;
                    if self.remaining_flows == 0 {
                        (Event::AppFinish).into()
                    } else {
                        Events::new()
                    }
                }
            }
            AppEventKind::UserNotification(..) | AppEventKind::AdapterNotification(..) => {
                unreachable!();
            }
        }
    }

    fn plink_probe_round(&self, cur_ts: Timestamp) -> Trace {
        use crate::RNG;
        use rand::seq::SliceRandom;

        let mut trace = Trace::new();
        let mut n = self.nhosts;
        let mut ids = RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let mut ids: Vec<_> = (0..n).collect();
            ids.shuffle(&mut *rng);
            ids
        });

        if n % 2 == 1 {
            ids.push(*ids.last().unwrap());
            n += 1;
        }

        let vnames: Vec<_> = ids
            .into_iter()
            .map(|i| {
                let vname = format!("host_{}", i);
                vname
            })
            .collect();

        for (i, j) in (0..n).step_by(2).zip((1..n).step_by(2)) {
            let sname = &vnames[i];
            let dname = &vnames[j];
            let flow = Flow::new(self.msg_size, sname, dname, None);
            let rec = TraceRecord::new(cur_ts, flow, None);
            trace.add_record(rec);
        }

        trace
    }

    fn on_event_plink_probe(&mut self, event: AppEvent) -> Events {
        match event.event {
            AppEventKind::AppStart => {
                assert_eq!(0, event.ts);
                let trace = self.plink_probe_round(0);
                self.remaining_flows = trace.recs.len();
                Event::FlowArrive(trace.recs).into()
            }
            AppEventKind::FlowComplete(recs) => {
                self.remaining_flows -= recs.len();
                let cur_ts = event.ts;
                if cur_ts > self.dur_ms * 2 * 1_000_000 {
                    log::warn!("background flow was expect to finish in {} ms, however it has been running for {} ms", self.dur_ms, cur_ts / 1_000_000);
                }

                assert!(!self.stopped || self.stopped && cur_ts >= self.dur_ms * 1_000_000);

                if cur_ts < self.dur_ms * 1_000_000 {
                    // start new flows
                    if self.remaining_flows == 0 {
                        let new_flows = self.plink_probe_round(cur_ts);
                        self.remaining_flows += new_flows.recs.len();
                        Event::FlowArrive(new_flows.recs).into()
                    } else {
                        Events::new()
                    }
                } else {
                    self.stopped = true;
                    if self.remaining_flows == 0 {
                        (Event::AppFinish).into()
                    } else {
                        Events::new()
                    }
                }
            }
            AppEventKind::UserNotification(..) | AppEventKind::AdapterNotification(..) => {
                unreachable!();
            }
        }
    }
}

impl<T: Clone + std::fmt::Debug> Application for BackgroundFlowApp<T> {
    type Output = T;

    fn on_event(&mut self, event: AppEvent) -> Events {
        match self.pattern {
            BackgroundFlowPattern::Alltoall => self.on_event_alltoall(event),
            BackgroundFlowPattern::PlinkProbe => self.on_event_plink_probe(event),
        }
    }

    fn answer(&mut self) -> Self::Output {
        self.output_stub.clone()
    }
}
