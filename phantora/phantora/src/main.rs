use phantora::args::get_args;
use phantora::simulator;
use phantora::torch_call::{TorchCallJson, TorchCallMsg};

use cuda_call::{capi, CudaCallMsg, HostId, ResponseId};
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::net::UnixDatagram;

fn main() {
    let env = env_logger::Env::new()
        .filter("PHANTORA_LOG")
        .write_style("PHANTORA_LOG_STYLE");
    env_logger::init_from_env(env);

    let _args = get_args();
    main_loop();
}

#[derive(Serialize, Deserialize)]
enum Message {
    Cuda(CudaCallMsg),
    Torch(TorchCallMsg, i32 /* device index */),
    Exit(ResponseId, i64),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExitJson {
    pub pid: u32,
    pub tid: i32,
    pub hostname: String,
    pub cur: i64,
}

fn main_loop() {
    // Create domain socket for Phantora simulator
    let socket_path = capi::simulator_socket_path();
    let _ = fs::remove_file(&socket_path);
    let recv_socket = UnixDatagram::bind(socket_path).unwrap();

    let mut buf = [0u8; 1024 * 1024];
    let netconfig: netsim::config::Config = netsim::config::read_config(&get_args().net_config);
    println!("netconfig: {:#?}", netconfig);
    let cluster = netsim::config::build_cloud(&netconfig);
    let netsim = netsim::simulator::SimulatorBuilder::new()
        .with_setting(netconfig.simulator)
        .cluster(cluster)
        .host_mapping(netconfig.host_mapping)
        .build()
        .expect("Fail to create network simulator");
    let mut simulator = simulator::Simulator::new(netsim);

    loop {
        let sz = recv_socket.recv(&mut buf).unwrap();
        assert!(sz < buf.len());
        let last = sz - 1;
        let tag = buf[last];
        let buf = &buf[..last];
        let message;
        if tag == 1 {
            // CUDA call message
            let msg = bincode::deserialize::<CudaCallMsg>(&buf).unwrap();
            message = Message::Cuda(msg);
        } else if tag == 2 {
            // torch call message
            let info = std::str::from_utf8(&buf).unwrap();
            let callmsg = serde_json::from_str::<TorchCallJson>(info)
                .unwrap()
                .into_msg();
            if let Some(arg_device) = callmsg.gpu_index() {
                message = Message::Torch(callmsg, arg_device as _);
            } else {
                continue;
            }
        } else if tag == 3 {
            // Exit message
            let info = std::str::from_utf8(&buf).unwrap();
            let exitmsg = serde_json::from_str::<ExitJson>(info).unwrap();
            message = Message::Exit(
                ResponseId {
                    host: HostId {
                        hostname: exitmsg.hostname,
                        pid: exitmsg.pid,
                    },
                    tid: exitmsg.tid,
                },
                exitmsg.cur,
            );
        } else {
            panic!("Unknown message tag {}", tag);
        }

        match message {
            Message::Cuda(msg) => {
                log::debug!("{:?}", msg);
                simulator.handle_cuda_call(msg);
            }
            Message::Torch(callmsg, arg_device) => {
                log::debug!("{:?}", callmsg);
                if let Some(call) = callmsg.into_call(arg_device) {
                    simulator.handle_torch_call(call);
                }
            }
            Message::Exit(host, cur) => {
                log::debug!("{:?} exited at {}", host, cur);
                simulator.handle_exit(host, cur);
            }
        }
    }
}
