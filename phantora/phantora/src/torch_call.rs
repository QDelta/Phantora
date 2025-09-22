use cuda_call::{CudaStream, HostId, ResponseId};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct TorchCallJson {
    pid: u32,
    tid: i32,
    hostname: String,
    stream: Option<(i32, i32)>,
    cur: i64,
    name: String,
    args: Vec<TorchValueJson>,
}

impl TorchCallJson {
    pub fn into_msg(self) -> TorchCallMsg {
        let args = self
            .args
            .into_iter()
            .map(TorchValueJson::into_value)
            .collect();
        TorchCallMsg {
            pid: self.pid,
            tid: self.tid,
            hostname: self.hostname,
            stream: self.stream,
            curr_time: self.cur,
            name: self.name,
            args,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TorchValueJson {
    Tensor {
        shape: Vec<i64>,
        dtype: i32,
        device: String,
    },
    Tuple {
        elements: Vec<TorchValueJson>,
    },
    List {
        elements: Vec<TorchValueJson>,
    },
    Double {
        value: f64,
    },
    Int {
        value: i64,
    },
    Bool {
        value: bool,
    },
    String {
        value: String,
    },
    Device {
        value: String,
    },
}

impl TorchValueJson {
    fn into_value(self) -> TorchValue {
        match self {
            TorchValueJson::Tensor {
                shape,
                dtype,
                device,
            } => {
                let dtype = tch::Kind::from_c_int(dtype).unwrap();
                let device = parse_device(device);
                TorchValue::Tensor(shape, dtype, device)
            }
            TorchValueJson::Tuple { elements } => {
                let elements = elements
                    .into_iter()
                    .map(TorchValueJson::into_value)
                    .collect();
                TorchValue::Tuple(elements)
            }
            TorchValueJson::List { elements } => {
                let elements = elements
                    .into_iter()
                    .map(TorchValueJson::into_value)
                    .collect();
                TorchValue::List(elements)
            }
            TorchValueJson::Double { value } => TorchValue::Double(value),
            TorchValueJson::Int { value } => TorchValue::Int(value),
            TorchValueJson::Bool { value } => TorchValue::Bool(value),
            TorchValueJson::String { value } => TorchValue::String(value),
            TorchValueJson::Device { value } => TorchValue::Device(parse_device(value)),
        }
    }
}

fn parse_device<T: AsRef<str>>(s: T) -> tch::Device {
    let s = s.as_ref();
    if let Some(id) = s.strip_prefix("cuda:") {
        tch::Device::Cuda(id.parse().unwrap())
    } else if s == "cpu" {
        tch::Device::Cpu
    } else if s == "meta" {
        tch::Device::Cpu
    } else {
        log::warn!("Unknown device: {}", s);
        tch::Device::Cpu
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TorchCallMsg {
    pub pid: u32,
    pub tid: i32,
    pub hostname: String,
    pub stream: Option<(i32, i32)>,
    pub curr_time: i64,
    pub name: String,
    pub args: Vec<TorchValue>,
}

impl TorchCallMsg {
    pub fn is_aten(&self) -> bool {
        // TODO: support convolution_backward in bindings
        self.name.starts_with("aten::") && self.name != "aten::convolution_backward"
    }

    pub fn gpu_index(&self) -> Option<usize> {
        self.args.iter().find_map(TorchValue::gpu_index)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TorchValue {
    Tensor(Vec<i64>, tch::Kind, tch::Device),
    Tuple(Vec<TorchValue>),
    List(Vec<TorchValue>),
    Double(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Device(tch::Device),
}

impl TorchValue {
    pub fn gpu_index(&self) -> Option<usize> {
        match self {
            TorchValue::Tensor(_, _, device) => match device {
                tch::Device::Cuda(i) => Some(*i as usize),
                _ => None,
            },
            TorchValue::Tuple(elements) | TorchValue::List(elements) => {
                elements.iter().find_map(TorchValue::gpu_index)
            }
            _ => None,
        }
    }
}

fn maybe_tensor_list(value: &[TorchValue]) -> Option<Vec<TensorInfo>> {
    value
        .iter()
        .map(|v| match v {
            TorchValue::Tensor(shape, dtype, _) => Some(TensorInfo {
                shape: shape.clone(),
                dtype: *dtype,
            }),
            _ => None,
        })
        .collect()
}

impl TorchCallMsg {
    pub fn into_call(self, arg_device: i32) -> Option<TorchCall> {
        let stream = match self.stream {
            Some((device, id)) => CudaStream { device, id },
            None => CudaStream {
                device: arg_device,
                id: 0,
            },
        };

        use TorchValue::*;
        let call_info = match self.name.as_ref() {
            "aten::mm" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::MM(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::matmul" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => {
                    Some(TorchCallInfo::MatMul(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::linear" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => {
                    Some(TorchCallInfo::Linear(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        None,
                    ))
                }
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _)] => {
                    Some(TorchCallInfo::Linear(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        Some(TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        }),
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::bmm" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::BMM(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::addmm" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), ..] => {
                    Some(TorchCallInfo::AddMM(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::baddbmm" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), ..] => {
                    Some(TorchCallInfo::BAddBMM(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::mul" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::Mul(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                [Tensor(shape, kind, _), _] => Some(TorchCallInfo::MulScalar(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::mul_" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::Mul_(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                [Tensor(shape, kind, _), _] => Some(TorchCallInfo::MulScalar_(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_foreach_mul_" => match self.args.as_slice() {
                [List(tensor_list1), List(tensor_list2), ..] => maybe_tensor_list(tensor_list1)
                    .and_then(|tensor_list1| {
                        maybe_tensor_list(tensor_list2).and_then(|tensor_list2| {
                            Some(TorchCallInfo::ForeachMul_(tensor_list1, tensor_list2))
                        })
                    })
                    .or_else(|| {
                        log::warn!("{} args not match: {:?}", self.name, self.args);
                        None
                    }),
                [List(tensor_list1), ..] => match maybe_tensor_list(tensor_list1) {
                    Some(tensor_list1) => Some(TorchCallInfo::ForeachMulScalar_(tensor_list1)),
                    None => {
                        log::warn!("{} args not match: {:?}", self.name, self.args);
                        None
                    }
                },
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::add" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), ..] => {
                    Some(TorchCallInfo::Add(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::add_" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), ..] => {
                    Some(TorchCallInfo::Add_(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::div" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::Div(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                [Tensor(shape, kind, _), _] => Some(TorchCallInfo::DivScalar(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::pow" => match self.args.as_slice() {
                [Tensor(shape, kind, _), _] | [_, Tensor(shape, kind, _)] => {
                    Some(TorchCallInfo::Pow(TensorInfo {
                        shape: shape.clone(),
                        dtype: *kind,
                    }))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::addcmul_" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), ..] => {
                    Some(TorchCallInfo::AddCMul_(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::addcdiv_" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), ..] => {
                    Some(TorchCallInfo::AddCDiv_(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_foreach_addcmul_" => match self.args.as_slice() {
                [List(tensor_list1), List(tensor_list2), List(tensor_list3), ..] => {
                    maybe_tensor_list(tensor_list1)
                        .and_then(|tensor_list1| {
                            maybe_tensor_list(tensor_list2).and_then(|tensor_list2| {
                                maybe_tensor_list(tensor_list3).and_then(|tensor_list3| {
                                    Some(TorchCallInfo::ForeachAddCMul_(
                                        tensor_list1,
                                        tensor_list2,
                                        tensor_list3,
                                    ))
                                })
                            })
                        })
                        .or_else(|| {
                            log::warn!("{} args not match: {:?}", self.name, self.args);
                            None
                        })
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_foreach_addcdiv_" => match self.args.as_slice() {
                [List(tensor_list1), List(tensor_list2), List(tensor_list3), ..] => {
                    maybe_tensor_list(tensor_list1)
                        .and_then(|tensor_list1| {
                            maybe_tensor_list(tensor_list2).and_then(|tensor_list2| {
                                maybe_tensor_list(tensor_list3).and_then(|tensor_list3| {
                                    Some(TorchCallInfo::ForeachAddCDiv_(
                                        tensor_list1,
                                        tensor_list2,
                                        tensor_list3,
                                    ))
                                })
                            })
                        })
                        .or_else(|| {
                            log::warn!("{} args not match: {:?}", self.name, self.args);
                            None
                        })
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::where" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _)] => {
                    Some(TorchCallInfo::Where(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), _] => {
                    Some(TorchCallInfo::WhereScalar(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::sqrt" => match self.args.as_slice() {
                [Tensor(shape, kind, _)] => Some(TorchCallInfo::Sqrt(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::softmax" => match self.args.as_slice() {
                [Tensor(shape, kind, _), Int(dim), ..] => Some(TorchCallInfo::Softmax(
                    TensorInfo {
                        shape: shape.clone(),
                        dtype: *kind,
                    },
                    *dim,
                )),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_softmax_backward_data" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Int(dim), ..] => {
                    Some(TorchCallInfo::SoftmaxBackward(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        *dim,
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::zeros_like" => match self.args.as_slice() {
                [Tensor(shape, kind, _), ..] => Some(TorchCallInfo::ZerosLike(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::to" => match self.args.as_slice() {
                [Tensor(shape, kind, _), Int(dtype_idx), Bool(_), Bool(_)] => {
                    Some(TorchCallInfo::ConvDType(
                        TensorInfo {
                            shape: shape.clone(),
                            dtype: *kind,
                        },
                        tch::Kind::from_c_int(*dtype_idx as _).unwrap(),
                    ))
                }
                _ => {
                    // this is ok, only consider conv-dtype-only calls
                    log::debug!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::scaled_dot_product_attention" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), _, Bool(causal), Bool(gqa)] => {
                    Some(TorchCallInfo::SDPA {
                        q: TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        k: TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        v: TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                        causal: *causal,
                        gqa: *gqa,
                    })
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_scaled_dot_product_flash_attention_backward" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), Tensor(shape4, kind4, _), Tensor(shape5, kind5, _), Tensor(shape6, kind6, _), Int(max_q), Int(max_k), _, Bool(causal), _, _, _] => {
                    Some(TorchCallInfo::SDPABackward {
                        grad: TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        q: TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        k: TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                        v: TensorInfo {
                            shape: shape4.clone(),
                            dtype: *kind4,
                        },
                        out: TensorInfo {
                            shape: shape5.clone(),
                            dtype: *kind5,
                        },
                        logsumexp: TensorInfo {
                            shape: shape6.clone(),
                            dtype: *kind6,
                        },
                        max_q: *max_q,
                        max_k: *max_k,
                        causal: *causal,
                    })
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::conv2d" => match self.args.as_slice() {
                [Tensor(input_shape, input_kind, _), Tensor(weight_shape, weight_kind, _), List(stride), List(padding), List(dilation), Int(groups)] => {
                    match stride.as_slice() {
                        [Int(s1), Int(s2)] => match padding.as_slice() {
                            [Int(p1), Int(p2)] => match dilation.as_slice() {
                                [Int(d1), Int(d2)] => Some(TorchCallInfo::Conv2d {
                                    input: TensorInfo {
                                        shape: input_shape.clone(),
                                        dtype: *input_kind,
                                    },
                                    weight: TensorInfo {
                                        shape: weight_shape.clone(),
                                        dtype: *weight_kind,
                                    },
                                    bias: None,
                                    stride: [*s1, *s2],
                                    padding: [*p1, *p2],
                                    dilation: [*d1, *d2],
                                    groups: *groups,
                                }),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    }
                }
                [Tensor(input_shape, input_kind, _), Tensor(weight_shape, weight_kind, _), Tensor(bias_shape, bias_kind, _), List(stride), List(padding), List(dilation), Int(groups)] => {
                    match stride.as_slice() {
                        [Int(s1), Int(s2)] => match padding.as_slice() {
                            [Int(p1), Int(p2)] => match dilation.as_slice() {
                                [Int(d1), Int(d2)] => Some(TorchCallInfo::Conv2d {
                                    input: TensorInfo {
                                        shape: input_shape.clone(),
                                        dtype: *input_kind,
                                    },
                                    weight: TensorInfo {
                                        shape: weight_shape.clone(),
                                        dtype: *weight_kind,
                                    },
                                    bias: Some(TensorInfo {
                                        shape: bias_shape.clone(),
                                        dtype: *bias_kind,
                                    }),
                                    stride: [*s1, *s2],
                                    padding: [*p1, *p2],
                                    dilation: [*d1, *d2],
                                    groups: *groups,
                                }),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    }
                }
                _ => None,
            },
            "aten::_slow_conv2d_backward" => match self.args.as_slice() {
                [Tensor(grad_output_shape, grad_output_kind, _), Tensor(input_shape, input_kind, _), Tensor(weight_shape, weight_kind, _), List(kernel), List(stride), List(padding), List(_)] => {
                    match kernel.as_slice() {
                        [Int(k1), Int(k2)] => match stride.as_slice() {
                            [Int(s1), Int(s2)] => match padding.as_slice() {
                                [Int(p1), Int(p2)] => Some(TorchCallInfo::Conv2dBackward {
                                    grad_output: TensorInfo {
                                        shape: grad_output_shape.clone(),
                                        dtype: *grad_output_kind,
                                    },
                                    input: TensorInfo {
                                        shape: input_shape.clone(),
                                        dtype: *input_kind,
                                    },
                                    weight: TensorInfo {
                                        shape: weight_shape.clone(),
                                        dtype: *weight_kind,
                                    },
                                    kernel: [*k1, *k2],
                                    stride: [*s1, *s2],
                                    padding: [*p1, *p2],
                                }),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        };
        call_info.map(|info| TorchCall {
            time: self.curr_time,
            id: ResponseId {
                host: HostId {
                    hostname: self.hostname,
                    pid: self.pid,
                },
                tid: self.tid,
            },
            stream,
            info,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorInfo {
    pub shape: Vec<i64>,
    pub dtype: tch::Kind,
}

#[derive(strum::Display, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TorchCallInfo {
    MM(TensorInfo, TensorInfo),
    MatMul(TensorInfo, TensorInfo),
    Linear(TensorInfo, TensorInfo, Option<TensorInfo>),
    BMM(TensorInfo, TensorInfo),
    AddMM(TensorInfo, TensorInfo, TensorInfo),
    BAddBMM(TensorInfo, TensorInfo, TensorInfo),
    Mul(TensorInfo, TensorInfo),
    MulScalar(TensorInfo),
    Mul_(TensorInfo, TensorInfo),
    MulScalar_(TensorInfo),
    ForeachMul_(Vec<TensorInfo>, Vec<TensorInfo>),
    ForeachMulScalar_(Vec<TensorInfo>),
    Add(TensorInfo, TensorInfo),
    Add_(TensorInfo, TensorInfo),
    Div(TensorInfo, TensorInfo),
    DivScalar(TensorInfo),
    Pow(TensorInfo),
    AddCMul_(TensorInfo, TensorInfo, TensorInfo),
    AddCDiv_(TensorInfo, TensorInfo, TensorInfo),
    ForeachAddCMul_(Vec<TensorInfo>, Vec<TensorInfo>, Vec<TensorInfo>),
    ForeachAddCDiv_(Vec<TensorInfo>, Vec<TensorInfo>, Vec<TensorInfo>),
    Where(TensorInfo, TensorInfo, TensorInfo),
    WhereScalar(TensorInfo, TensorInfo),
    Sqrt(TensorInfo),
    Softmax(TensorInfo, i64),
    SoftmaxBackward(TensorInfo, TensorInfo, i64),
    ZerosLike(TensorInfo),
    ConvDType(TensorInfo, tch::Kind),
    SDPA {
        q: TensorInfo,
        k: TensorInfo,
        v: TensorInfo,
        causal: bool,
        gqa: bool,
    },
    SDPABackward {
        grad: TensorInfo,
        q: TensorInfo,
        k: TensorInfo,
        v: TensorInfo,
        out: TensorInfo,
        logsumexp: TensorInfo,
        max_q: i64,
        max_k: i64,
        causal: bool,
    },
    Conv2d {
        input: TensorInfo,
        weight: TensorInfo,
        bias: Option<TensorInfo>,
        stride: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        groups: i64,
    },
    Conv2dBackward {
        grad_output: TensorInfo,
        input: TensorInfo,
        weight: TensorInfo,
        kernel: [i64; 2],
        stride: [i64; 2],
        padding: [i64; 2],
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TorchCall {
    pub time: i64,
    pub id: ResponseId,
    pub stream: CudaStream,
    pub info: TorchCallInfo,
}
