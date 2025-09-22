use crate::cuda_bindings::*;
use crate::torch_call::{TensorInfo, TorchCall, TorchCallInfo};
use crate::{assert_cuda, estimate};
use lru::LruCache;
use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::NonZeroUsize;
use std::ptr;
use std::time::Duration;
use tch::{self, Device, Kind, Tensor};

enum KindRange {
    Integer,
    Bool,
    Float,
}

fn kind_range(kind: Kind) -> KindRange {
    match kind {
        Kind::Bool => KindRange::Bool,
        Kind::Uint8
        | Kind::Int8
        | Kind::Int16
        | Kind::UInt16
        | Kind::Int
        | Kind::Int64
        | Kind::UInt32
        | Kind::UInt64 => KindRange::Integer,
        _ => KindRange::Float,
    }
}

macro_rules! estimate_torch {
    ($n:expr, $e:expr) => {{
        tch::autocast(true, || estimate!($n, $e))
    }};
}

pub struct TorchEstimator {
    tensor_cache: LruCache<(i64, Kind), Tensor>,
    compute_cache: HashMap<TorchCallInfo, Duration>,
    sequence_cache: BTreeMap<u64, Vec<Duration>>,
}

impl TorchEstimator {
    pub fn new() -> Self {
        let mat = Tensor::randn(&[1024, 1024], (Kind::Float, Device::Cuda(0)));
        let _ = mat.mm(&mat); // initialization

        Self {
            tensor_cache: LruCache::new(NonZeroUsize::new(32).unwrap()),
            compute_cache: HashMap::new(),
            sequence_cache: BTreeMap::new(),
        }
    }

    fn allocate(&mut self, info: &TensorInfo) -> Tensor {
        let shape = info.shape.as_slice();
        let kind = info.dtype;
        let total_size = shape.iter().product();
        match self.tensor_cache.get(&(total_size, kind)) {
            Some(t) => t.view_(shape),
            None => {
                let t = match kind_range(kind) {
                    KindRange::Bool => Tensor::randint(2, shape, (kind, Device::Cuda(0))),
                    KindRange::Integer => Tensor::randint(128, shape, (kind, Device::Cuda(0))),
                    KindRange::Float => Tensor::randn(shape, (kind, Device::Cuda(0))),
                };
                self.tensor_cache.put((total_size, kind), t.view_(shape));
                t
            }
        }
    }

    fn allocate_list(&mut self, info: &[TensorInfo]) -> Vec<Tensor> {
        info.iter().map(|info| self.allocate(info)).collect()
    }

    fn cache(&mut self, t: Tensor) {
        if let Device::Cuda(0) = t.device() {
            let total_size = t.size().iter().product();
            let kind = t.kind();
            self.tensor_cache.put((total_size, kind), t);
        }
    }

    fn run(&mut self, niter: i32, call: &TorchCallInfo) -> Duration {
        match call {
            TorchCallInfo::MM(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.mm(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::MatMul(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.matmul(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Linear(info1, info2, bias_info) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let bias = bias_info.as_ref().map(|info| self.allocate(info));
                let (result, dur) =
                    estimate_torch!(niter, t1.linear::<&Tensor>(&t2, bias.as_ref()));
                self.cache(result);
                dur
            }
            TorchCallInfo::BMM(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.bmm(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::AddMM(info1, info2, info3) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t1.addmm(&t2, &t3));
                self.cache(result);
                dur
            }
            TorchCallInfo::BAddBMM(info1, info2, info3) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t1.baddbmm(&t2, &t3, 1, 1));
                self.cache(result);
                dur
            }
            TorchCallInfo::Mul(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_mul(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::MulScalar(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.multiply_scalar(2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Mul_(info1, info2) => {
                let mut t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_mul_(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::MulScalar_(info) => {
                let mut t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.multiply_scalar_(2));
                self.cache(result);
                dur
            }
            TorchCallInfo::ForeachMul_(info_list1, info_list2) => {
                let t_list2 = self.allocate_list(info_list2);
                let t_list1 = self.allocate_list(info_list1);
                let (_, dur) = estimate_torch!(
                    niter,
                    Tensor::f_internal_foreach_mul_list_(&t_list1, &t_list2)
                );
                dur
            }
            TorchCallInfo::ForeachMulScalar_(info_list) => {
                let t_list = self.allocate_list(info_list);
                let (_, dur) = estimate_torch!(niter, Tensor::f_internal_foreach_mul_(&t_list, 2));
                dur
            }
            TorchCallInfo::Add(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_add(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Add_(info1, info2) => {
                let mut t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_add_(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Div(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_div(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::DivScalar(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.divide_scalar(2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Pow(info) => {
                let mut t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.pow_(2));
                self.cache(result);
                dur
            }
            TorchCallInfo::AddCMul_(info1, info2, info3) => {
                let mut t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t1.addcmul_(&t2, &t3));
                self.cache(result);
                dur
            }
            TorchCallInfo::AddCDiv_(info1, info2, info3) => {
                let mut t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t1.addcdiv_(&t2, &t3));
                self.cache(result);
                dur
            }
            TorchCallInfo::ForeachAddCMul_(info_list1, info_list2, info_list3) => {
                let t_list2 = self.allocate_list(info_list2);
                let t_list3 = self.allocate_list(info_list3);
                let t_list1 = self.allocate_list(info_list1);
                let (_, dur) = estimate_torch!(
                    niter,
                    Tensor::f_internal_foreach_addcmul_(&t_list1, &t_list2, &t_list3, 2)
                );
                dur
            }
            TorchCallInfo::ForeachAddCDiv_(info_list1, info_list2, info_list3) => {
                let t_list2 = self.allocate_list(info_list2);
                let t_list3 = self.allocate_list(info_list3);
                let t_list1 = self.allocate_list(info_list1);
                let (_, dur) = estimate_torch!(
                    niter,
                    Tensor::f_internal_foreach_addcdiv_(&t_list1, &t_list2, &t_list3, 2)
                );
                dur
            }
            TorchCallInfo::Where(info1, info2, info3) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t2.where_self(&t1, &t3));
                self.cache(result);
                dur
            }
            TorchCallInfo::WhereScalar(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t2.where_scalarother(&t1, 1));
                self.cache(result);
                dur
            }
            TorchCallInfo::Sqrt(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.sqrt());
                self.cache(result);
                dur
            }
            TorchCallInfo::Softmax(info, dim) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.softmax(*dim, info.dtype));
                self.cache(result);
                dur
            }
            TorchCallInfo::SoftmaxBackward(info1, info2, dim) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(
                    niter,
                    Tensor::internal_softmax_backward_data(&t1, &t2, *dim, info1.dtype)
                );
                self.cache(result);
                dur
            }
            TorchCallInfo::ZerosLike(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.zeros_like());
                self.cache(result);
                dur
            }
            TorchCallInfo::ConvDType(info, kind) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.totype(*kind));
                self.cache(result);
                dur
            }
            TorchCallInfo::SDPA {
                q,
                k,
                v,
                causal,
                gqa,
            } => {
                let t_q = self.allocate(q);
                let t_k = self.allocate(k);
                let t_v = self.allocate(v);
                let (result, dur) = estimate_torch!(
                    niter,
                    Tensor::f_scaled_dot_product_attention(
                        &t_q,
                        &t_k,
                        &t_v,
                        None::<Tensor>,
                        0.0,
                        *causal,
                        None,
                        *gqa
                    )
                    .unwrap()
                );
                self.cache(result);
                dur
            }
            TorchCallInfo::SDPABackward {
                grad,
                q,
                k,
                v,
                out,
                logsumexp,
                max_q,
                max_k,
                causal,
            } => {
                let t_grad = self.allocate(grad);
                let t_q = self.allocate(q);
                let t_k = self.allocate(k);
                let t_v = self.allocate(v);
                let t_out = self.allocate(out);
                let t_logsumexp = self.allocate(logsumexp);
                let cum_seq = Tensor::new(); // should be undefined
                let philox_seed = Tensor::zeros(&[2], (Kind::UInt64, Device::Cuda(0)));
                let philox_offset = Tensor::zeros(&[1], (Kind::UInt64, Device::Cuda(0)));
                let (_, dur) = estimate_torch!(
                    niter,
                    Tensor::f_internal_scaled_dot_product_flash_attention_backward(
                        &t_grad,
                        &t_q,
                        &t_k,
                        &t_v,
                        &t_out,
                        &t_logsumexp,
                        &cum_seq,
                        &cum_seq,
                        *max_q,
                        *max_k,
                        0.0,
                        *causal,
                        &philox_seed,
                        &philox_offset,
                        None,
                    )
                    .unwrap()
                );
                self.cache(t_grad);
                self.cache(t_out);
                dur
            }
            TorchCallInfo::Conv2d {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let input = self.allocate(input);
                let weight = self.allocate(weight);
                let bias = bias.as_ref().map(|info| self.allocate(info));
                let (result, dur) = estimate_torch!(
                    niter,
                    input.conv2d(&weight, bias.as_ref(), stride, padding, dilation, *groups)
                );
                self.cache(result);
                dur
            }
            TorchCallInfo::Conv2dBackward {
                grad_output,
                input,
                weight,
                kernel,
                stride,
                padding,
            } => {
                let grad_bias = Tensor::empty(&[weight.shape[0]], (weight.dtype, Device::Cuda(0)));
                let input = self.allocate(input);
                let weight = self.allocate(weight);
                // let grad_input = input.empty_like();
                // let grad_weight = weight.empty_like();
                let grad_output = self.allocate(grad_output);

                let (_, dur) = estimate_torch!(
                    niter,
                    input.internal_slow_conv2d_backward(
                        &input,
                        &weight,
                        &grad_bias,
                        &grad_output,
                        &weight,
                        kernel,
                        stride,
                        padding
                    )
                );
                // self.cache(grad_input);
                // self.cache(grad_weight);
                self.cache(grad_bias);
                dur
            }
        }
    }

    pub fn estimate(&mut self, call: &TorchCallInfo) -> Duration {
        if let Some(value) = self.compute_cache.get(call) {
            *value
        } else {
            let duration = self.run(2, call);
            self.compute_cache.insert(call.clone(), duration);
            duration
        }
    }

    pub fn estimate_sequence(&mut self, calls: &[TorchCall]) -> Vec<Duration> {
        // self.tensor_cache.clear();
        let seq_hash = {
            let mut hasher = DefaultHasher::new();
            for call in calls {
                TorchCallInfo::hash(&call.info, &mut hasher);
            }
            hasher.finish()
        };
        if let Some(value) = self.sequence_cache.get(&seq_hash) {
            return value.clone();
        } else {
            let mut durs = Vec::new();
            for call in calls {
                durs.push(self.run(1, &call.info));
            }
            self.sequence_cache.insert(seq_hash, durs.clone());
            durs
        }
    }
}
