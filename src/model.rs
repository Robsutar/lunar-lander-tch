use tch::nn::{self, Adam, Linear, Module, Optimizer, OptimizerConfig, VarStore};
use tch::{Device, Kind, Tensor};

use crate::{State, StepActionEvent};

const DEVICE: Device = Device::Cpu;
type DType = f32;
