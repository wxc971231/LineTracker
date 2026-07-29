"""SimpleCNN 推理入口共用的模型、数据和统计基础设施。"""

from .model_loader import (
    InferenceBundle,
    infer_model_type,
    install_torch_npu_checkpoint_shim,
    load_inference_bundle,
    load_resolved_config,
    resolve_inference_device,
    resolve_run_dir,
)
from .source_loader import (
    load_split_manifest,
    load_split_sources,
    resolve_source_path,
    select_sources,
    split_manifest_path,
)

__all__ = [
    "InferenceBundle",
    "infer_model_type",
    "install_torch_npu_checkpoint_shim",
    "load_inference_bundle",
    "load_resolved_config",
    "resolve_inference_device",
    "resolve_run_dir",
    "load_split_manifest",
    "load_split_sources",
    "resolve_source_path",
    "select_sources",
    "split_manifest_path",
]
