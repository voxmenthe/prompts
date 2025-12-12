# ExecuTorch

[`ExecuTorch`](https://github.com/pytorch/executorch) is an end-to-end solution for enabling on-device inference capabilities across mobile and edge devices including wearables, embedded devices and microcontrollers. It is part of the PyTorch ecosystem and supports the deployment of PyTorch models with a focus on portability, productivity, and performance.

ExecuTorch introduces well defined entry points to perform model, device, and/or use-case specific optimizations such as backend delegation, user-defined compiler transformations, memory planning, and more. The first step in preparing a PyTorch model for execution on an edge device using ExecuTorch is to export the model. This is achieved through the use of a PyTorch API called [`torch.export`](https://pytorch.org/docs/stable/export.html).

## ExecuTorch Integration

An integration point is being developed to ensure that 🤗 Transformers can be exported using `torch.export`. The goal of this integration is not only to enable export but also to ensure that the exported artifact can be further lowered and optimized to run efficiently in `ExecuTorch`, particularly for mobile and edge use cases.

### class transformers.TorchExportableModuleWithStaticCache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/executorch.py#L462)

( model: PreTrainedModel batch\_size: typing.Optional[int] = None max\_cache\_len: typing.Optional[int] = None device: typing.Optional[torch.device] = None  )

A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
specifically for decoder-only LM to `StaticCache`. This module ensures that the
exported model is compatible with further lowering and execution in `ExecuTorch`.

Note:
This class is specifically designed to support export process using `torch.export`
in a way that ensures the model can be further lowered and run efficiently in `ExecuTorch`.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/executorch.py#L547)

( input\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → torch.Tensor

Parameters

* **input\_ids** (`torch.Tensor`) — Tensor representing current input token id to the module.
* **inputs\_embeds** (`torch.Tensor`) — Tensor representing current input embeddings to the module.
* **cache\_position** (`torch.Tensor`) — Tensor representing current input position in the cache.

Returns

torch.Tensor

Logits output from the model.

Forward pass of the module, which is compatible with the ExecuTorch runtime.

This forward adapter serves two primary purposes:

1. **Making the Model `torch.export`-Compatible**:
   The adapter hides unsupported objects, such as the `Cache`, from the graph inputs and outputs,
   enabling the model to be exportable using `torch.export` without encountering issues.
2. **Ensuring Compatibility with `ExecuTorch` runtime**:
   The adapter matches the model’s forward signature with that in `executorch/extension/llm/runner`,
   ensuring that the exported model can be executed in `ExecuTorch` out-of-the-box.

#### transformers.convert\_and\_export\_with\_cache

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/executorch.py#L746)

( model: PreTrainedModel example\_input\_ids: typing.Optional[torch.Tensor] = None example\_cache\_position: typing.Optional[torch.Tensor] = None dynamic\_shapes: typing.Optional[dict] = None strict: typing.Optional[bool] = None  ) → Exported program (`torch.export.ExportedProgram`)

Parameters

* **model** (`PreTrainedModel`) — The pretrained model to be exported.
* **example\_input\_ids** (`Optional[torch.Tensor]`) — Example input token id used by `torch.export`.
* **example\_cache\_position** (`Optional[torch.Tensor]`) — Example current cache position used by `torch.export`.
* **dynamic\_shapes(`Optional[dict]`)** — Dynamic shapes used by `torch.export`.
* **strict(`Optional[bool]`)** — Flag to instruct `torch.export` to use `torchdynamo`.

Returns

Exported program (`torch.export.ExportedProgram`)

The exported program generated via `torch.export`.

Convert a `PreTrainedModel` into an exportable module and export it using `torch.export`,
ensuring the exported model is compatible with `ExecuTorch`.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/executorch.md)
