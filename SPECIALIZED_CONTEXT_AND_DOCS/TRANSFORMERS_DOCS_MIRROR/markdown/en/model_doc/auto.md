# Auto Classes

In many cases, the architecture you want to use can be guessed from the name or the path of the pretrained model you
are supplying to the `from_pretrained()` method. AutoClasses are here to do this job for you so that you
automatically retrieve the relevant model given the name/path to the pretrained weights/config/vocabulary.

Instantiating one of [AutoConfig](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoConfig), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and
[AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer) will directly create a class of the relevant architecture. For instance


```
model = AutoModel.from_pretrained("google-bert/bert-base-cased")
```

will create a model that is an instance of [BertModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel).

There is one class of `AutoModel` for each task.

## Extending the Auto Classes

Each of the auto classes has a method to be extended with your custom classes. For instance, if you have defined a
custom class of model `NewModel`, make sure you have a `NewModelConfig` then you can add those to the auto
classes like this:


```
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```

You will then be able to use the auto classes like you would usually do!

If your `NewModelConfig` is a subclass of [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), make sure its
`model_type` attribute is set to the same key you use when registering the config (here `"new-model"`).

Likewise, if your `NewModel` is a subclass of [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel), make sure its
`config_class` attribute is set to the same class you use when registering the model (here
`NewModelConfig`).

## AutoConfig

### class transformers.AutoConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/configuration_auto.py#L1163)

( )

This is a generic configuration class that will be instantiated as one of the configuration classes of the library
when created with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoConfig.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/configuration_auto.py#L1186)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike[str]] \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model configuration hosted inside a model repo on
    huggingface.co.
  + A path to a *directory* containing a configuration file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.save_pretrained) method, or the [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) method,
    e.g., `./my_model_directory/`.
  + A path or url to a saved configuration JSON *file*, e.g.,
    `./my_model_directory/configuration.json`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download the model weights and configuration files and override the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **return\_unused\_kwargs** (`bool`, *optional*, defaults to `False`) —
  If `False`, then this function returns just the final configuration object.

  If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused\_kwargs* is a
  dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
  part of `kwargs` which has not been used to update `config` and is otherwise ignored.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **kwargs(additional** keyword arguments, *optional*) —
  The values in kwargs of any keys which are configuration attributes will be used to override the loaded
  values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
  by the `return_unused_kwargs` keyword parameter.

Instantiate one of the configuration classes of the library from a pretrained model configuration.

The configuration class to instantiate is selected based on the `model_type` property of the config object that
is loaded, or when it’s missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

* **aimv2** — [Aimv2Config](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Config) (AIMv2 model)
* **aimv2\_vision\_model** — [Aimv2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionConfig) (Aimv2VisionModel model)
* **albert** — [AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig) (ALBERT model)
* **align** — [AlignConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignConfig) (ALIGN model)
* **altclip** — [AltCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPConfig) (AltCLIP model)
* **apertus** — [ApertusConfig](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusConfig) (Apertus model)
* **arcee** — [ArceeConfig](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeConfig) (Arcee model)
* **aria** — [AriaConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig) (Aria model)
* **aria\_text** — [AriaTextConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextConfig) (AriaText model)
* **audio-spectrogram-transformer** — [ASTConfig](/docs/transformers/v4.56.2/en/model_doc/audio-spectrogram-transformer#transformers.ASTConfig) (Audio Spectrogram Transformer model)
* **autoformer** — [AutoformerConfig](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig) (Autoformer model)
* **aya\_vision** — [AyaVisionConfig](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig) (AyaVision model)
* **bamba** — [BambaConfig](/docs/transformers/v4.56.2/en/model_doc/bamba#transformers.BambaConfig) (Bamba model)
* **bark** — [BarkConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkConfig) (Bark model)
* **bart** — [BartConfig](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartConfig) (BART model)
* **beit** — [BeitConfig](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitConfig) (BEiT model)
* **bert** — [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) (BERT model)
* **bert-generation** — [BertGenerationConfig](/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationConfig) (Bert Generation model)
* **big\_bird** — [BigBirdConfig](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig) (BigBird model)
* **bigbird\_pegasus** — [BigBirdPegasusConfig](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig) (BigBird-Pegasus model)
* **biogpt** — [BioGptConfig](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig) (BioGpt model)
* **bit** — [BitConfig](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitConfig) (BiT model)
* **bitnet** — [BitNetConfig](/docs/transformers/v4.56.2/en/model_doc/bitnet#transformers.BitNetConfig) (BitNet model)
* **blenderbot** — [BlenderbotConfig](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotConfig) (Blenderbot model)
* **blenderbot-small** — [BlenderbotSmallConfig](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig) (BlenderbotSmall model)
* **blip** — [BlipConfig](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipConfig) (BLIP model)
* **blip-2** — [Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config) (BLIP-2 model)
* **blip\_2\_qformer** — [Blip2QFormerConfig](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerConfig) (BLIP-2 QFormer model)
* **bloom** — [BloomConfig](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig) (BLOOM model)
* **bridgetower** — [BridgeTowerConfig](/docs/transformers/v4.56.2/en/model_doc/bridgetower#transformers.BridgeTowerConfig) (BridgeTower model)
* **bros** — [BrosConfig](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosConfig) (BROS model)
* **camembert** — [CamembertConfig](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig) (CamemBERT model)
* **canine** — [CanineConfig](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineConfig) (CANINE model)
* **chameleon** — [ChameleonConfig](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonConfig) (Chameleon model)
* **chinese\_clip** — [ChineseCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig) (Chinese-CLIP model)
* **chinese\_clip\_vision\_model** — [ChineseCLIPVisionConfig](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPVisionConfig) (ChineseCLIPVisionModel model)
* **clap** — [ClapConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapConfig) (CLAP model)
* **clip** — [CLIPConfig](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPConfig) (CLIP model)
* **clip\_text\_model** — [CLIPTextConfig](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTextConfig) (CLIPTextModel model)
* **clip\_vision\_model** — [CLIPVisionConfig](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPVisionConfig) (CLIPVisionModel model)
* **clipseg** — [CLIPSegConfig](/docs/transformers/v4.56.2/en/model_doc/clipseg#transformers.CLIPSegConfig) (CLIPSeg model)
* **clvp** — [ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig) (CLVP model)
* **code\_llama** — [LlamaConfig](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig) (CodeLlama model)
* **codegen** — [CodeGenConfig](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenConfig) (CodeGen model)
* **cohere** — [CohereConfig](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereConfig) (Cohere model)
* **cohere2** — [Cohere2Config](/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2Config) (Cohere2 model)
* **cohere2\_vision** — [Cohere2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/cohere2_vision#transformers.Cohere2VisionConfig) (Cohere2Vision model)
* **colpali** — [ColPaliConfig](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliConfig) (ColPali model)
* **colqwen2** — [ColQwen2Config](/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2Config) (ColQwen2 model)
* **conditional\_detr** — [ConditionalDetrConfig](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig) (Conditional DETR model)
* **convbert** — [ConvBertConfig](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertConfig) (ConvBERT model)
* **convnext** — [ConvNextConfig](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextConfig) (ConvNeXT model)
* **convnextv2** — [ConvNextV2Config](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Config) (ConvNeXTV2 model)
* **cpmant** — [CpmAntConfig](/docs/transformers/v4.56.2/en/model_doc/cpmant#transformers.CpmAntConfig) (CPM-Ant model)
* **csm** — [CsmConfig](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig) (CSM model)
* **ctrl** — [CTRLConfig](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig) (CTRL model)
* **cvt** — [CvtConfig](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtConfig) (CvT model)
* **d\_fine** — [DFineConfig](/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineConfig) (D-FINE model)
* **dab-detr** — [DabDetrConfig](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrConfig) (DAB-DETR model)
* **dac** — [DacConfig](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacConfig) (DAC model)
* **data2vec-audio** — [Data2VecAudioConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig) (Data2VecAudio model)
* **data2vec-text** — [Data2VecTextConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig) (Data2VecText model)
* **data2vec-vision** — [Data2VecVisionConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig) (Data2VecVision model)
* **dbrx** — [DbrxConfig](/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxConfig) (DBRX model)
* **deberta** — [DebertaConfig](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaConfig) (DeBERTa model)
* **deberta-v2** — [DebertaV2Config](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config) (DeBERTa-v2 model)
* **decision\_transformer** — [DecisionTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/decision_transformer#transformers.DecisionTransformerConfig) (Decision Transformer model)
* **deepseek\_v2** — [DeepseekV2Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config) (DeepSeek-V2 model)
* **deepseek\_v3** — [DeepseekV3Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Config) (DeepSeek-V3 model)
* **deepseek\_vl** — [DeepseekVLConfig](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLConfig) (DeepseekVL model)
* **deepseek\_vl\_hybrid** — [DeepseekVLHybridConfig](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridConfig) (DeepseekVLHybrid model)
* **deformable\_detr** — [DeformableDetrConfig](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrConfig) (Deformable DETR model)
* **deit** — [DeiTConfig](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTConfig) (DeiT model)
* **depth\_anything** — [DepthAnythingConfig](/docs/transformers/v4.56.2/en/model_doc/depth_anything#transformers.DepthAnythingConfig) (Depth Anything model)
* **depth\_pro** — [DepthProConfig](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProConfig) (DepthPro model)
* **deta** — [DetaConfig](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaConfig) (DETA model)
* **detr** — [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig) (DETR model)
* **dia** — [DiaConfig](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaConfig) (Dia model)
* **diffllama** — [DiffLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaConfig) (DiffLlama model)
* **dinat** — [DinatConfig](/docs/transformers/v4.56.2/en/model_doc/dinat#transformers.DinatConfig) (DiNAT model)
* **dinov2** — [Dinov2Config](/docs/transformers/v4.56.2/en/model_doc/dinov2#transformers.Dinov2Config) (DINOv2 model)
* **dinov2\_with\_registers** — [Dinov2WithRegistersConfig](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersConfig) (DINOv2 with Registers model)
* **dinov3\_convnext** — [DINOv3ConvNextConfig](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ConvNextConfig) (DINOv3 ConvNext model)
* **dinov3\_vit** — [DINOv3ViTConfig](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ViTConfig) (DINOv3 ViT model)
* **distilbert** — [DistilBertConfig](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertConfig) (DistilBERT model)
* **doge** — [DogeConfig](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeConfig) (Doge model)
* **donut-swin** — [DonutSwinConfig](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinConfig) (DonutSwin model)
* **dots1** — [Dots1Config](/docs/transformers/v4.56.2/en/model_doc/dots1#transformers.Dots1Config) (dots1 model)
* **dpr** — [DPRConfig](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRConfig) (DPR model)
* **dpt** — [DPTConfig](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTConfig) (DPT model)
* **efficientformer** — [EfficientFormerConfig](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerConfig) (EfficientFormer model)
* **efficientloftr** — [EfficientLoFTRConfig](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRConfig) (EfficientLoFTR model)
* **efficientnet** — [EfficientNetConfig](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetConfig) (EfficientNet model)
* **electra** — [ElectraConfig](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig) (ELECTRA model)
* **emu3** — [Emu3Config](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Config) (Emu3 model)
* **encodec** — [EncodecConfig](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecConfig) (EnCodec model)
* **encoder-decoder** — [EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) (Encoder decoder model)
* **eomt** — [EomtConfig](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtConfig) (EoMT model)
* **ernie** — [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) (ERNIE model)
* **ernie4\_5** — [Ernie4\_5Config](/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Config) (Ernie4\_5 model)
* **ernie4\_5\_moe** — [Ernie4\_5\_MoeConfig](/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeConfig) (Ernie4\_5\_MoE model)
* **ernie\_m** — [ErnieMConfig](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMConfig) (ErnieM model)
* **esm** — [EsmConfig](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig) (ESM model)
* **evolla** — [EvollaConfig](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaConfig) (Evolla model)
* **exaone4** — [Exaone4Config](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config) (EXAONE-4.0 model)
* **falcon** — [FalconConfig](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig) (Falcon model)
* **falcon\_h1** — [FalconH1Config](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1Config) (FalconH1 model)
* **falcon\_mamba** — [FalconMambaConfig](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaConfig) (FalconMamba model)
* **fastspeech2\_conformer** — [FastSpeech2ConformerConfig](/docs/transformers/v4.56.2/en/model_doc/fastspeech2_conformer#transformers.FastSpeech2ConformerConfig) (FastSpeech2Conformer model)
* **fastspeech2\_conformer\_with\_hifigan** — [FastSpeech2ConformerWithHifiGanConfig](/docs/transformers/v4.56.2/en/model_doc/fastspeech2_conformer#transformers.FastSpeech2ConformerWithHifiGanConfig) (FastSpeech2ConformerWithHifiGan model)
* **flaubert** — [FlaubertConfig](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertConfig) (FlauBERT model)
* **flava** — [FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig) (FLAVA model)
* **florence2** — [Florence2Config](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Config) (Florence2 model)
* **fnet** — [FNetConfig](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig) (FNet model)
* **focalnet** — [FocalNetConfig](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetConfig) (FocalNet model)
* **fsmt** — [FSMTConfig](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTConfig) (FairSeq Machine-Translation model)
* **funnel** — [FunnelConfig](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelConfig) (Funnel Transformer model)
* **fuyu** — [FuyuConfig](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuConfig) (Fuyu model)
* **gemma** — [GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaConfig) (Gemma model)
* **gemma2** — [Gemma2Config](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Config) (Gemma2 model)
* **gemma3** — [Gemma3Config](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config) (Gemma3ForConditionalGeneration model)
* **gemma3\_text** — [Gemma3TextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextConfig) (Gemma3ForCausalLM model)
* **gemma3n** — [Gemma3nConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nConfig) (Gemma3nForConditionalGeneration model)
* **gemma3n\_audio** — [Gemma3nAudioConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nAudioConfig) (Gemma3nAudioEncoder model)
* **gemma3n\_text** — [Gemma3nTextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextConfig) (Gemma3nForCausalLM model)
* **gemma3n\_vision** — [Gemma3nVisionConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nVisionConfig) (TimmWrapperModel model)
* **git** — [GitConfig](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitConfig) (GIT model)
* **glm** — [GlmConfig](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmConfig) (GLM model)
* **glm4** — [Glm4Config](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Config) (GLM4 model)
* **glm4\_moe** — [Glm4MoeConfig](/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeConfig) (Glm4MoE model)
* **glm4v** — [Glm4vConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vConfig) (GLM4V model)
* **glm4v\_moe** — [Glm4vMoeConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeConfig) (GLM4VMOE model)
* **glm4v\_moe\_text** — [Glm4vMoeTextConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeTextConfig) (GLM4VMOE model)
* **glm4v\_text** — [Glm4vTextConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vTextConfig) (GLM4V model)
* **glpn** — [GLPNConfig](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNConfig) (GLPN model)
* **got\_ocr2** — [GotOcr2Config](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Config) (GOT-OCR2 model)
* **gpt-sw3** — [GPT2Config](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config) (GPT-Sw3 model)
* **gpt2** — [GPT2Config](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config) (OpenAI GPT-2 model)
* **gpt\_bigcode** — [GPTBigCodeConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeConfig) (GPTBigCode model)
* **gpt\_neo** — [GPTNeoConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig) (GPT Neo model)
* **gpt\_neox** — [GPTNeoXConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXConfig) (GPT NeoX model)
* **gpt\_neox\_japanese** — [GPTNeoXJapaneseConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseConfig) (GPT NeoX Japanese model)
* **gpt\_oss** — [GptOssConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssConfig) (GptOss model)
* **gptj** — [GPTJConfig](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJConfig) (GPT-J model)
* **gptsan-japanese** — [GPTSanJapaneseConfig](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseConfig) (GPTSAN-japanese model)
* **granite** — [GraniteConfig](/docs/transformers/v4.56.2/en/model_doc/granite#transformers.GraniteConfig) (Granite model)
* **granite\_speech** — [GraniteSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechConfig) (GraniteSpeech model)
* **granitemoe** — [GraniteMoeConfig](/docs/transformers/v4.56.2/en/model_doc/granitemoe#transformers.GraniteMoeConfig) (GraniteMoeMoe model)
* **granitemoehybrid** — [GraniteMoeHybridConfig](/docs/transformers/v4.56.2/en/model_doc/granitemoehybrid#transformers.GraniteMoeHybridConfig) (GraniteMoeHybrid model)
* **granitemoeshared** — [GraniteMoeSharedConfig](/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedConfig) (GraniteMoeSharedMoe model)
* **granitevision** — [LlavaNextConfig](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextConfig) (LLaVA-NeXT model)
* **graphormer** — [GraphormerConfig](/docs/transformers/v4.56.2/en/model_doc/graphormer#transformers.GraphormerConfig) (Graphormer model)
* **grounding-dino** — [GroundingDinoConfig](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoConfig) (Grounding DINO model)
* **groupvit** — [GroupViTConfig](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTConfig) (GroupViT model)
* **helium** — [HeliumConfig](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumConfig) (Helium model)
* **hgnet\_v2** — [HGNetV2Config](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Config) (HGNet-V2 model)
* **hiera** — [HieraConfig](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraConfig) (Hiera model)
* **hubert** — [HubertConfig](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertConfig) (Hubert model)
* **hunyuan\_v1\_dense** — [HunYuanDenseV1Config](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Config) (HunYuanDenseV1 model)
* **hunyuan\_v1\_moe** — [HunYuanMoEV1Config](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Config) (HunYuanMoeV1 model)
* **ibert** — [IBertConfig](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertConfig) (I-BERT model)
* **idefics** — [IdeficsConfig](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsConfig) (IDEFICS model)
* **idefics2** — [Idefics2Config](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Config) (Idefics2 model)
* **idefics3** — [Idefics3Config](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3Config) (Idefics3 model)
* **idefics3\_vision** — [Idefics3VisionConfig](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3VisionConfig) (Idefics3VisionTransformer model)
* **ijepa** — [IJepaConfig](/docs/transformers/v4.56.2/en/model_doc/ijepa#transformers.IJepaConfig) (I-JEPA model)
* **imagegpt** — [ImageGPTConfig](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig) (ImageGPT model)
* **informer** — [InformerConfig](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig) (Informer model)
* **instructblip** — [InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig) (InstructBLIP model)
* **instructblipvideo** — [InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig) (InstructBlipVideo model)
* **internvl** — [InternVLConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLConfig) (InternVL model)
* **internvl\_vision** — [InternVLVisionConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLVisionConfig) (InternVLVision model)
* **jamba** — [JambaConfig](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaConfig) (Jamba model)
* **janus** — [JanusConfig](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusConfig) (Janus model)
* **jetmoe** — [JetMoeConfig](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeConfig) (JetMoe model)
* **jukebox** — [JukeboxConfig](/docs/transformers/v4.56.2/en/model_doc/jukebox#transformers.JukeboxConfig) (Jukebox model)
* **kosmos-2** — [Kosmos2Config](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Config) (KOSMOS-2 model)
* **kosmos-2.5** — [Kosmos2\_5Config](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Config) (KOSMOS-2.5 model)
* **kyutai\_speech\_to\_text** — [KyutaiSpeechToTextConfig](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextConfig) (KyutaiSpeechToText model)
* **layoutlm** — [LayoutLMConfig](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig) (LayoutLM model)
* **layoutlmv2** — [LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config) (LayoutLMv3 model)
* **led** — [LEDConfig](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDConfig) (LED model)
* **levit** — [LevitConfig](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitConfig) (LeViT model)
* **lfm2** — [Lfm2Config](/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Config) (Lfm2 model)
* **lightglue** — [LightGlueConfig](/docs/transformers/v4.56.2/en/model_doc/lightglue#transformers.LightGlueConfig) (LightGlue model)
* **lilt** — [LiltConfig](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltConfig) (LiLT model)
* **llama** — [LlamaConfig](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig) (LLaMA model)
* **llama4** — [Llama4Config](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Config) (Llama4 model)
* **llama4\_text** — [Llama4TextConfig](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4TextConfig) (Llama4ForCausalLM model)
* **llava** — [LlavaConfig](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaConfig) (LLaVa model)
* **llava\_next** — [LlavaNextConfig](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextConfig) (LLaVA-NeXT model)
* **llava\_next\_video** — [LlavaNextVideoConfig](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoConfig) (LLaVa-NeXT-Video model)
* **llava\_onevision** — [LlavaOnevisionConfig](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionConfig) (LLaVA-Onevision model)
* **longformer** — [LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig) (Longformer model)
* **longt5** — [LongT5Config](/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Config) (LongT5 model)
* **luke** — [LukeConfig](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig) (LUKE model)
* **lxmert** — [LxmertConfig](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertConfig) (LXMERT model)
* **m2m\_100** — [M2M100Config](/docs/transformers/v4.56.2/en/model_doc/m2m_100#transformers.M2M100Config) (M2M100 model)
* **mamba** — [MambaConfig](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaConfig) (Mamba model)
* **mamba2** — [Mamba2Config](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2Config) (mamba2 model)
* **marian** — [MarianConfig](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianConfig) (Marian model)
* **markuplm** — [MarkupLMConfig](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMConfig) (MarkupLM model)
* **mask2former** — [Mask2FormerConfig](/docs/transformers/v4.56.2/en/model_doc/mask2former#transformers.Mask2FormerConfig) (Mask2Former model)
* **maskformer** — [MaskFormerConfig](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig) (MaskFormer model)
* **maskformer-swin** — `MaskFormerSwinConfig` (MaskFormerSwin model)
* **mbart** — [MBartConfig](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig) (mBART model)
* **mctct** — [MCTCTConfig](/docs/transformers/v4.56.2/en/model_doc/mctct#transformers.MCTCTConfig) (M-CTC-T model)
* **mega** — [MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig) (MEGA model)
* **megatron-bert** — [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) (Megatron-BERT model)
* **metaclip\_2** — [MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config) (MetaCLIP 2 model)
* **mgp-str** — [MgpstrConfig](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrConfig) (MGP-STR model)
* **mimi** — [MimiConfig](/docs/transformers/v4.56.2/en/model_doc/mimi#transformers.MimiConfig) (Mimi model)
* **minimax** — [MiniMaxConfig](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxConfig) (MiniMax model)
* **mistral** — [MistralConfig](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig) (Mistral model)
* **mistral3** — [Mistral3Config](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Config) (Mistral3 model)
* **mixtral** — [MixtralConfig](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralConfig) (Mixtral model)
* **mlcd** — [MLCDVisionConfig](/docs/transformers/v4.56.2/en/model_doc/mlcd#transformers.MLCDVisionConfig) (MLCD model)
* **mllama** — [MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig) (Mllama model)
* **mm-grounding-dino** — [MMGroundingDinoConfig](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoConfig) (MM Grounding DINO model)
* **mobilebert** — [MobileBertConfig](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig) (MobileBERT model)
* **mobilenet\_v1** — [MobileNetV1Config](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1Config) (MobileNetV1 model)
* **mobilenet\_v2** — [MobileNetV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2Config) (MobileNetV2 model)
* **mobilevit** — [MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig) (MobileViT model)
* **mobilevitv2** — [MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config) (MobileViTV2 model)
* **modernbert** — [ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig) (ModernBERT model)
* **modernbert-decoder** — [ModernBertDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderConfig) (ModernBertDecoder model)
* **moonshine** — [MoonshineConfig](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig) (Moonshine model)
* **moshi** — [MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig) (Moshi model)
* **mpnet** — [MPNetConfig](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig) (MPNet model)
* **mpt** — [MptConfig](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptConfig) (MPT model)
* **mra** — [MraConfig](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraConfig) (MRA model)
* **mt5** — [MT5Config](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config) (MT5 model)
* **musicgen** — [MusicgenConfig](/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenConfig) (MusicGen model)
* **musicgen\_melody** — [MusicgenMelodyConfig](/docs/transformers/v4.56.2/en/model_doc/musicgen_melody#transformers.MusicgenMelodyConfig) (MusicGen Melody model)
* **mvp** — [MvpConfig](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig) (MVP model)
* **nat** — [NatConfig](/docs/transformers/v4.56.2/en/model_doc/nat#transformers.NatConfig) (NAT model)
* **nemotron** — [NemotronConfig](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronConfig) (Nemotron model)
* **nezha** — [NezhaConfig](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig) (Nezha model)
* **nllb-moe** — [NllbMoeConfig](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeConfig) (NLLB-MOE model)
* **nougat** — [VisionEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) (Nougat model)
* **nystromformer** — [NystromformerConfig](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig) (Nyströmformer model)
* **olmo** — [OlmoConfig](/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoConfig) (OLMo model)
* **olmo2** — [Olmo2Config](/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Config) (OLMo2 model)
* **olmoe** — [OlmoeConfig](/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeConfig) (OLMoE model)
* **omdet-turbo** — [OmDetTurboConfig](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboConfig) (OmDet-Turbo model)
* **oneformer** — [OneFormerConfig](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerConfig) (OneFormer model)
* **open-llama** — [OpenLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaConfig) (OpenLlama model)
* **openai-gpt** — [OpenAIGPTConfig](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig) (OpenAI GPT model)
* **opt** — [OPTConfig](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig) (OPT model)
* **ovis2** — [Ovis2Config](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Config) (Ovis2 model)
* **owlv2** — [Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config) (OWLv2 model)
* **owlvit** — [OwlViTConfig](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTConfig) (OWL-ViT model)
* **paligemma** — [PaliGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaConfig) (PaliGemma model)
* **patchtsmixer** — [PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig) (PatchTSMixer model)
* **patchtst** — [PatchTSTConfig](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTConfig) (PatchTST model)
* **pegasus** — [PegasusConfig](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusConfig) (Pegasus model)
* **pegasus\_x** — [PegasusXConfig](/docs/transformers/v4.56.2/en/model_doc/pegasus_x#transformers.PegasusXConfig) (PEGASUS-X model)
* **perceiver** — [PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig) (Perceiver model)
* **perception\_encoder** — [TimmWrapperConfig](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperConfig) (PerceptionEncoder model)
* **perception\_lm** — [PerceptionLMConfig](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMConfig) (PerceptionLM model)
* **persimmon** — [PersimmonConfig](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonConfig) (Persimmon model)
* **phi** — [PhiConfig](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiConfig) (Phi model)
* **phi3** — [Phi3Config](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Config) (Phi3 model)
* **phi4\_multimodal** — [Phi4MultimodalConfig](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalConfig) (Phi4Multimodal model)
* **phimoe** — [PhimoeConfig](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeConfig) (Phimoe model)
* **pix2struct** — [Pix2StructConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructConfig) (Pix2Struct model)
* **pixtral** — [PixtralVisionConfig](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionConfig) (Pixtral model)
* **plbart** — [PLBartConfig](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig) (PLBart model)
* **poolformer** — [PoolFormerConfig](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerConfig) (PoolFormer model)
* **pop2piano** — [Pop2PianoConfig](/docs/transformers/v4.56.2/en/model_doc/pop2piano#transformers.Pop2PianoConfig) (Pop2Piano model)
* **prompt\_depth\_anything** — [PromptDepthAnythingConfig](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingConfig) (PromptDepthAnything model)
* **prophetnet** — [ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig) (ProphetNet model)
* **pvt** — [PvtConfig](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtConfig) (PVT model)
* **pvt\_v2** — [PvtV2Config](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Config) (PVTv2 model)
* **qdqbert** — [QDQBertConfig](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig) (QDQBert model)
* **qwen2** — [Qwen2Config](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Config) (Qwen2 model)
* **qwen2\_5\_omni** — [Qwen2\_5OmniConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniConfig) (Qwen2\_5Omni model)
* **qwen2\_5\_vl** — [Qwen2\_5\_VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLConfig) (Qwen2\_5\_VL model)
* **qwen2\_5\_vl\_text** — [Qwen2\_5\_VLTextConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLTextConfig) (Qwen2\_5\_VL model)
* **qwen2\_audio** — [Qwen2AudioConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioConfig) (Qwen2Audio model)
* **qwen2\_audio\_encoder** — [Qwen2AudioEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioEncoderConfig) (Qwen2AudioEncoder model)
* **qwen2\_moe** — [Qwen2MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeConfig) (Qwen2MoE model)
* **qwen2\_vl** — [Qwen2VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLConfig) (Qwen2VL model)
* **qwen2\_vl\_text** — [Qwen2VLTextConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLTextConfig) (Qwen2VL model)
* **qwen3** — [Qwen3Config](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Config) (Qwen3 model)
* **qwen3\_moe** — [Qwen3MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeConfig) (Qwen3MoE model)
* **rag** — [RagConfig](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig) (RAG model)
* **realm** — [RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig) (REALM model)
* **recurrent\_gemma** — [RecurrentGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaConfig) (RecurrentGemma model)
* **reformer** — [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig) (Reformer model)
* **regnet** — [RegNetConfig](/docs/transformers/v4.56.2/en/model_doc/regnet#transformers.RegNetConfig) (RegNet model)
* **rembert** — [RemBertConfig](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertConfig) (RemBERT model)
* **resnet** — [ResNetConfig](/docs/transformers/v4.56.2/en/model_doc/resnet#transformers.ResNetConfig) (ResNet model)
* **retribert** — [RetriBertConfig](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertConfig) (RetriBERT model)
* **roberta** — [RobertaConfig](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaPreLayerNormConfig](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertConfig](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig) (RoCBert model)
* **roformer** — [RoFormerConfig](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig) (RoFormer model)
* **rt\_detr** — [RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig) (RT-DETR model)
* **rt\_detr\_resnet** — [RTDetrResNetConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrResNetConfig) (RT-DETR-ResNet model)
* **rt\_detr\_v2** — [RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config) (RT-DETRv2 model)
* **rwkv** — [RwkvConfig](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvConfig) (RWKV model)
* **sam** — [SamConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamConfig) (SAM model)
* **sam2** — [Sam2Config](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Config) (SAM2 model)
* **sam2\_hiera\_det\_model** — [Sam2HieraDetConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2HieraDetConfig) (Sam2HieraDetModel model)
* **sam2\_video** — [Sam2VideoConfig](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoConfig) (Sam2VideoModel model)
* **sam2\_vision\_model** — [Sam2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2VisionConfig) (Sam2VisionModel model)
* **sam\_hq** — [SamHQConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQConfig) (SAM-HQ model)
* **sam\_hq\_vision\_model** — [SamHQVisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQVisionConfig) (SamHQVisionModel model)
* **sam\_vision\_model** — [SamVisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamVisionConfig) (SamVisionModel model)
* **seamless\_m4t** — [SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig) (SeamlessM4T model)
* **seamless\_m4t\_v2** — [SeamlessM4Tv2Config](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Config) (SeamlessM4Tv2 model)
* **seed\_oss** — [SeedOssConfig](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssConfig) (SeedOss model)
* **segformer** — [SegformerConfig](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerConfig) (SegFormer model)
* **seggpt** — [SegGptConfig](/docs/transformers/v4.56.2/en/model_doc/seggpt#transformers.SegGptConfig) (SegGPT model)
* **sew** — [SEWConfig](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWConfig) (SEW model)
* **sew-d** — [SEWDConfig](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDConfig) (SEW-D model)
* **shieldgemma2** — [ShieldGemma2Config](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2Config) (Shieldgemma2 model)
* **siglip** — [SiglipConfig](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipConfig) (SigLIP model)
* **siglip2** — [Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config) (SigLIP2 model)
* **siglip\_vision\_model** — [SiglipVisionConfig](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipVisionConfig) (SiglipVisionModel model)
* **smollm3** — [SmolLM3Config](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Config) (SmolLM3 model)
* **smolvlm** — [SmolVLMConfig](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMConfig) (SmolVLM model)
* **smolvlm\_vision** — [SmolVLMVisionConfig](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMVisionConfig) (SmolVLMVisionTransformer model)
* **speech-encoder-decoder** — [SpeechEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig) (Speech Encoder decoder model)
* **speech\_to\_text** — [Speech2TextConfig](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextConfig) (Speech2Text model)
* **speech\_to\_text\_2** — [Speech2Text2Config](/docs/transformers/v4.56.2/en/model_doc/speech_to_text_2#transformers.Speech2Text2Config) (Speech2Text2 model)
* **speecht5** — [SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config) (SpeechT5 model)
* **splinter** — [SplinterConfig](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterConfig) (Splinter model)
* **squeezebert** — [SqueezeBertConfig](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig) (SqueezeBERT model)
* **stablelm** — [StableLmConfig](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmConfig) (StableLm model)
* **starcoder2** — [Starcoder2Config](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2Config) (Starcoder2 model)
* **superglue** — [SuperGlueConfig](/docs/transformers/v4.56.2/en/model_doc/superglue#transformers.SuperGlueConfig) (SuperGlue model)
* **superpoint** — [SuperPointConfig](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointConfig) (SuperPoint model)
* **swiftformer** — [SwiftFormerConfig](/docs/transformers/v4.56.2/en/model_doc/swiftformer#transformers.SwiftFormerConfig) (SwiftFormer model)
* **swin** — [SwinConfig](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinConfig) (Swin Transformer model)
* **swin2sr** — [Swin2SRConfig](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRConfig) (Swin2SR model)
* **swinv2** — [Swinv2Config](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2Config) (Swin Transformer V2 model)
* **switch\_transformers** — [SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig) (SwitchTransformers model)
* **t5** — [T5Config](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config) (T5 model)
* **t5gemma** — [T5GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig) (T5Gemma model)
* **table-transformer** — [TableTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/table-transformer#transformers.TableTransformerConfig) (Table Transformer model)
* **tapas** — [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) (TAPAS model)
* **textnet** — [TextNetConfig](/docs/transformers/v4.56.2/en/model_doc/textnet#transformers.TextNetConfig) (TextNet model)
* **time\_series\_transformer** — [TimeSeriesTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerConfig) (Time Series Transformer model)
* **timesfm** — [TimesFmConfig](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmConfig) (TimesFm model)
* **timesformer** — [TimesformerConfig](/docs/transformers/v4.56.2/en/model_doc/timesformer#transformers.TimesformerConfig) (TimeSformer model)
* **timm\_backbone** — [TimmBackboneConfig](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackboneConfig) (TimmBackbone model)
* **timm\_wrapper** — [TimmWrapperConfig](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperConfig) (TimmWrapperModel model)
* **trajectory\_transformer** — [TrajectoryTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerConfig) (Trajectory Transformer model)
* **transfo-xl** — [TransfoXLConfig](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLConfig) (Transformer-XL model)
* **trocr** — [TrOCRConfig](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRConfig) (TrOCR model)
* **tvlt** — [TvltConfig](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltConfig) (TVLT model)
* **tvp** — [TvpConfig](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpConfig) (TVP model)
* **udop** — [UdopConfig](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopConfig) (UDOP model)
* **umt5** — [UMT5Config](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config) (UMT5 model)
* **unispeech** — [UniSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechConfig) (UniSpeech model)
* **unispeech-sat** — [UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig) (UniSpeechSat model)
* **univnet** — [UnivNetConfig](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetConfig) (UnivNet model)
* **upernet** — [UperNetConfig](/docs/transformers/v4.56.2/en/model_doc/upernet#transformers.UperNetConfig) (UPerNet model)
* **van** — [VanConfig](/docs/transformers/v4.56.2/en/model_doc/van#transformers.VanConfig) (VAN model)
* **video\_llava** — [VideoLlavaConfig](/docs/transformers/v4.56.2/en/model_doc/video_llava#transformers.VideoLlavaConfig) (VideoLlava model)
* **videomae** — [VideoMAEConfig](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEConfig) (VideoMAE model)
* **vilt** — [ViltConfig](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltConfig) (ViLT model)
* **vipllava** — [VipLlavaConfig](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaConfig) (VipLlava model)
* **vision-encoder-decoder** — [VisionEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) (Vision Encoder decoder model)
* **vision-text-dual-encoder** — [VisionTextDualEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig) (VisionTextDualEncoder model)
* **visual\_bert** — [VisualBertConfig](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig) (VisualBERT model)
* **vit** — [ViTConfig](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTConfig) (ViT model)
* **vit\_hybrid** — [ViTHybridConfig](/docs/transformers/v4.56.2/en/model_doc/vit_hybrid#transformers.ViTHybridConfig) (ViT Hybrid model)
* **vit\_mae** — [ViTMAEConfig](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEConfig) (ViTMAE model)
* **vit\_msn** — [ViTMSNConfig](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNConfig) (ViTMSN model)
* **vitdet** — [VitDetConfig](/docs/transformers/v4.56.2/en/model_doc/vitdet#transformers.VitDetConfig) (VitDet model)
* **vitmatte** — [VitMatteConfig](/docs/transformers/v4.56.2/en/model_doc/vitmatte#transformers.VitMatteConfig) (ViTMatte model)
* **vitpose** — [VitPoseConfig](/docs/transformers/v4.56.2/en/model_doc/vitpose#transformers.VitPoseConfig) (ViTPose model)
* **vitpose\_backbone** — `VitPoseBackboneConfig` (ViTPoseBackbone model)
* **vits** — [VitsConfig](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsConfig) (VITS model)
* **vivit** — [VivitConfig](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitConfig) (ViViT model)
* **vjepa2** — [VJEPA2Config](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Config) (VJEPA2Model model)
* **voxtral** — [VoxtralConfig](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralConfig) (Voxtral model)
* **voxtral\_encoder** — [VoxtralEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralEncoderConfig) (Voxtral Encoder model)
* **wav2vec2** — [Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) (Wav2Vec2 model)
* **wav2vec2-bert** — [Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig) (Wav2Vec2-BERT model)
* **wav2vec2-conformer** — [Wav2Vec2ConformerConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig) (Wav2Vec2-Conformer model)
* **wavlm** — [WavLMConfig](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMConfig) (WavLM model)
* **whisper** — [WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig) (Whisper model)
* **xclip** — [XCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/xclip#transformers.XCLIPConfig) (X-CLIP model)
* **xcodec** — [XcodecConfig](/docs/transformers/v4.56.2/en/model_doc/xcodec#transformers.XcodecConfig) (X-CODEC model)
* **xglm** — [XGLMConfig](/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMConfig) (XGLM model)
* **xlm** — [XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig) (XLM model)
* **xlm-prophetnet** — [XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig) (XLM-ProphetNet model)
* **xlm-roberta** — [XLMRobertaConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaXLConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig) (XLM-RoBERTa-XL model)
* **xlnet** — [XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig) (XLNet model)
* **xlstm** — [xLSTMConfig](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMConfig) (xLSTM model)
* **xmod** — [XmodConfig](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig) (X-MOD model)
* **yolos** — [YolosConfig](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosConfig) (YOLOS model)
* **yoso** — [YosoConfig](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig) (YOSO model)
* **zamba** — [ZambaConfig](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaConfig) (Zamba model)
* **zamba2** — [Zamba2Config](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2Config) (Zamba2 model)
* **zoedepth** — [ZoeDepthConfig](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthConfig) (ZoeDepth model)

Examples:


```
>>> from transformers import AutoConfig

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")

>>> # Download configuration from huggingface.co (user-uploaded) and cache.
>>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")

>>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
>>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")

>>> # Load a specific configuration file.
>>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")

>>> # Change some config attributes when loading a pretrained config.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
>>> config.output_attentions
True

>>> config, unused_kwargs = AutoConfig.from_pretrained(
...     "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
... )
>>> config.output_attentions
True

>>> unused_kwargs
{'foo': False}
```

#### register

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/configuration_auto.py#L1335)

( model\_type config exist\_ok = False  )

Parameters

* **model\_type** (`str`) — The model type like “bert” or “gpt”.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) — The config to register.

Register a new configuration for this class.

## AutoTokenizer

### class transformers.AutoTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/tokenization_auto.py#L917)

( )

This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
created with the [AutoTokenizer.from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/tokenization_auto.py#L931)

( pretrained\_model\_name\_or\_path \*inputs \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
    using the [save\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained) method, e.g., `./my_model_directory/`.
  + A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
    single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
    applicable to all derived classes)
* **inputs** (additional positional arguments, *optional*) —
  Will be passed along to the Tokenizer `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  The configuration object used to determine the tokenizer class to instantiate.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download the model weights and configuration files and override the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **subfolder** (`str`, *optional*) —
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
  facebook/rag-token-base), specify it here.
* **use\_fast** (`bool`, *optional*, defaults to `True`) —
  Use a [fast Rust-based tokenizer](https://huggingface.co/docs/tokenizers/index) if it is supported for
  a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer
  is returned instead.
* **tokenizer\_type** (`str`, *optional*) —
  Tokenizer type to be loaded.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the Tokenizer `__init__()` method. Can be used to set special tokens like
  `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
  `additional_special_tokens`. See parameters in the `__init__()` for more details.

Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

The tokenizer class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **aimv2** — [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) or [CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) (AIMv2 model)
* **albert** — `AlbertTokenizer` or [AlbertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertTokenizerFast) (ALBERT model)
* **align** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (ALIGN model)
* **arcee** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Arcee model)
* **aria** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Aria model)
* **aya\_vision** — [CohereTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereTokenizerFast) (AyaVision model)
* **bark** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (Bark model)
* **bart** — [BartTokenizer](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartTokenizer) or [BartTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartTokenizerFast) (BART model)
* **barthez** — [BarthezTokenizer](/docs/transformers/v4.56.2/en/model_doc/barthez#transformers.BarthezTokenizer) or [BarthezTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/barthez#transformers.BarthezTokenizerFast) (BARThez model)
* **bartpho** — [BartphoTokenizer](/docs/transformers/v4.56.2/en/model_doc/bartpho#transformers.BartphoTokenizer) (BARTpho model)
* **bert** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (BERT model)
* **bert-generation** — [BertGenerationTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationTokenizer) (Bert Generation model)
* **bert-japanese** — [BertJapaneseTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert-japanese#transformers.BertJapaneseTokenizer) (BertJapanese model)
* **bertweet** — [BertweetTokenizer](/docs/transformers/v4.56.2/en/model_doc/bertweet#transformers.BertweetTokenizer) (BERTweet model)
* **big\_bird** — [BigBirdTokenizer](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdTokenizer) or [BigBirdTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdTokenizerFast) (BigBird model)
* **bigbird\_pegasus** — [PegasusTokenizer](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusTokenizer) or [PegasusTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusTokenizerFast) (BigBird-Pegasus model)
* **biogpt** — [BioGptTokenizer](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptTokenizer) (BioGpt model)
* **bitnet** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (BitNet model)
* **blenderbot** — [BlenderbotTokenizer](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotTokenizer) or [BlenderbotTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotTokenizerFast) (Blenderbot model)
* **blenderbot-small** — [BlenderbotSmallTokenizer](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer) (BlenderbotSmall model)
* **blip** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (BLIP model)
* **blip-2** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (BLIP-2 model)
* **bloom** — [BloomTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomTokenizerFast) (BLOOM model)
* **bridgetower** — [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) or [RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast) (BridgeTower model)
* **bros** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (BROS model)
* **byt5** — [ByT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/byt5#transformers.ByT5Tokenizer) (ByT5 model)
* **camembert** — [CamembertTokenizer](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertTokenizer) or [CamembertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertTokenizerFast) (CamemBERT model)
* **canine** — [CanineTokenizer](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineTokenizer) (CANINE model)
* **chameleon** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Chameleon model)
* **chinese\_clip** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (Chinese-CLIP model)
* **clap** — [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) or [RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast) (CLAP model)
* **clip** — [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) or [CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) (CLIP model)
* **clipseg** — [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) or [CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) (CLIPSeg model)
* **clvp** — [ClvpTokenizer](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpTokenizer) (CLVP model)
* **code\_llama** — [CodeLlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/code_llama#transformers.CodeLlamaTokenizer) or [CodeLlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/code_llama#transformers.CodeLlamaTokenizerFast) (CodeLlama model)
* **codegen** — [CodeGenTokenizer](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenTokenizer) or [CodeGenTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenTokenizerFast) (CodeGen model)
* **cohere** — [CohereTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereTokenizerFast) (Cohere model)
* **cohere2** — [CohereTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereTokenizerFast) (Cohere2 model)
* **colpali** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (ColPali model)
* **colqwen2** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (ColQwen2 model)
* **convbert** — [ConvBertTokenizer](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertTokenizer) or [ConvBertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertTokenizerFast) (ConvBERT model)
* **cpm** — [CpmTokenizer](/docs/transformers/v4.56.2/en/model_doc/cpm#transformers.CpmTokenizer) or [CpmTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/cpm#transformers.CpmTokenizerFast) (CPM model)
* **cpmant** — [CpmAntTokenizer](/docs/transformers/v4.56.2/en/model_doc/cpmant#transformers.CpmAntTokenizer) (CPM-Ant model)
* **ctrl** — [CTRLTokenizer](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLTokenizer) (CTRL model)
* **data2vec-audio** — [Wav2Vec2CTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer) (Data2VecAudio model)
* **data2vec-text** — [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) or [RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast) (Data2VecText model)
* **dbrx** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (DBRX model)
* **deberta** — [DebertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaTokenizer) or [DebertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaTokenizerFast) (DeBERTa model)
* **deberta-v2** — [DebertaV2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Tokenizer) or [DebertaV2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2TokenizerFast) (DeBERTa-v2 model)
* **deepseek\_v2** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (DeepSeek-V2 model)
* **deepseek\_v3** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (DeepSeek-V3 model)
* **deepseek\_vl** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (DeepseekVL model)
* **deepseek\_vl\_hybrid** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (DeepseekVLHybrid model)
* **dia** — [DiaTokenizer](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaTokenizer) (Dia model)
* **diffllama** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (DiffLlama model)
* **distilbert** — [DistilBertTokenizer](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertTokenizer) or [DistilBertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertTokenizerFast) (DistilBERT model)
* **dpr** — [DPRQuestionEncoderTokenizer](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoderTokenizer) or [DPRQuestionEncoderTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoderTokenizerFast) (DPR model)
* **electra** — [ElectraTokenizer](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraTokenizer) or [ElectraTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraTokenizerFast) (ELECTRA model)
* **emu3** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (Emu3 model)
* **ernie** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (ERNIE model)
* **ernie4\_5** — [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Ernie4\_5 model)
* **ernie4\_5\_moe** — [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Ernie4\_5\_MoE model)
* **ernie\_m** — [ErnieMTokenizer](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMTokenizer) (ErnieM model)
* **esm** — [EsmTokenizer](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmTokenizer) (ESM model)
* **exaone4** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (EXAONE-4.0 model)
* **falcon** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (Falcon model)
* **falcon\_mamba** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (FalconMamba model)
* **fastspeech2\_conformer** — (FastSpeech2Conformer model)
* **flaubert** — [FlaubertTokenizer](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertTokenizer) (FlauBERT model)
* **fnet** — [FNetTokenizer](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetTokenizer) or [FNetTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetTokenizerFast) (FNet model)
* **fsmt** — [FSMTTokenizer](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTTokenizer) (FairSeq Machine-Translation model)
* **funnel** — [FunnelTokenizer](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelTokenizer) or [FunnelTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelTokenizerFast) (Funnel Transformer model)
* **gemma** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (Gemma model)
* **gemma2** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (Gemma2 model)
* **gemma3** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (Gemma3ForConditionalGeneration model)
* **gemma3\_text** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (Gemma3ForCausalLM model)
* **gemma3n** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (Gemma3nForConditionalGeneration model)
* **gemma3n\_text** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (Gemma3nForCausalLM model)
* **git** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (GIT model)
* **glm** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (GLM model)
* **glm4** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (GLM4 model)
* **glm4\_moe** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (Glm4MoE model)
* **glm4v** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (GLM4V model)
* **glm4v\_moe** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (GLM4VMOE model)
* **gpt-sw3** — [GPTSw3Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt-sw3#transformers.GPTSw3Tokenizer) (GPT-Sw3 model)
* **gpt2** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (OpenAI GPT-2 model)
* **gpt\_bigcode** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (GPTBigCode model)
* **gpt\_neo** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (GPT Neo model)
* **gpt\_neox** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (GPT NeoX model)
* **gpt\_neox\_japanese** — [GPTNeoXJapaneseTokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseTokenizer) (GPT NeoX Japanese model)
* **gpt\_oss** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (GptOss model)
* **gptj** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (GPT-J model)
* **gptsan-japanese** — [GPTSanJapaneseTokenizer](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseTokenizer) (GPTSAN-japanese model)
* **granite** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) (Granite model)
* **granitemoe** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) (GraniteMoeMoe model)
* **granitemoehybrid** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) (GraniteMoeHybrid model)
* **granitemoeshared** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) (GraniteMoeSharedMoe model)
* **grounding-dino** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (Grounding DINO model)
* **groupvit** — [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) or [CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) (GroupViT model)
* **helium** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (Helium model)
* **herbert** — [HerbertTokenizer](/docs/transformers/v4.56.2/en/model_doc/herbert#transformers.HerbertTokenizer) or [HerbertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/herbert#transformers.HerbertTokenizerFast) (HerBERT model)
* **hubert** — [Wav2Vec2CTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer) (Hubert model)
* **ibert** — [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) or [RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast) (I-BERT model)
* **idefics** — [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (IDEFICS model)
* **idefics2** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Idefics2 model)
* **idefics3** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Idefics3 model)
* **instructblip** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (InstructBLIP model)
* **instructblipvideo** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (InstructBlipVideo model)
* **internvl** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (InternVL model)
* **jamba** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Jamba model)
* **janus** — [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Janus model)
* **jetmoe** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (JetMoe model)
* **jukebox** — [JukeboxTokenizer](/docs/transformers/v4.56.2/en/model_doc/jukebox#transformers.JukeboxTokenizer) (Jukebox model)
* **kosmos-2** — [XLMRobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) or [XLMRobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast) (KOSMOS-2 model)
* **kosmos-2.5** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (KOSMOS-2.5 model)
* **layoutlm** — [LayoutLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMTokenizer) or [LayoutLMTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMTokenizerFast) (LayoutLM model)
* **layoutlmv2** — [LayoutLMv2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer) or [LayoutLMv2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2TokenizerFast) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer) or [LayoutLMv3TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3TokenizerFast) (LayoutLMv3 model)
* **layoutxlm** — [LayoutXLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer) or [LayoutXLMTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast) (LayoutXLM model)
* **led** — [LEDTokenizer](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDTokenizer) or [LEDTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDTokenizerFast) (LED model)
* **lilt** — [LayoutLMv3Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer) or [LayoutLMv3TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3TokenizerFast) (LiLT model)
* **llama** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (LLaMA model)
* **llama4** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Llama4 model)
* **llama4\_text** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Llama4ForCausalLM model)
* **llava** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (LLaVa model)
* **llava\_next** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (LLaVA-NeXT model)
* **llava\_next\_video** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (LLaVa-NeXT-Video model)
* **llava\_onevision** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (LLaVA-Onevision model)
* **longformer** — [LongformerTokenizer](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerTokenizer) or [LongformerTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerTokenizerFast) (Longformer model)
* **longt5** — [T5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer) or [T5TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5TokenizerFast) (LongT5 model)
* **luke** — [LukeTokenizer](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeTokenizer) (LUKE model)
* **lxmert** — [LxmertTokenizer](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertTokenizer) or [LxmertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertTokenizerFast) (LXMERT model)
* **m2m\_100** — [M2M100Tokenizer](/docs/transformers/v4.56.2/en/model_doc/m2m_100#transformers.M2M100Tokenizer) (M2M100 model)
* **mamba** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (Mamba model)
* **mamba2** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (mamba2 model)
* **marian** — [MarianTokenizer](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianTokenizer) (Marian model)
* **mbart** — [MBartTokenizer](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartTokenizer) or [MBartTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartTokenizerFast) (mBART model)
* **mbart50** — [MBart50Tokenizer](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBart50Tokenizer) or [MBart50TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBart50TokenizerFast) (mBART-50 model)
* **mega** — [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) or [RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast) (MEGA model)
* **megatron-bert** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (Megatron-BERT model)
* **metaclip\_2** — [XLMRobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) or [XLMRobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast) (MetaCLIP 2 model)
* **mgp-str** — [MgpstrTokenizer](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrTokenizer) (MGP-STR model)
* **minimax** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (MiniMax model)
* **mistral** — [MistralCommonTokenizer](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer) (Mistral model)
* **mixtral** — [MistralCommonTokenizer](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer) (Mixtral model)
* **mllama** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Mllama model)
* **mluke** — [MLukeTokenizer](/docs/transformers/v4.56.2/en/model_doc/mluke#transformers.MLukeTokenizer) (mLUKE model)
* **mm-grounding-dino** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (MM Grounding DINO model)
* **mobilebert** — [MobileBertTokenizer](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertTokenizer) or [MobileBertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertTokenizerFast) (MobileBERT model)
* **modernbert** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (ModernBERT model)
* **moonshine** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (Moonshine model)
* **moshi** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (Moshi model)
* **mpnet** — [MPNetTokenizer](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetTokenizer) or [MPNetTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetTokenizerFast) (MPNet model)
* **mpt** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (MPT model)
* **mra** — [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) or [RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast) (MRA model)
* **mt5** — [MT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Tokenizer) or [MT5TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5TokenizerFast) (MT5 model)
* **musicgen** — [T5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer) or [T5TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5TokenizerFast) (MusicGen model)
* **musicgen\_melody** — [T5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer) or [T5TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5TokenizerFast) (MusicGen Melody model)
* **mvp** — [MvpTokenizer](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpTokenizer) or [MvpTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpTokenizerFast) (MVP model)
* **myt5** — [MyT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/myt5#transformers.MyT5Tokenizer) (myt5 model)
* **nemotron** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (Nemotron model)
* **nezha** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (Nezha model)
* **nllb** — [NllbTokenizer](/docs/transformers/v4.56.2/en/model_doc/nllb#transformers.NllbTokenizer) or [NllbTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/nllb#transformers.NllbTokenizerFast) (NLLB model)
* **nllb-moe** — [NllbTokenizer](/docs/transformers/v4.56.2/en/model_doc/nllb#transformers.NllbTokenizer) or [NllbTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/nllb#transformers.NllbTokenizerFast) (NLLB-MOE model)
* **nystromformer** — `AlbertTokenizer` or [AlbertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertTokenizerFast) (Nyströmformer model)
* **olmo** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (OLMo model)
* **olmo2** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (OLMo2 model)
* **olmoe** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (OLMoE model)
* **omdet-turbo** — [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) or [CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) (OmDet-Turbo model)
* **oneformer** — [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) or [CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) (OneFormer model)
* **openai-gpt** — [OpenAIGPTTokenizer](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTTokenizer) or [OpenAIGPTTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTTokenizerFast) (OpenAI GPT model)
* **opt** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (OPT model)
* **owlv2** — [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) or [CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) (OWLv2 model)
* **owlvit** — [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) or [CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) (OWL-ViT model)
* **paligemma** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (PaliGemma model)
* **pegasus** — [PegasusTokenizer](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusTokenizer) or [PegasusTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusTokenizerFast) (Pegasus model)
* **pegasus\_x** — [PegasusTokenizer](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusTokenizer) or [PegasusTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusTokenizerFast) (PEGASUS-X model)
* **perceiver** — [PerceiverTokenizer](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverTokenizer) (Perceiver model)
* **persimmon** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Persimmon model)
* **phi** — [CodeGenTokenizer](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenTokenizer) or [CodeGenTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenTokenizerFast) (Phi model)
* **phi3** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Phi3 model)
* **phimoe** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Phimoe model)
* **phobert** — [PhobertTokenizer](/docs/transformers/v4.56.2/en/model_doc/phobert#transformers.PhobertTokenizer) (PhoBERT model)
* **pix2struct** — [T5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer) or [T5TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5TokenizerFast) (Pix2Struct model)
* **pixtral** — [MistralCommonTokenizer](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer) (Pixtral model)
* **plbart** — [PLBartTokenizer](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartTokenizer) (PLBart model)
* **prophetnet** — [ProphetNetTokenizer](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetTokenizer) (ProphetNet model)
* **qdqbert** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (QDQBert model)
* **qwen2** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (Qwen2 model)
* **qwen2\_5\_omni** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (Qwen2\_5Omni model)
* **qwen2\_5\_vl** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (Qwen2\_5\_VL model)
* **qwen2\_audio** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (Qwen2Audio model)
* **qwen2\_moe** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (Qwen2MoE model)
* **qwen2\_vl** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (Qwen2VL model)
* **qwen3** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (Qwen3 model)
* **qwen3\_moe** — [Qwen2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) or [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast) (Qwen3MoE model)
* **rag** — [RagTokenizer](/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagTokenizer) (RAG model)
* **realm** — [RealmTokenizer](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmTokenizer) or [RealmTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmTokenizerFast) (REALM model)
* **recurrent\_gemma** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (RecurrentGemma model)
* **reformer** — [ReformerTokenizer](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerTokenizer) or [ReformerTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerTokenizerFast) (Reformer model)
* **rembert** — [RemBertTokenizer](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertTokenizer) or [RemBertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertTokenizerFast) (RemBERT model)
* **retribert** — [RetriBertTokenizer](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertTokenizer) or [RetriBertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertTokenizerFast) (RetriBERT model)
* **roberta** — [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) or [RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) or [RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertTokenizer](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertTokenizer) (RoCBert model)
* **roformer** — [RoFormerTokenizer](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerTokenizer) or [RoFormerTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerTokenizerFast) (RoFormer model)
* **rwkv** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (RWKV model)
* **seamless\_m4t** — [SeamlessM4TTokenizer](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizer) or [SeamlessM4TTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizerFast) (SeamlessM4T model)
* **seamless\_m4t\_v2** — [SeamlessM4TTokenizer](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizer) or [SeamlessM4TTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizerFast) (SeamlessM4Tv2 model)
* **shieldgemma2** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (Shieldgemma2 model)
* **siglip** — [SiglipTokenizer](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipTokenizer) (SigLIP model)
* **siglip2** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (SigLIP2 model)
* **smollm3** — [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) (SmolLM3 model)
* **speech\_to\_text** — [Speech2TextTokenizer](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer) (Speech2Text model)
* **speech\_to\_text\_2** — [Speech2Text2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer) (Speech2Text2 model)
* **speecht5** — [SpeechT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer) (SpeechT5 model)
* **splinter** — [SplinterTokenizer](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterTokenizer) or [SplinterTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterTokenizerFast) (Splinter model)
* **squeezebert** — [SqueezeBertTokenizer](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertTokenizer) or [SqueezeBertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertTokenizerFast) (SqueezeBERT model)
* **stablelm** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (StableLm model)
* **starcoder2** — [GPT2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Tokenizer) or [GPT2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2TokenizerFast) (Starcoder2 model)
* **switch\_transformers** — [T5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer) or [T5TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5TokenizerFast) (SwitchTransformers model)
* **t5** — [T5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer) or [T5TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5TokenizerFast) (T5 model)
* **t5gemma** — [GemmaTokenizer](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizer) or [GemmaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaTokenizerFast) (T5Gemma model)
* **tapas** — [TapasTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasTokenizer) (TAPAS model)
* **tapex** — [TapexTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapex#transformers.TapexTokenizer) (TAPEX model)
* **transfo-xl** — [TransfoXLTokenizer](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLTokenizer) (Transformer-XL model)
* **tvp** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (TVP model)
* **udop** — [UdopTokenizer](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopTokenizer) or [UdopTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopTokenizerFast) (UDOP model)
* **umt5** — [T5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer) or [T5TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5TokenizerFast) (UMT5 model)
* **video\_llava** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (VideoLlava model)
* **vilt** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (ViLT model)
* **vipllava** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (VipLlava model)
* **visual\_bert** — [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) or [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) (VisualBERT model)
* **vits** — [VitsTokenizer](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsTokenizer) (VITS model)
* **voxtral** — [MistralCommonTokenizer](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer) (Voxtral model)
* **wav2vec2** — [Wav2Vec2CTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer) (Wav2Vec2 model)
* **wav2vec2-bert** — [Wav2Vec2CTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer) (Wav2Vec2-BERT model)
* **wav2vec2-conformer** — [Wav2Vec2CTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer) (Wav2Vec2-Conformer model)
* **wav2vec2\_phoneme** — [Wav2Vec2PhonemeCTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer) (Wav2Vec2Phoneme model)
* **whisper** — [WhisperTokenizer](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperTokenizer) or [WhisperTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperTokenizerFast) (Whisper model)
* **xclip** — [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) or [CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) (X-CLIP model)
* **xglm** — [XGLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMTokenizer) or [XGLMTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMTokenizerFast) (XGLM model)
* **xlm** — [XLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMTokenizer) (XLM model)
* **xlm-prophetnet** — [XLMProphetNetTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetTokenizer) (XLM-ProphetNet model)
* **xlm-roberta** — [XLMRobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) or [XLMRobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) or [XLMRobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast) (XLM-RoBERTa-XL model)
* **xlnet** — [XLNetTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer) or [XLNetTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizerFast) (XLNet model)
* **xlstm** — [GPTNeoXTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast) (xLSTM model)
* **xmod** — [XLMRobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) or [XLMRobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast) (X-MOD model)
* **yoso** — `AlbertTokenizer` or [AlbertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertTokenizerFast) (YOSO model)
* **zamba** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Zamba model)
* **zamba2** — [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer) or [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) (Zamba2 model)

Examples:


```
>>> from transformers import AutoTokenizer

>>> # Download vocabulary from huggingface.co and cache.
>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

>>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
>>> tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

>>> # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
>>> # tokenizer = AutoTokenizer.from_pretrained("./test/bert_saved_model/")

>>> # Download vocabulary from huggingface.co and define model-specific arguments
>>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", add_prefix_space=True)
```

#### register

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/tokenization_auto.py#L1159)

( config\_class slow\_tokenizer\_class = None fast\_tokenizer\_class = None exist\_ok = False  )

Parameters

* **config\_class** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The configuration corresponding to the model to register.
* **slow\_tokenizer\_class** (`PretrainedTokenizer`, *optional*) —
  The slow tokenizer to register.
* **fast\_tokenizer\_class** (`PretrainedTokenizerFast`, *optional*) —
  The fast tokenizer to register.

Register a new tokenizer in this mapping.

## AutoFeatureExtractor

### class transformers.AutoFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/feature_extraction_auto.py#L253)

( )

This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the
library when created with the [AutoFeatureExtractor.from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoFeatureExtractor.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/feature_extraction_auto.py#L267)

( pretrained\_model\_name\_or\_path \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained feature\_extractor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a feature extractor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g.,
    `./my_model_directory/`.
  + a path or url to a saved feature extractor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force to (re-)download the feature extractor files and override the cached versions
  if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
* **token** (`str` or *bool*, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **return\_unused\_kwargs** (`bool`, *optional*, defaults to `False`) —
  If `False`, then this function returns just the final feature extractor object. If `True`, then this
  functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused\_kwargs* is a dictionary
  consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
  `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **kwargs** (`dict[str, Any]`, *optional*) —
  The values in kwargs of any keys which are feature extractor attributes will be used to override the
  loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
  controlled by the `return_unused_kwargs` keyword parameter.

Instantiate one of the feature extractor classes of the library from a pretrained model vocabulary.

The feature extractor class to instantiate is selected based on the `model_type` property of the config object
(either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s
missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

* **audio-spectrogram-transformer** — [ASTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/audio-spectrogram-transformer#transformers.ASTFeatureExtractor) (Audio Spectrogram Transformer model)
* **beit** — [BeitFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitFeatureExtractor) (BEiT model)
* **chinese\_clip** — [ChineseCLIPFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPFeatureExtractor) (Chinese-CLIP model)
* **clap** — [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor) (CLAP model)
* **clip** — [CLIPFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPFeatureExtractor) (CLIP model)
* **clipseg** — [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) (CLIPSeg model)
* **clvp** — [ClvpFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpFeatureExtractor) (CLVP model)
* **conditional\_detr** — [ConditionalDetrFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrFeatureExtractor) (Conditional DETR model)
* **convnext** — [ConvNextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextFeatureExtractor) (ConvNeXT model)
* **cvt** — [ConvNextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextFeatureExtractor) (CvT model)
* **dac** — [DacFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacFeatureExtractor) (DAC model)
* **data2vec-audio** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (Data2VecAudio model)
* **data2vec-vision** — [BeitFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitFeatureExtractor) (Data2VecVision model)
* **deformable\_detr** — [DeformableDetrFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrFeatureExtractor) (Deformable DETR model)
* **deit** — [DeiTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTFeatureExtractor) (DeiT model)
* **detr** — [DetrFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrFeatureExtractor) (DETR model)
* **dia** — [DiaFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaFeatureExtractor) (Dia model)
* **dinat** — [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) (DiNAT model)
* **donut-swin** — [DonutFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutFeatureExtractor) (DonutSwin model)
* **dpt** — [DPTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTFeatureExtractor) (DPT model)
* **encodec** — [EncodecFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor) (EnCodec model)
* **flava** — [FlavaFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaFeatureExtractor) (FLAVA model)
* **gemma3n** — [Gemma3nAudioFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nAudioFeatureExtractor) (Gemma3nForConditionalGeneration model)
* **glpn** — [GLPNFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNFeatureExtractor) (GLPN model)
* **granite\_speech** — [GraniteSpeechFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechFeatureExtractor) (GraniteSpeech model)
* **groupvit** — [CLIPFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPFeatureExtractor) (GroupViT model)
* **hubert** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (Hubert model)
* **imagegpt** — [ImageGPTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTFeatureExtractor) (ImageGPT model)
* **kyutai\_speech\_to\_text** — [KyutaiSpeechToTextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextFeatureExtractor) (KyutaiSpeechToText model)
* **layoutlmv2** — [LayoutLMv2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2FeatureExtractor) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3FeatureExtractor) (LayoutLMv3 model)
* **levit** — [LevitFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitFeatureExtractor) (LeViT model)
* **maskformer** — [MaskFormerFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerFeatureExtractor) (MaskFormer model)
* **mctct** — [MCTCTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/mctct#transformers.MCTCTFeatureExtractor) (M-CTC-T model)
* **mimi** — [EncodecFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor) (Mimi model)
* **mobilenet\_v1** — [MobileNetV1FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1FeatureExtractor) (MobileNetV1 model)
* **mobilenet\_v2** — [MobileNetV2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2FeatureExtractor) (MobileNetV2 model)
* **mobilevit** — [MobileViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTFeatureExtractor) (MobileViT model)
* **moonshine** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (Moonshine model)
* **moshi** — [EncodecFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor) (Moshi model)
* **nat** — [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) (NAT model)
* **owlvit** — `OwlViTFeatureExtractor` (OWL-ViT model)
* **perceiver** — [PerceiverFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverFeatureExtractor) (Perceiver model)
* **phi4\_multimodal** — [Phi4MultimodalFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalFeatureExtractor) (Phi4Multimodal model)
* **poolformer** — [PoolFormerFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerFeatureExtractor) (PoolFormer model)
* **pop2piano** — [Pop2PianoFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/pop2piano#transformers.models.pop2piano.feature_extraction_pop2piano._LazyModule.__getattr__.%3Clocals%3E.Placeholder) (Pop2Piano model)
* **regnet** — [ConvNextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextFeatureExtractor) (RegNet model)
* **resnet** — [ConvNextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextFeatureExtractor) (ResNet model)
* **seamless\_m4t** — [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) (SeamlessM4T model)
* **seamless\_m4t\_v2** — [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) (SeamlessM4Tv2 model)
* **segformer** — [SegformerFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerFeatureExtractor) (SegFormer model)
* **sew** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (SEW model)
* **sew-d** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (SEW-D model)
* **speech\_to\_text** — [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) (Speech2Text model)
* **speecht5** — [SpeechT5FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor) (SpeechT5 model)
* **swiftformer** — [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) (SwiftFormer model)
* **swin** — [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) (Swin Transformer model)
* **swinv2** — [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) (Swin Transformer V2 model)
* **table-transformer** — [DetrFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrFeatureExtractor) (Table Transformer model)
* **timesformer** — [VideoMAEFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEFeatureExtractor) (TimeSformer model)
* **tvlt** — [TvltFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltFeatureExtractor) (TVLT model)
* **unispeech** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (UniSpeech model)
* **unispeech-sat** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (UniSpeechSat model)
* **univnet** — [UnivNetFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetFeatureExtractor) (UnivNet model)
* **van** — [ConvNextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextFeatureExtractor) (VAN model)
* **videomae** — [VideoMAEFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEFeatureExtractor) (VideoMAE model)
* **vilt** — [ViltFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltFeatureExtractor) (ViLT model)
* **vit** — [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) (ViT model)
* **vit\_mae** — [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) (ViTMAE model)
* **vit\_msn** — [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) (ViTMSN model)
* **wav2vec2** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (Wav2Vec2 model)
* **wav2vec2-bert** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (Wav2Vec2-BERT model)
* **wav2vec2-conformer** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (Wav2Vec2-Conformer model)
* **wavlm** — [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) (WavLM model)
* **whisper** — [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor) (Whisper model)
* **xclip** — [CLIPFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPFeatureExtractor) (X-CLIP model)
* **xcodec** — [DacFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacFeatureExtractor) (X-CODEC model)
* **yolos** — [YolosFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosFeatureExtractor) (YOLOS model)

Passing `token=True` is required when you want to use a private model.

Examples:


```
>>> from transformers import AutoFeatureExtractor

>>> # Download feature extractor from huggingface.co and cache.
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

>>> # If feature extractor files are in a directory (e.g. feature extractor was saved using *save_pretrained('./test/saved_model/')*)
>>> # feature_extractor = AutoFeatureExtractor.from_pretrained("./test/saved_model/")
```

#### register

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/feature_extraction_auto.py#L407)

( config\_class feature\_extractor\_class exist\_ok = False  )

Parameters

* **config\_class** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The configuration corresponding to the model to register.
* **feature\_extractor\_class** (`FeatureExtractorMixin`) — The feature extractor to register.

Register a new feature extractor for this class.

## AutoImageProcessor

### class transformers.AutoImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/image_processing_auto.py#L351)

( )

This is a generic image processor class that will be instantiated as one of the image processor classes of the
library when created with the [AutoImageProcessor.from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/image_processing_auto.py#L365)

( pretrained\_model\_name\_or\_path \*inputs \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained image\_processor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a image processor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin.save_pretrained) method, e.g.,
    `./my_model_directory/`.
  + a path or url to a saved image processor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model image processor should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force to (re-)download the image processor files and override the cached versions if
  they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
* **token** (`str` or *bool*, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **use\_fast** (`bool`, *optional*, defaults to `False`) —
  Use a fast torchvision-base image processor if it is supported for a given model.
  If a fast image processor is not available for a given model, a normal numpy-based image processor
  is returned instead.
* **return\_unused\_kwargs** (`bool`, *optional*, defaults to `False`) —
  If `False`, then this function returns just the final image processor object. If `True`, then this
  functions returns a `Tuple(image_processor, unused_kwargs)` where *unused\_kwargs* is a dictionary
  consisting of the key/value pairs whose keys are not image processor attributes: i.e., the part of
  `kwargs` which has not been used to update `image_processor` and is otherwise ignored.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **image\_processor\_filename** (`str`, *optional*, defaults to `"config.json"`) —
  The name of the file in the model directory to use for the image processor config.
* **kwargs** (`dict[str, Any]`, *optional*) —
  The values in kwargs of any keys which are image processor attributes will be used to override the
  loaded values. Behavior concerning key/value pairs whose keys are *not* image processor attributes is
  controlled by the `return_unused_kwargs` keyword parameter.

Instantiate one of the image processor classes of the library from a pretrained model vocabulary.

The image processor class to instantiate is selected based on the `model_type` property of the config object
(either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s
missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

* **aimv2** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (AIMv2 model)
* **aimv2\_vision\_model** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (Aimv2VisionModel model)
* **align** — [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) or [EfficientNetImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessorFast) (ALIGN model)
* **aria** — [AriaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaImageProcessor) (Aria model)
* **beit** — [BeitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessor) or [BeitImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessorFast) (BEiT model)
* **bit** — [BitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitImageProcessor) or [BitImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitImageProcessorFast) (BiT model)
* **blip** — [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) or [BlipImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessorFast) (BLIP model)
* **blip-2** — [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) or [BlipImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessorFast) (BLIP-2 model)
* **bridgetower** — [BridgeTowerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/bridgetower#transformers.BridgeTowerImageProcessor) or [BridgeTowerImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/bridgetower#transformers.BridgeTowerImageProcessorFast) (BridgeTower model)
* **chameleon** — [ChameleonImageProcessor](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonImageProcessor) or [ChameleonImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonImageProcessorFast) (Chameleon model)
* **chinese\_clip** — [ChineseCLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPImageProcessor) or [ChineseCLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPImageProcessorFast) (Chinese-CLIP model)
* **clip** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (CLIP model)
* **clipseg** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (CLIPSeg model)
* **cohere2\_vision** — [Cohere2VisionImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/cohere2_vision#transformers.Cohere2VisionImageProcessorFast) (Cohere2Vision model)
* **conditional\_detr** — [ConditionalDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrImageProcessor) or [ConditionalDetrImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrImageProcessorFast) (Conditional DETR model)
* **convnext** — [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) or [ConvNextImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessorFast) (ConvNeXT model)
* **convnextv2** — [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) or [ConvNextImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessorFast) (ConvNeXTV2 model)
* **cvt** — [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) or [ConvNextImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessorFast) (CvT model)
* **data2vec-vision** — [BeitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessor) or [BeitImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessorFast) (Data2VecVision model)
* **deepseek\_vl** — [DeepseekVLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLImageProcessor) or [DeepseekVLImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLImageProcessorFast) (DeepseekVL model)
* **deepseek\_vl\_hybrid** — [DeepseekVLHybridImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridImageProcessor) or [DeepseekVLHybridImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridImageProcessorFast) (DeepseekVLHybrid model)
* **deformable\_detr** — [DeformableDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrImageProcessor) or [DeformableDetrImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrImageProcessorFast) (Deformable DETR model)
* **deit** — [DeiTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTImageProcessor) or [DeiTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTImageProcessorFast) (DeiT model)
* **depth\_anything** — [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor) or [DPTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessorFast) (Depth Anything model)
* **depth\_pro** — [DepthProImageProcessor](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProImageProcessor) or [DepthProImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProImageProcessorFast) (DepthPro model)
* **deta** — [DetaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaImageProcessor) (DETA model)
* **detr** — [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor) or [DetrImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessorFast) (DETR model)
* **dinat** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (DiNAT model)
* **dinov2** — [BitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitImageProcessor) or [BitImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitImageProcessorFast) (DINOv2 model)
* **dinov3\_vit** — [DINOv3ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ViTImageProcessorFast) (DINOv3 ViT model)
* **donut-swin** — [DonutImageProcessor](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutImageProcessor) or [DonutImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutImageProcessorFast) (DonutSwin model)
* **dpt** — [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor) or [DPTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessorFast) (DPT model)
* **efficientformer** — [EfficientFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerImageProcessor) (EfficientFormer model)
* **efficientloftr** — [EfficientLoFTRImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRImageProcessor) (EfficientLoFTR model)
* **efficientnet** — [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) or [EfficientNetImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessorFast) (EfficientNet model)
* **eomt** — [EomtImageProcessor](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtImageProcessor) or [EomtImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtImageProcessorFast) (EoMT model)
* **flava** — [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor) or [FlavaImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessorFast) (FLAVA model)
* **focalnet** — [BitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitImageProcessor) or [BitImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitImageProcessorFast) (FocalNet model)
* **fuyu** — [FuyuImageProcessor](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor) (Fuyu model)
* **gemma3** — [Gemma3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor) or [Gemma3ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessorFast) (Gemma3ForConditionalGeneration model)
* **gemma3n** — [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor) or [SiglipImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessorFast) (Gemma3nForConditionalGeneration model)
* **git** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (GIT model)
* **glm4v** — [Glm4vImageProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vImageProcessor) or [Glm4vImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vImageProcessorFast) (GLM4V model)
* **glpn** — [GLPNImageProcessor](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNImageProcessor) (GLPN model)
* **got\_ocr2** — [GotOcr2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessor) or [GotOcr2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ImageProcessorFast) (GOT-OCR2 model)
* **grounding-dino** — [GroundingDinoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoImageProcessor) or [GroundingDinoImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoImageProcessorFast) (Grounding DINO model)
* **groupvit** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (GroupViT model)
* **hiera** — [BitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitImageProcessor) or [BitImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitImageProcessorFast) (Hiera model)
* **idefics** — [IdeficsImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsImageProcessor) (IDEFICS model)
* **idefics2** — [Idefics2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ImageProcessor) or [Idefics2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ImageProcessorFast) (Idefics2 model)
* **idefics3** — [Idefics3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3ImageProcessor) or [Idefics3ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3ImageProcessorFast) (Idefics3 model)
* **ijepa** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (I-JEPA model)
* **imagegpt** — [ImageGPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTImageProcessor) (ImageGPT model)
* **instructblip** — [BlipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessor) or [BlipImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipImageProcessorFast) (InstructBLIP model)
* **instructblipvideo** — [InstructBlipVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoImageProcessor) (InstructBlipVideo model)
* **janus** — [JanusImageProcessor](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusImageProcessor) or [JanusImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusImageProcessorFast) (Janus model)
* **kosmos-2** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (KOSMOS-2 model)
* **kosmos-2.5** — [Kosmos2\_5ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5ImageProcessor) or [Kosmos2\_5ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5ImageProcessorFast) (KOSMOS-2.5 model)
* **layoutlmv2** — [LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor) or [LayoutLMv2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessorFast) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) or [LayoutLMv3ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessorFast) (LayoutLMv3 model)
* **levit** — [LevitImageProcessor](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessor) or [LevitImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitImageProcessorFast) (LeViT model)
* **lightglue** — [LightGlueImageProcessor](/docs/transformers/v4.56.2/en/model_doc/lightglue#transformers.LightGlueImageProcessor) (LightGlue model)
* **llama4** — `Llama4ImageProcessor` or [Llama4ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ImageProcessorFast) (Llama4 model)
* **llava** — [LlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaImageProcessor) or [LlavaImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaImageProcessorFast) (LLaVa model)
* **llava\_next** — [LlavaNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextImageProcessor) or [LlavaNextImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/llava_next#transformers.LlavaNextImageProcessorFast) (LLaVA-NeXT model)
* **llava\_next\_video** — [LlavaNextVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoImageProcessor) (LLaVa-NeXT-Video model)
* **llava\_onevision** — [LlavaOnevisionImageProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionImageProcessor) or [LlavaOnevisionImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionImageProcessorFast) (LLaVA-Onevision model)
* **mask2former** — [Mask2FormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor) or [Mask2FormerImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/mask2former#transformers.Mask2FormerImageProcessorFast) (Mask2Former model)
* **maskformer** — [MaskFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor) or [MaskFormerImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessorFast) (MaskFormer model)
* **metaclip\_2** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (MetaCLIP 2 model)
* **mgp-str** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (MGP-STR model)
* **mistral3** — [PixtralImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessor) or [PixtralImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessorFast) (Mistral3 model)
* **mlcd** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (MLCD model)
* **mllama** — [MllamaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor) (Mllama model)
* **mm-grounding-dino** — [GroundingDinoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoImageProcessor) or [GroundingDinoImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoImageProcessorFast) (MM Grounding DINO model)
* **mobilenet\_v1** — [MobileNetV1ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1ImageProcessor) or [MobileNetV1ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1ImageProcessorFast) (MobileNetV1 model)
* **mobilenet\_v2** — [MobileNetV2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2ImageProcessor) or [MobileNetV2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2ImageProcessorFast) (MobileNetV2 model)
* **mobilevit** — [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) or [MobileViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessorFast) (MobileViT model)
* **mobilevitv2** — [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) or [MobileViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessorFast) (MobileViTV2 model)
* **nat** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (NAT model)
* **nougat** — [NougatImageProcessor](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatImageProcessor) or [NougatImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatImageProcessorFast) (Nougat model)
* **oneformer** — [OneFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor) or [OneFormerImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessorFast) (OneFormer model)
* **ovis2** — [Ovis2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ImageProcessor) or [Ovis2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ImageProcessorFast) (Ovis2 model)
* **owlv2** — [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor) or [Owlv2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessorFast) (OWLv2 model)
* **owlvit** — [OwlViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTImageProcessor) or [OwlViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTImageProcessorFast) (OWL-ViT model)
* **paligemma** — [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor) or [SiglipImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessorFast) (PaliGemma model)
* **perceiver** — [PerceiverImageProcessor](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverImageProcessor) or [PerceiverImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverImageProcessorFast) (Perceiver model)
* **perception\_lm** — [PerceptionLMImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMImageProcessorFast) (PerceptionLM model)
* **phi4\_multimodal** — [Phi4MultimodalImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalImageProcessorFast) (Phi4Multimodal model)
* **pix2struct** — [Pix2StructImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructImageProcessor) (Pix2Struct model)
* **pixtral** — [PixtralImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessor) or [PixtralImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessorFast) (Pixtral model)
* **poolformer** — [PoolFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerImageProcessor) or [PoolFormerImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerImageProcessorFast) (PoolFormer model)
* **prompt\_depth\_anything** — [PromptDepthAnythingImageProcessor](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingImageProcessor) (PromptDepthAnything model)
* **pvt** — [PvtImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtImageProcessor) or [PvtImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtImageProcessorFast) (PVT model)
* **pvt\_v2** — [PvtImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtImageProcessor) or [PvtImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtImageProcessorFast) (PVTv2 model)
* **qwen2\_5\_vl** — [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) or [Qwen2VLImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessorFast) (Qwen2\_5\_VL model)
* **qwen2\_vl** — [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) or [Qwen2VLImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessorFast) (Qwen2VL model)
* **regnet** — [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) or [ConvNextImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessorFast) (RegNet model)
* **resnet** — [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) or [ConvNextImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessorFast) (ResNet model)
* **rt\_detr** — [RTDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrImageProcessor) or [RTDetrImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrImageProcessorFast) (RT-DETR model)
* **sam** — [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor) or [SamImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessorFast) (SAM model)
* **sam2** — [Sam2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2ImageProcessorFast) (SAM2 model)
* **sam\_hq** — [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor) or [SamImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessorFast) (SAM-HQ model)
* **segformer** — [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor) or [SegformerImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessorFast) (SegFormer model)
* **seggpt** — [SegGptImageProcessor](/docs/transformers/v4.56.2/en/model_doc/seggpt#transformers.SegGptImageProcessor) (SegGPT model)
* **shieldgemma2** — [Gemma3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor) or [Gemma3ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessorFast) (Shieldgemma2 model)
* **siglip** — [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor) or [SiglipImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessorFast) (SigLIP model)
* **siglip2** — [Siglip2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessor) or [Siglip2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ImageProcessorFast) (SigLIP2 model)
* **smolvlm** — [SmolVLMImageProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMImageProcessor) or [SmolVLMImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMImageProcessorFast) (SmolVLM model)
* **superglue** — [SuperGlueImageProcessor](/docs/transformers/v4.56.2/en/model_doc/superglue#transformers.SuperGlueImageProcessor) (SuperGlue model)
* **superpoint** — [SuperPointImageProcessor](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointImageProcessor) or [SuperPointImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointImageProcessorFast) (SuperPoint model)
* **swiftformer** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (SwiftFormer model)
* **swin** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (Swin Transformer model)
* **swin2sr** — [Swin2SRImageProcessor](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRImageProcessor) or [Swin2SRImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRImageProcessorFast) (Swin2SR model)
* **swinv2** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (Swin Transformer V2 model)
* **table-transformer** — [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor) or [DetrImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessorFast) (Table Transformer model)
* **textnet** — [TextNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/textnet#transformers.TextNetImageProcessor) or [TextNetImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/textnet#transformers.TextNetImageProcessorFast) (TextNet model)
* **timesformer** — [VideoMAEImageProcessor](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEImageProcessor) (TimeSformer model)
* **timm\_wrapper** — [TimmWrapperImageProcessor](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperImageProcessor) (TimmWrapperModel model)
* **tvlt** — [TvltImageProcessor](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltImageProcessor) (TVLT model)
* **tvp** — [TvpImageProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpImageProcessor) or [TvpImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpImageProcessorFast) (TVP model)
* **udop** — [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) or [LayoutLMv3ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessorFast) (UDOP model)
* **upernet** — [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor) or [SegformerImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessorFast) (UPerNet model)
* **van** — [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) or [ConvNextImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessorFast) (VAN model)
* **videomae** — [VideoMAEImageProcessor](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEImageProcessor) (VideoMAE model)
* **vilt** — [ViltImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor) or [ViltImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessorFast) (ViLT model)
* **vipllava** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (VipLlava model)
* **vit** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (ViT model)
* **vit\_hybrid** — [ViTHybridImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit_hybrid#transformers.ViTHybridImageProcessor) (ViT Hybrid model)
* **vit\_mae** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (ViTMAE model)
* **vit\_msn** — [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) or [ViTImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessorFast) (ViTMSN model)
* **vitmatte** — [VitMatteImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vitmatte#transformers.VitMatteImageProcessor) or [VitMatteImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/vitmatte#transformers.VitMatteImageProcessorFast) (ViTMatte model)
* **xclip** — [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) or [CLIPImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessorFast) (X-CLIP model)
* **yolos** — [YolosImageProcessor](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosImageProcessor) or [YolosImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosImageProcessorFast) (YOLOS model)
* **zoedepth** — [ZoeDepthImageProcessor](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthImageProcessor) or [ZoeDepthImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthImageProcessorFast) (ZoeDepth model)

Passing `token=True` is required when you want to use a private model.

Examples:


```
>>> from transformers import AutoImageProcessor

>>> # Download image processor from huggingface.co and cache.
>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

>>> # If image processor files are in a directory (e.g. image processor was saved using *save_pretrained('./test/saved_model/')*)
>>> # image_processor = AutoImageProcessor.from_pretrained("./test/saved_model/")
```

#### register

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/image_processing_auto.py#L627)

( config\_class image\_processor\_class = None slow\_image\_processor\_class = None fast\_image\_processor\_class = None exist\_ok = False  )

Parameters

* **config\_class** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The configuration corresponding to the model to register.
* **image\_processor\_class** ([ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)) — The image processor to register.

Register a new image processor for this class.

## AutoVideoProcessor

### class transformers.AutoVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/video_processing_auto.py#L202)

( )

This is a generic video processor class that will be instantiated as one of the video processor classes of the
library when created with the [AutoVideoProcessor.from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoVideoProcessor.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/video_processing_auto.py#L216)

( pretrained\_model\_name\_or\_path \*inputs \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained video\_processor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a video processor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/video_processor#transformers.BaseVideoProcessor.save_pretrained) method, e.g.,
    `./my_model_directory/`.
  + a path or url to a saved video processor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model video processor should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force to (re-)download the video processor files and override the cached versions if
  they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
* **token** (`str` or *bool*, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **return\_unused\_kwargs** (`bool`, *optional*, defaults to `False`) —
  If `False`, then this function returns just the final video processor object. If `True`, then this
  functions returns a `Tuple(video_processor, unused_kwargs)` where *unused\_kwargs* is a dictionary
  consisting of the key/value pairs whose keys are not video processor attributes: i.e., the part of
  `kwargs` which has not been used to update `video_processor` and is otherwise ignored.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **kwargs** (`dict[str, Any]`, *optional*) —
  The values in kwargs of any keys which are video processor attributes will be used to override the
  loaded values. Behavior concerning key/value pairs whose keys are *not* video processor attributes is
  controlled by the `return_unused_kwargs` keyword parameter.

Instantiate one of the video processor classes of the library from a pretrained model vocabulary.

The video processor class to instantiate is selected based on the `model_type` property of the config object
(either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s
missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

* **glm4v** — [Glm4vVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vVideoProcessor) (GLM4V model)
* **instructblip** — [InstructBlipVideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoVideoProcessor) (InstructBLIP model)
* **instructblipvideo** — [InstructBlipVideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoVideoProcessor) (InstructBlipVideo model)
* **internvl** — [InternVLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLVideoProcessor) (InternVL model)
* **llava\_next\_video** — [LlavaNextVideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoVideoProcessor) (LLaVa-NeXT-Video model)
* **llava\_onevision** — [LlavaOnevisionVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionVideoProcessor) (LLaVA-Onevision model)
* **qwen2\_5\_omni** — [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor) (Qwen2\_5Omni model)
* **qwen2\_5\_vl** — [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor) (Qwen2\_5\_VL model)
* **qwen2\_vl** — [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor) (Qwen2VL model)
* **sam2\_video** — [Sam2VideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoVideoProcessor) (Sam2VideoModel model)
* **smolvlm** — [SmolVLMVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMVideoProcessor) (SmolVLM model)
* **video\_llava** — [VideoLlavaVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/video_llava#transformers.VideoLlavaVideoProcessor) (VideoLlava model)
* **vjepa2** — [VJEPA2VideoProcessor](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2VideoProcessor) (VJEPA2Model model)

Passing `token=True` is required when you want to use a private model.

Examples:


```
>>> from transformers import AutoVideoProcessor

>>> # Download video processor from huggingface.co and cache.
>>> video_processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")

>>> # If video processor files are in a directory (e.g. video processor was saved using *save_pretrained('./test/saved_model/')*)
>>> # video_processor = AutoVideoProcessor.from_pretrained("./test/saved_model/")
```

#### register

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/video_processing_auto.py#L371)

( config\_class video\_processor\_class exist\_ok = False  )

Parameters

* **config\_class** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The configuration corresponding to the model to register.
* **video\_processor\_class** ([BaseVideoProcessor](/docs/transformers/v4.56.2/en/main_classes/video_processor#transformers.BaseVideoProcessor)) —
  The video processor to register.

Register a new video processor for this class.

## AutoProcessor

### class transformers.AutoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/processing_auto.py#L183)

( )

This is a generic processor class that will be instantiated as one of the processor classes of the library when
created with the [AutoProcessor.from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor.from_pretrained) class method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/processing_auto.py#L197)

( pretrained\_model\_name\_or\_path \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained feature\_extractor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a processor files saved using the `save_pretrained()` method,
    e.g., `./my_model_directory/`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force to (re-)download the feature extractor files and override the cached versions
  if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
* **token** (`str` or *bool*, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **return\_unused\_kwargs** (`bool`, *optional*, defaults to `False`) —
  If `False`, then this function returns just the final feature extractor object. If `True`, then this
  functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused\_kwargs* is a dictionary
  consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
  `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **kwargs** (`dict[str, Any]`, *optional*) —
  The values in kwargs of any keys which are feature extractor attributes will be used to override the
  loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
  controlled by the `return_unused_kwargs` keyword parameter.

Instantiate one of the processor classes of the library from a pretrained model vocabulary.

The processor class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible):

* **aimv2** — [CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) (AIMv2 model)
* **align** — [AlignProcessor](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignProcessor) (ALIGN model)
* **altclip** — [AltCLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPProcessor) (AltCLIP model)
* **aria** — [AriaProcessor](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaProcessor) (Aria model)
* **aya\_vision** — [AyaVisionProcessor](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionProcessor) (AyaVision model)
* **bark** — [BarkProcessor](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkProcessor) (Bark model)
* **blip** — [BlipProcessor](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipProcessor) (BLIP model)
* **blip-2** — [Blip2Processor](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Processor) (BLIP-2 model)
* **bridgetower** — [BridgeTowerProcessor](/docs/transformers/v4.56.2/en/model_doc/bridgetower#transformers.BridgeTowerProcessor) (BridgeTower model)
* **chameleon** — [ChameleonProcessor](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonProcessor) (Chameleon model)
* **chinese\_clip** — [ChineseCLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPProcessor) (Chinese-CLIP model)
* **clap** — [ClapProcessor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapProcessor) (CLAP model)
* **clip** — [CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) (CLIP model)
* **clipseg** — [CLIPSegProcessor](/docs/transformers/v4.56.2/en/model_doc/clipseg#transformers.CLIPSegProcessor) (CLIPSeg model)
* **clvp** — [ClvpProcessor](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpProcessor) (CLVP model)
* **cohere2\_vision** — [Cohere2VisionProcessor](/docs/transformers/v4.56.2/en/model_doc/cohere2_vision#transformers.Cohere2VisionProcessor) (Cohere2Vision model)
* **colpali** — [ColPaliProcessor](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliProcessor) (ColPali model)
* **colqwen2** — [ColQwen2Processor](/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2Processor) (ColQwen2 model)
* **deepseek\_vl** — [DeepseekVLProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLProcessor) (DeepseekVL model)
* **deepseek\_vl\_hybrid** — [DeepseekVLHybridProcessor](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridProcessor) (DeepseekVLHybrid model)
* **dia** — [DiaProcessor](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor) (Dia model)
* **emu3** — [Emu3Processor](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Processor) (Emu3 model)
* **evolla** — [EvollaProcessor](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaProcessor) (Evolla model)
* **flava** — [FlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaProcessor) (FLAVA model)
* **florence2** — [Florence2Processor](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Processor) (Florence2 model)
* **fuyu** — [FuyuProcessor](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuProcessor) (Fuyu model)
* **gemma3** — [Gemma3Processor](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Processor) (Gemma3ForConditionalGeneration model)
* **gemma3n** — [Gemma3nProcessor](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nProcessor) (Gemma3nForConditionalGeneration model)
* **git** — [GitProcessor](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitProcessor) (GIT model)
* **glm4v** — [Glm4vProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vProcessor) (GLM4V model)
* **glm4v\_moe** — [Glm4vProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vProcessor) (GLM4VMOE model)
* **got\_ocr2** — [GotOcr2Processor](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Processor) (GOT-OCR2 model)
* **granite\_speech** — [GraniteSpeechProcessor](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechProcessor) (GraniteSpeech model)
* **grounding-dino** — [GroundingDinoProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoProcessor) (Grounding DINO model)
* **groupvit** — [CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) (GroupViT model)
* **hubert** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (Hubert model)
* **idefics** — [IdeficsProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsProcessor) (IDEFICS model)
* **idefics2** — [Idefics2Processor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Processor) (Idefics2 model)
* **idefics3** — [Idefics3Processor](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3Processor) (Idefics3 model)
* **instructblip** — [InstructBlipProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipProcessor) (InstructBLIP model)
* **instructblipvideo** — [InstructBlipVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoProcessor) (InstructBlipVideo model)
* **internvl** — [InternVLProcessor](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLProcessor) (InternVL model)
* **janus** — [JanusProcessor](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusProcessor) (Janus model)
* **kosmos-2** — [Kosmos2Processor](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Processor) (KOSMOS-2 model)
* **kosmos-2.5** — [Kosmos2\_5Processor](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Processor) (KOSMOS-2.5 model)
* **kyutai\_speech\_to\_text** — [KyutaiSpeechToTextProcessor](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextProcessor) (KyutaiSpeechToText model)
* **layoutlmv2** — [LayoutLMv2Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Processor) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) (LayoutLMv3 model)
* **llama4** — [Llama4Processor](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Processor) (Llama4 model)
* **llava** — [LlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaProcessor) (LLaVa model)
* **llava\_next** — [LlavaNextProcessor](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextProcessor) (LLaVA-NeXT model)
* **llava\_next\_video** — [LlavaNextVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoProcessor) (LLaVa-NeXT-Video model)
* **llava\_onevision** — [LlavaOnevisionProcessor](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionProcessor) (LLaVA-Onevision model)
* **markuplm** — [MarkupLMProcessor](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMProcessor) (MarkupLM model)
* **mctct** — [MCTCTProcessor](/docs/transformers/v4.56.2/en/model_doc/mctct#transformers.MCTCTProcessor) (M-CTC-T model)
* **metaclip\_2** — [CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) (MetaCLIP 2 model)
* **mgp-str** — [MgpstrProcessor](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrProcessor) (MGP-STR model)
* **mistral3** — [PixtralProcessor](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralProcessor) (Mistral3 model)
* **mllama** — [MllamaProcessor](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaProcessor) (Mllama model)
* **mm-grounding-dino** — [GroundingDinoProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoProcessor) (MM Grounding DINO model)
* **moonshine** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (Moonshine model)
* **oneformer** — [OneFormerProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerProcessor) (OneFormer model)
* **ovis2** — [Ovis2Processor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Processor) (Ovis2 model)
* **owlv2** — [Owlv2Processor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Processor) (OWLv2 model)
* **owlvit** — [OwlViTProcessor](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTProcessor) (OWL-ViT model)
* **paligemma** — [PaliGemmaProcessor](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaProcessor) (PaliGemma model)
* **perception\_lm** — [PerceptionLMProcessor](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMProcessor) (PerceptionLM model)
* **phi4\_multimodal** — [Phi4MultimodalProcessor](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalProcessor) (Phi4Multimodal model)
* **pix2struct** — [Pix2StructProcessor](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructProcessor) (Pix2Struct model)
* **pixtral** — [PixtralProcessor](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralProcessor) (Pixtral model)
* **pop2piano** — [Pop2PianoProcessor](/docs/transformers/v4.56.2/en/model_doc/pop2piano#transformers.models.pop2piano.processing_pop2piano._LazyModule.__getattr__.%3Clocals%3E.Placeholder) (Pop2Piano model)
* **qwen2\_5\_omni** — [Qwen2\_5OmniProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniProcessor) (Qwen2\_5Omni model)
* **qwen2\_5\_vl** — [Qwen2\_5\_VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLProcessor) (Qwen2\_5\_VL model)
* **qwen2\_audio** — [Qwen2AudioProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioProcessor) (Qwen2Audio model)
* **qwen2\_vl** — [Qwen2VLProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLProcessor) (Qwen2VL model)
* **sam** — [SamProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamProcessor) (SAM model)
* **sam2** — [Sam2Processor](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Processor) (SAM2 model)
* **sam\_hq** — [SamHQProcessor](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQProcessor) (SAM-HQ model)
* **seamless\_m4t** — [SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor) (SeamlessM4T model)
* **sew** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (SEW model)
* **sew-d** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (SEW-D model)
* **shieldgemma2** — [ShieldGemma2Processor](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2Processor) (Shieldgemma2 model)
* **siglip** — [SiglipProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipProcessor) (SigLIP model)
* **siglip2** — [Siglip2Processor](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Processor) (SigLIP2 model)
* **smolvlm** — [SmolVLMProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMProcessor) (SmolVLM model)
* **speech\_to\_text** — [Speech2TextProcessor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextProcessor) (Speech2Text model)
* **speech\_to\_text\_2** — [Speech2Text2Processor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor) (Speech2Text2 model)
* **speecht5** — [SpeechT5Processor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor) (SpeechT5 model)
* **trocr** — [TrOCRProcessor](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRProcessor) (TrOCR model)
* **tvlt** — [TvltProcessor](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltProcessor) (TVLT model)
* **tvp** — [TvpProcessor](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpProcessor) (TVP model)
* **udop** — [UdopProcessor](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopProcessor) (UDOP model)
* **unispeech** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (UniSpeech model)
* **unispeech-sat** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (UniSpeechSat model)
* **video\_llava** — [VideoLlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/video_llava#transformers.VideoLlavaProcessor) (VideoLlava model)
* **vilt** — [ViltProcessor](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor) (ViLT model)
* **vipllava** — [LlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaProcessor) (VipLlava model)
* **vision-text-dual-encoder** — [VisionTextDualEncoderProcessor](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderProcessor) (VisionTextDualEncoder model)
* **voxtral** — [VoxtralProcessor](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralProcessor) (Voxtral model)
* **wav2vec2** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (Wav2Vec2 model)
* **wav2vec2-bert** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (Wav2Vec2-BERT model)
* **wav2vec2-conformer** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (Wav2Vec2-Conformer model)
* **wavlm** — [Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) (WavLM model)
* **whisper** — [WhisperProcessor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperProcessor) (Whisper model)
* **xclip** — [XCLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/xclip#transformers.XCLIPProcessor) (X-CLIP model)

Passing `token=True` is required when you want to use a private model.

Examples:


```
>>> from transformers import AutoProcessor

>>> # Download processor from huggingface.co and cache.
>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

>>> # If processor files are in a directory (e.g. processor was saved using *save_pretrained('./test/saved_model/')*)
>>> # processor = AutoProcessor.from_pretrained("./test/saved_model/")
```

#### register

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/processing_auto.py#L425)

( config\_class processor\_class exist\_ok = False  )

Parameters

* **config\_class** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The configuration corresponding to the model to register.
* **processor\_class** ([ProcessorMixin](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin)) — The processor to register.

Register a new processor for this class.

## Generic model classes

The following auto classes are available for instantiating a base model class without a specific head.

### AutoModel

### class transformers.AutoModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1898)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the base model classes of the library when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [ASTConfig](/docs/transformers/v4.56.2/en/model_doc/audio-spectrogram-transformer#transformers.ASTConfig) configuration class: [ASTModel](/docs/transformers/v4.56.2/en/model_doc/audio-spectrogram-transformer#transformers.ASTModel) (Audio Spectrogram Transformer model)
  + [Aimv2Config](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Config) configuration class: [Aimv2Model](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Model) (AIMv2 model)
  + [Aimv2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionConfig) configuration class: [Aimv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionModel) (Aimv2VisionModel model)
  + [AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig) configuration class: `AlbertModel` (ALBERT model)
  + [AlignConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignConfig) configuration class: [AlignModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignModel) (ALIGN model)
  + [AltCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPConfig) configuration class: [AltCLIPModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPModel) (AltCLIP model)
  + [ApertusConfig](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusConfig) configuration class: [ApertusModel](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusModel) (Apertus model)
  + [ArceeConfig](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeConfig) configuration class: [ArceeModel](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeModel) (Arcee model)
  + [AriaConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig) configuration class: [AriaModel](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaModel) (Aria model)
  + [AriaTextConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextConfig) configuration class: [AriaTextModel](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextModel) (AriaText model)
  + [AutoformerConfig](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig) configuration class: [AutoformerModel](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerModel) (Autoformer model)
  + [AyaVisionConfig](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig) configuration class: [AyaVisionModel](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionModel) (AyaVision model)
  + [BambaConfig](/docs/transformers/v4.56.2/en/model_doc/bamba#transformers.BambaConfig) configuration class: [BambaModel](/docs/transformers/v4.56.2/en/model_doc/bamba#transformers.BambaModel) (Bamba model)
  + [BarkConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkConfig) configuration class: [BarkModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkModel) (Bark model)
  + [BartConfig](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartConfig) configuration class: [BartModel](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartModel) (BART model)
  + [BeitConfig](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitConfig) configuration class: [BeitModel](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitModel) (BEiT model)
  + [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) configuration class: [BertModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel) (BERT model)
  + [BertGenerationConfig](/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationConfig) configuration class: [BertGenerationEncoder](/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationEncoder) (Bert Generation model)
  + [BigBirdConfig](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig) configuration class: [BigBirdModel](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdModel) (BigBird model)
  + [BigBirdPegasusConfig](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig) configuration class: [BigBirdPegasusModel](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusModel) (BigBird-Pegasus model)
  + [BioGptConfig](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig) configuration class: [BioGptModel](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptModel) (BioGpt model)
  + [BitConfig](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitConfig) configuration class: [BitModel](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitModel) (BiT model)
  + [BitNetConfig](/docs/transformers/v4.56.2/en/model_doc/bitnet#transformers.BitNetConfig) configuration class: [BitNetModel](/docs/transformers/v4.56.2/en/model_doc/bitnet#transformers.BitNetModel) (BitNet model)
  + [BlenderbotConfig](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotConfig) configuration class: [BlenderbotModel](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotModel) (Blenderbot model)
  + [BlenderbotSmallConfig](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig) configuration class: [BlenderbotSmallModel](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel) (BlenderbotSmall model)
  + [Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config) configuration class: [Blip2Model](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Model) (BLIP-2 model)
  + [Blip2QFormerConfig](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerConfig) configuration class: [Blip2QFormerModel](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerModel) (BLIP-2 QFormer model)
  + [BlipConfig](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipConfig) configuration class: [BlipModel](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipModel) (BLIP model)
  + [BloomConfig](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig) configuration class: [BloomModel](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomModel) (BLOOM model)
  + [BridgeTowerConfig](/docs/transformers/v4.56.2/en/model_doc/bridgetower#transformers.BridgeTowerConfig) configuration class: [BridgeTowerModel](/docs/transformers/v4.56.2/en/model_doc/bridgetower#transformers.BridgeTowerModel) (BridgeTower model)
  + [BrosConfig](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosConfig) configuration class: [BrosModel](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosModel) (BROS model)
  + [CLIPConfig](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPConfig) configuration class: [CLIPModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPModel) (CLIP model)
  + [CLIPSegConfig](/docs/transformers/v4.56.2/en/model_doc/clipseg#transformers.CLIPSegConfig) configuration class: [CLIPSegModel](/docs/transformers/v4.56.2/en/model_doc/clipseg#transformers.CLIPSegModel) (CLIPSeg model)
  + [CLIPTextConfig](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTextConfig) configuration class: [CLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTextModel) (CLIPTextModel model)
  + [CLIPVisionConfig](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPVisionConfig) configuration class: [CLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPVisionModel) (CLIPVisionModel model)
  + [CTRLConfig](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig) configuration class: [CTRLModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel) (CTRL model)
  + [CamembertConfig](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig) configuration class: [CamembertModel](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertModel) (CamemBERT model)
  + [CanineConfig](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineConfig) configuration class: [CanineModel](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineModel) (CANINE model)
  + [ChameleonConfig](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonConfig) configuration class: [ChameleonModel](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonModel) (Chameleon model)
  + [ChineseCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig) configuration class: [ChineseCLIPModel](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPModel) (Chinese-CLIP model)
  + [ChineseCLIPVisionConfig](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPVisionConfig) configuration class: [ChineseCLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPVisionModel) (ChineseCLIPVisionModel model)
  + [ClapConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapConfig) configuration class: [ClapModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapModel) (CLAP model)
  + [ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig) configuration class: [ClvpModelForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpModelForConditionalGeneration) (CLVP model)
  + [CodeGenConfig](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenConfig) configuration class: [CodeGenModel](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenModel) (CodeGen model)
  + [Cohere2Config](/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2Config) configuration class: [Cohere2Model](/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2Model) (Cohere2 model)
  + [Cohere2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/cohere2_vision#transformers.Cohere2VisionConfig) configuration class: [Cohere2VisionModel](/docs/transformers/v4.56.2/en/model_doc/cohere2_vision#transformers.Cohere2VisionModel) (Cohere2Vision model)
  + [CohereConfig](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereConfig) configuration class: [CohereModel](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereModel) (Cohere model)
  + [ConditionalDetrConfig](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig) configuration class: [ConditionalDetrModel](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrModel) (Conditional DETR model)
  + [ConvBertConfig](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertConfig) configuration class: [ConvBertModel](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertModel) (ConvBERT model)
  + [ConvNextConfig](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextConfig) configuration class: [ConvNextModel](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextModel) (ConvNeXT model)
  + [ConvNextV2Config](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Config) configuration class: [ConvNextV2Model](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Model) (ConvNeXTV2 model)
  + [CpmAntConfig](/docs/transformers/v4.56.2/en/model_doc/cpmant#transformers.CpmAntConfig) configuration class: [CpmAntModel](/docs/transformers/v4.56.2/en/model_doc/cpmant#transformers.CpmAntModel) (CPM-Ant model)
  + [CsmConfig](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig) configuration class: [CsmForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmForConditionalGeneration) (CSM model)
  + [CvtConfig](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtConfig) configuration class: [CvtModel](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtModel) (CvT model)
  + [DFineConfig](/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineConfig) configuration class: [DFineModel](/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineModel) (D-FINE model)
  + [DINOv3ConvNextConfig](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ConvNextConfig) configuration class: [DINOv3ConvNextModel](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ConvNextModel) (DINOv3 ConvNext model)
  + [DINOv3ViTConfig](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ViTConfig) configuration class: [DINOv3ViTModel](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ViTModel) (DINOv3 ViT model)
  + [DPRConfig](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRConfig) configuration class: [DPRQuestionEncoder](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoder) (DPR model)
  + [DPTConfig](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTConfig) configuration class: [DPTModel](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTModel) (DPT model)
  + [DabDetrConfig](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrConfig) configuration class: [DabDetrModel](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrModel) (DAB-DETR model)
  + [DacConfig](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacConfig) configuration class: [DacModel](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel) (DAC model)
  + [Data2VecAudioConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig) configuration class: [Data2VecAudioModel](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioModel) (Data2VecAudio model)
  + [Data2VecTextConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig) configuration class: [Data2VecTextModel](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextModel) (Data2VecText model)
  + [Data2VecVisionConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig) configuration class: [Data2VecVisionModel](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionModel) (Data2VecVision model)
  + [DbrxConfig](/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxConfig) configuration class: [DbrxModel](/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxModel) (DBRX model)
  + [DebertaConfig](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaConfig) configuration class: [DebertaModel](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaModel) (DeBERTa model)
  + [DebertaV2Config](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config) configuration class: [DebertaV2Model](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Model) (DeBERTa-v2 model)
  + [DecisionTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/decision_transformer#transformers.DecisionTransformerConfig) configuration class: [DecisionTransformerModel](/docs/transformers/v4.56.2/en/model_doc/decision_transformer#transformers.DecisionTransformerModel) (Decision Transformer model)
  + [DeepseekV2Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config) configuration class: [DeepseekV2Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Model) (DeepSeek-V2 model)
  + [DeepseekV3Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Config) configuration class: [DeepseekV3Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Model) (DeepSeek-V3 model)
  + [DeepseekVLConfig](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLConfig) configuration class: [DeepseekVLModel](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLModel) (DeepseekVL model)
  + [DeepseekVLHybridConfig](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridConfig) configuration class: [DeepseekVLHybridModel](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridModel) (DeepseekVLHybrid model)
  + [DeformableDetrConfig](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrConfig) configuration class: [DeformableDetrModel](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrModel) (Deformable DETR model)
  + [DeiTConfig](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTConfig) configuration class: [DeiTModel](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTModel) (DeiT model)
  + [DepthProConfig](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProConfig) configuration class: [DepthProModel](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProModel) (DepthPro model)
  + [DetaConfig](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaConfig) configuration class: [DetaModel](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaModel) (DETA model)
  + [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig) configuration class: [DetrModel](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrModel) (DETR model)
  + [DiaConfig](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaConfig) configuration class: [DiaModel](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel) (Dia model)
  + [DiffLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaConfig) configuration class: [DiffLlamaModel](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaModel) (DiffLlama model)
  + [DinatConfig](/docs/transformers/v4.56.2/en/model_doc/dinat#transformers.DinatConfig) configuration class: [DinatModel](/docs/transformers/v4.56.2/en/model_doc/dinat#transformers.DinatModel) (DiNAT model)
  + [Dinov2Config](/docs/transformers/v4.56.2/en/model_doc/dinov2#transformers.Dinov2Config) configuration class: [Dinov2Model](/docs/transformers/v4.56.2/en/model_doc/dinov2#transformers.Dinov2Model) (DINOv2 model)
  + [Dinov2WithRegistersConfig](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersConfig) configuration class: [Dinov2WithRegistersModel](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersModel) (DINOv2 with Registers model)
  + [DistilBertConfig](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertConfig) configuration class: [DistilBertModel](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertModel) (DistilBERT model)
  + [DogeConfig](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeConfig) configuration class: [DogeModel](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeModel) (Doge model)
  + [DonutSwinConfig](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinConfig) configuration class: [DonutSwinModel](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinModel) (DonutSwin model)
  + [Dots1Config](/docs/transformers/v4.56.2/en/model_doc/dots1#transformers.Dots1Config) configuration class: [Dots1Model](/docs/transformers/v4.56.2/en/model_doc/dots1#transformers.Dots1Model) (dots1 model)
  + [EfficientFormerConfig](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerConfig) configuration class: [EfficientFormerModel](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerModel) (EfficientFormer model)
  + [EfficientLoFTRConfig](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRConfig) configuration class: [EfficientLoFTRModel](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRModel) (EfficientLoFTR model)
  + [EfficientNetConfig](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetConfig) configuration class: [EfficientNetModel](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetModel) (EfficientNet model)
  + [ElectraConfig](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig) configuration class: [ElectraModel](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraModel) (ELECTRA model)
  + [Emu3Config](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Config) configuration class: [Emu3Model](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Model) (Emu3 model)
  + [EncodecConfig](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecConfig) configuration class: [EncodecModel](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel) (EnCodec model)
  + [Ernie4\_5Config](/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Config) configuration class: [Ernie4\_5Model](/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Model) (Ernie4\_5 model)
  + [Ernie4\_5\_MoeConfig](/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeConfig) configuration class: [Ernie4\_5\_MoeModel](/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeModel) (Ernie4\_5\_MoE model)
  + [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieModel](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieModel) (ERNIE model)
  + [ErnieMConfig](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMConfig) configuration class: [ErnieMModel](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMModel) (ErnieM model)
  + [EsmConfig](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig) configuration class: [EsmModel](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmModel) (ESM model)
  + [EvollaConfig](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaConfig) configuration class: [EvollaModel](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaModel) (Evolla model)
  + [Exaone4Config](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config) configuration class: [Exaone4Model](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Model) (EXAONE-4.0 model)
  + [FNetConfig](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig) configuration class: [FNetModel](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetModel) (FNet model)
  + [FSMTConfig](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTConfig) configuration class: [FSMTModel](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTModel) (FairSeq Machine-Translation model)
  + [FalconConfig](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig) configuration class: [FalconModel](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconModel) (Falcon model)
  + [FalconH1Config](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1Config) configuration class: [FalconH1Model](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1Model) (FalconH1 model)
  + [FalconMambaConfig](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaConfig) configuration class: [FalconMambaModel](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaModel) (FalconMamba model)
  + [FastSpeech2ConformerConfig](/docs/transformers/v4.56.2/en/model_doc/fastspeech2_conformer#transformers.FastSpeech2ConformerConfig) configuration class: [FastSpeech2ConformerModel](/docs/transformers/v4.56.2/en/model_doc/fastspeech2_conformer#transformers.FastSpeech2ConformerModel) (FastSpeech2Conformer model)
  + [FastSpeech2ConformerWithHifiGanConfig](/docs/transformers/v4.56.2/en/model_doc/fastspeech2_conformer#transformers.FastSpeech2ConformerWithHifiGanConfig) configuration class: [FastSpeech2ConformerWithHifiGan](/docs/transformers/v4.56.2/en/model_doc/fastspeech2_conformer#transformers.FastSpeech2ConformerWithHifiGan) (FastSpeech2ConformerWithHifiGan model)
  + [FlaubertConfig](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertConfig) configuration class: [FlaubertModel](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertModel) (FlauBERT model)
  + [FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig) configuration class: [FlavaModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaModel) (FLAVA model)
  + [Florence2Config](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Config) configuration class: [Florence2Model](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Model) (Florence2 model)
  + [FocalNetConfig](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetConfig) configuration class: [FocalNetModel](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetModel) (FocalNet model)
  + [FunnelConfig](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelConfig) configuration class: [FunnelModel](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelModel) or [FunnelBaseModel](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelBaseModel) (Funnel Transformer model)
  + [FuyuConfig](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuConfig) configuration class: [FuyuModel](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuModel) (Fuyu model)
  + [GLPNConfig](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNConfig) configuration class: [GLPNModel](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNModel) (GLPN model)
  + [GPT2Config](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config) configuration class: [GPT2Model](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Model) (OpenAI GPT-2 model)
  + [GPTBigCodeConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeConfig) configuration class: [GPTBigCodeModel](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeModel) (GPTBigCode model)
  + [GPTJConfig](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJConfig) configuration class: [GPTJModel](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJModel) (GPT-J model)
  + [GPTNeoConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig) configuration class: [GPTNeoModel](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoModel) (GPT Neo model)
  + [GPTNeoXConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXConfig) configuration class: [GPTNeoXModel](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXModel) (GPT NeoX model)
  + [GPTNeoXJapaneseConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseConfig) configuration class: [GPTNeoXJapaneseModel](/docs/transformers/v4.56.2/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseModel) (GPT NeoX Japanese model)
  + [GPTSanJapaneseConfig](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseConfig) configuration class: [GPTSanJapaneseForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseForConditionalGeneration) (GPTSAN-japanese model)
  + [Gemma2Config](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Config) configuration class: [Gemma2Model](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Model) (Gemma2 model)
  + [Gemma3Config](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config) configuration class: [Gemma3Model](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Model) (Gemma3ForConditionalGeneration model)
  + [Gemma3TextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextConfig) configuration class: [Gemma3TextModel](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextModel) (Gemma3ForCausalLM model)
  + [Gemma3nAudioConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nAudioConfig) configuration class: `Gemma3nAudioEncoder` (Gemma3nAudioEncoder model)
  + [Gemma3nConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nConfig) configuration class: [Gemma3nModel](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nModel) (Gemma3nForConditionalGeneration model)
  + [Gemma3nTextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextConfig) configuration class: [Gemma3nTextModel](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextModel) (Gemma3nForCausalLM model)
  + [Gemma3nVisionConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nVisionConfig) configuration class: [TimmWrapperModel](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperModel) (TimmWrapperModel model)
  + [GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaConfig) configuration class: [GemmaModel](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaModel) (Gemma model)
  + [GitConfig](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitConfig) configuration class: [GitModel](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitModel) (GIT model)
  + [Glm4Config](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Config) configuration class: [Glm4Model](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Model) (GLM4 model)
  + [Glm4MoeConfig](/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeConfig) configuration class: [Glm4MoeModel](/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeModel) (Glm4MoE model)
  + [Glm4vConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vConfig) configuration class: [Glm4vModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vModel) (GLM4V model)
  + [Glm4vMoeConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeConfig) configuration class: [Glm4vMoeModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeModel) (GLM4VMOE model)
  + [Glm4vMoeTextConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeTextConfig) configuration class: [Glm4vMoeTextModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeTextModel) (GLM4VMOE model)
  + [Glm4vTextConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vTextConfig) configuration class: [Glm4vTextModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vTextModel) (GLM4V model)
  + [GlmConfig](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmConfig) configuration class: [GlmModel](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmModel) (GLM model)
  + [GotOcr2Config](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Config) configuration class: [GotOcr2Model](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Model) (GOT-OCR2 model)
  + [GptOssConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssConfig) configuration class: [GptOssModel](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssModel) (GptOss model)
  + [GraniteConfig](/docs/transformers/v4.56.2/en/model_doc/granite#transformers.GraniteConfig) configuration class: [GraniteModel](/docs/transformers/v4.56.2/en/model_doc/granite#transformers.GraniteModel) (Granite model)
  + [GraniteMoeConfig](/docs/transformers/v4.56.2/en/model_doc/granitemoe#transformers.GraniteMoeConfig) configuration class: [GraniteMoeModel](/docs/transformers/v4.56.2/en/model_doc/granitemoe#transformers.GraniteMoeModel) (GraniteMoeMoe model)
  + [GraniteMoeHybridConfig](/docs/transformers/v4.56.2/en/model_doc/granitemoehybrid#transformers.GraniteMoeHybridConfig) configuration class: [GraniteMoeHybridModel](/docs/transformers/v4.56.2/en/model_doc/granitemoehybrid#transformers.GraniteMoeHybridModel) (GraniteMoeHybrid model)
  + [GraniteMoeSharedConfig](/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedConfig) configuration class: [GraniteMoeSharedModel](/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedModel) (GraniteMoeSharedMoe model)
  + [GraphormerConfig](/docs/transformers/v4.56.2/en/model_doc/graphormer#transformers.GraphormerConfig) configuration class: [GraphormerModel](/docs/transformers/v4.56.2/en/model_doc/graphormer#transformers.GraphormerModel) (Graphormer model)
  + [GroundingDinoConfig](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoConfig) configuration class: [GroundingDinoModel](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoModel) (Grounding DINO model)
  + [GroupViTConfig](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTConfig) configuration class: [GroupViTModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTModel) (GroupViT model)
  + [HGNetV2Config](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Config) configuration class: [HGNetV2Backbone](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Backbone) (HGNet-V2 model)
  + [HeliumConfig](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumConfig) configuration class: [HeliumModel](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumModel) (Helium model)
  + [HieraConfig](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraConfig) configuration class: [HieraModel](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraModel) (Hiera model)
  + [HubertConfig](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertConfig) configuration class: [HubertModel](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertModel) (Hubert model)
  + [HunYuanDenseV1Config](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Config) configuration class: [HunYuanDenseV1Model](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Model) (HunYuanDenseV1 model)
  + [HunYuanMoEV1Config](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Config) configuration class: [HunYuanMoEV1Model](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Model) (HunYuanMoeV1 model)
  + [IBertConfig](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertConfig) configuration class: [IBertModel](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertModel) (I-BERT model)
  + [IJepaConfig](/docs/transformers/v4.56.2/en/model_doc/ijepa#transformers.IJepaConfig) configuration class: [IJepaModel](/docs/transformers/v4.56.2/en/model_doc/ijepa#transformers.IJepaModel) (I-JEPA model)
  + [Idefics2Config](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Config) configuration class: [Idefics2Model](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Model) (Idefics2 model)
  + [Idefics3Config](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3Config) configuration class: [Idefics3Model](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3Model) (Idefics3 model)
  + [Idefics3VisionConfig](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3VisionConfig) configuration class: [Idefics3VisionTransformer](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3VisionTransformer) (Idefics3VisionTransformer model)
  + [IdeficsConfig](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsConfig) configuration class: [IdeficsModel](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsModel) (IDEFICS model)
  + [ImageGPTConfig](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig) configuration class: [ImageGPTModel](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTModel) (ImageGPT model)
  + [InformerConfig](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig) configuration class: [InformerModel](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerModel) (Informer model)
  + [InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig) configuration class: [InstructBlipModel](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipModel) (InstructBLIP model)
  + [InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig) configuration class: [InstructBlipVideoModel](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoModel) (InstructBlipVideo model)
  + [InternVLConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLConfig) configuration class: [InternVLModel](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLModel) (InternVL model)
  + [InternVLVisionConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLVisionConfig) configuration class: [InternVLVisionModel](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLVisionModel) (InternVLVision model)
  + [JambaConfig](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaConfig) configuration class: [JambaModel](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaModel) (Jamba model)
  + [JanusConfig](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusConfig) configuration class: [JanusModel](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusModel) (Janus model)
  + [JetMoeConfig](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeConfig) configuration class: [JetMoeModel](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeModel) (JetMoe model)
  + [JukeboxConfig](/docs/transformers/v4.56.2/en/model_doc/jukebox#transformers.JukeboxConfig) configuration class: [JukeboxModel](/docs/transformers/v4.56.2/en/model_doc/jukebox#transformers.JukeboxModel) (Jukebox model)
  + [Kosmos2Config](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Config) configuration class: [Kosmos2Model](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Model) (KOSMOS-2 model)
  + [Kosmos2\_5Config](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Config) configuration class: [Kosmos2\_5Model](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Model) (KOSMOS-2.5 model)
  + [KyutaiSpeechToTextConfig](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextConfig) configuration class: [KyutaiSpeechToTextModel](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextModel) (KyutaiSpeechToText model)
  + [LEDConfig](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDConfig) configuration class: [LEDModel](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDModel) (LED model)
  + [LayoutLMConfig](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig) configuration class: [LayoutLMModel](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMModel) (LayoutLM model)
  + [LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config) configuration class: [LayoutLMv2Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Model) (LayoutLMv2 model)
  + [LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config) configuration class: [LayoutLMv3Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model) (LayoutLMv3 model)
  + [LevitConfig](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitConfig) configuration class: [LevitModel](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitModel) (LeViT model)
  + [Lfm2Config](/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Config) configuration class: [Lfm2Model](/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Model) (Lfm2 model)
  + [LightGlueConfig](/docs/transformers/v4.56.2/en/model_doc/lightglue#transformers.LightGlueConfig) configuration class: [LightGlueForKeypointMatching](/docs/transformers/v4.56.2/en/model_doc/lightglue#transformers.LightGlueForKeypointMatching) (LightGlue model)
  + [LiltConfig](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltConfig) configuration class: [LiltModel](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltModel) (LiLT model)
  + [Llama4Config](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Config) configuration class: [Llama4ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForConditionalGeneration) (Llama4 model)
  + [Llama4TextConfig](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4TextConfig) configuration class: [Llama4TextModel](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4TextModel) (Llama4ForCausalLM model)
  + [LlamaConfig](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig) configuration class: [LlamaModel](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel) (LLaMA model)
  + [LlavaConfig](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaConfig) configuration class: [LlavaModel](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaModel) (LLaVa model)
  + [LlavaNextConfig](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextConfig) configuration class: [LlavaNextModel](/docs/transformers/v4.56.2/en/model_doc/llava_next#transformers.LlavaNextModel) (LLaVA-NeXT model)
  + [LlavaNextVideoConfig](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoConfig) configuration class: [LlavaNextVideoModel](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoModel) (LLaVa-NeXT-Video model)
  + [LlavaOnevisionConfig](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionConfig) configuration class: [LlavaOnevisionModel](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionModel) (LLaVA-Onevision model)
  + [LongT5Config](/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Config) configuration class: [LongT5Model](/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Model) (LongT5 model)
  + [LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig) configuration class: [LongformerModel](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerModel) (Longformer model)
  + [LukeConfig](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig) configuration class: [LukeModel](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeModel) (LUKE model)
  + [LxmertConfig](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertConfig) configuration class: [LxmertModel](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertModel) (LXMERT model)
  + [M2M100Config](/docs/transformers/v4.56.2/en/model_doc/m2m_100#transformers.M2M100Config) configuration class: [M2M100Model](/docs/transformers/v4.56.2/en/model_doc/m2m_100#transformers.M2M100Model) (M2M100 model)
  + [MBartConfig](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig) configuration class: [MBartModel](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartModel) (mBART model)
  + [MCTCTConfig](/docs/transformers/v4.56.2/en/model_doc/mctct#transformers.MCTCTConfig) configuration class: [MCTCTModel](/docs/transformers/v4.56.2/en/model_doc/mctct#transformers.MCTCTModel) (M-CTC-T model)
  + [MLCDVisionConfig](/docs/transformers/v4.56.2/en/model_doc/mlcd#transformers.MLCDVisionConfig) configuration class: [MLCDVisionModel](/docs/transformers/v4.56.2/en/model_doc/mlcd#transformers.MLCDVisionModel) (MLCD model)
  + [MMGroundingDinoConfig](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoConfig) configuration class: [MMGroundingDinoModel](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoModel) (MM Grounding DINO model)
  + [MPNetConfig](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig) configuration class: [MPNetModel](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetModel) (MPNet model)
  + [MT5Config](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config) configuration class: [MT5Model](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Model) (MT5 model)
  + [Mamba2Config](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2Config) configuration class: [Mamba2Model](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2Model) (mamba2 model)
  + [MambaConfig](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaConfig) configuration class: [MambaModel](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaModel) (Mamba model)
  + [MarianConfig](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianConfig) configuration class: [MarianModel](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianModel) (Marian model)
  + [MarkupLMConfig](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMConfig) configuration class: [MarkupLMModel](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMModel) (MarkupLM model)
  + [Mask2FormerConfig](/docs/transformers/v4.56.2/en/model_doc/mask2former#transformers.Mask2FormerConfig) configuration class: [Mask2FormerModel](/docs/transformers/v4.56.2/en/model_doc/mask2former#transformers.Mask2FormerModel) (Mask2Former model)
  + [MaskFormerConfig](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig) configuration class: [MaskFormerModel](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerModel) (MaskFormer model)
  + `MaskFormerSwinConfig` configuration class: `MaskFormerSwinModel` (MaskFormerSwin model)
  + [MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig) configuration class: [MegaModel](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel) (MEGA model)
  + [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertModel](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertModel) (Megatron-BERT model)
  + [MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config) configuration class: [MetaClip2Model](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Model) (MetaCLIP 2 model)
  + [MgpstrConfig](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrConfig) configuration class: [MgpstrForSceneTextRecognition](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrForSceneTextRecognition) (MGP-STR model)
  + [MimiConfig](/docs/transformers/v4.56.2/en/model_doc/mimi#transformers.MimiConfig) configuration class: [MimiModel](/docs/transformers/v4.56.2/en/model_doc/mimi#transformers.MimiModel) (Mimi model)
  + [MiniMaxConfig](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxConfig) configuration class: [MiniMaxModel](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxModel) (MiniMax model)
  + [Mistral3Config](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Config) configuration class: [Mistral3Model](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Model) (Mistral3 model)
  + [MistralConfig](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig) configuration class: [MistralModel](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralModel) (Mistral model)
  + [MixtralConfig](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralConfig) configuration class: [MixtralModel](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralModel) (Mixtral model)
  + [MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig) configuration class: [MllamaModel](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaModel) (Mllama model)
  + [MobileBertConfig](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [MobileBertModel](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertModel) (MobileBERT model)
  + [MobileNetV1Config](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1Config) configuration class: [MobileNetV1Model](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1Model) (MobileNetV1 model)
  + [MobileNetV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2Config) configuration class: [MobileNetV2Model](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2Model) (MobileNetV2 model)
  + [MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig) configuration class: [MobileViTModel](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTModel) (MobileViT model)
  + [MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config) configuration class: [MobileViTV2Model](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Model) (MobileViTV2 model)
  + [ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig) configuration class: [ModernBertModel](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertModel) (ModernBERT model)
  + [ModernBertDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderConfig) configuration class: [ModernBertDecoderModel](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderModel) (ModernBertDecoder model)
  + [MoonshineConfig](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig) configuration class: [MoonshineModel](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineModel) (Moonshine model)
  + [MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig) configuration class: [MoshiModel](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiModel) (Moshi model)
  + [MptConfig](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptConfig) configuration class: [MptModel](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptModel) (MPT model)
  + [MraConfig](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraConfig) configuration class: [MraModel](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraModel) (MRA model)
  + [MusicgenConfig](/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenConfig) configuration class: [MusicgenModel](/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenModel) (MusicGen model)
  + [MusicgenMelodyConfig](/docs/transformers/v4.56.2/en/model_doc/musicgen_melody#transformers.MusicgenMelodyConfig) configuration class: [MusicgenMelodyModel](/docs/transformers/v4.56.2/en/model_doc/musicgen_melody#transformers.MusicgenMelodyModel) (MusicGen Melody model)
  + [MvpConfig](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig) configuration class: [MvpModel](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpModel) (MVP model)
  + [NatConfig](/docs/transformers/v4.56.2/en/model_doc/nat#transformers.NatConfig) configuration class: [NatModel](/docs/transformers/v4.56.2/en/model_doc/nat#transformers.NatModel) (NAT model)
  + [NemotronConfig](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronConfig) configuration class: [NemotronModel](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronModel) (Nemotron model)
  + [NezhaConfig](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig) configuration class: [NezhaModel](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaModel) (Nezha model)
  + [NllbMoeConfig](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeConfig) configuration class: [NllbMoeModel](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeModel) (NLLB-MOE model)
  + [NystromformerConfig](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig) configuration class: [NystromformerModel](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerModel) (Nyströmformer model)
  + [OPTConfig](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig) configuration class: [OPTModel](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTModel) (OPT model)
  + [Olmo2Config](/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Config) configuration class: [Olmo2Model](/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Model) (OLMo2 model)
  + [OlmoConfig](/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoConfig) configuration class: [OlmoModel](/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoModel) (OLMo model)
  + [OlmoeConfig](/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeConfig) configuration class: [OlmoeModel](/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeModel) (OLMoE model)
  + [OmDetTurboConfig](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboConfig) configuration class: [OmDetTurboForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboForObjectDetection) (OmDet-Turbo model)
  + [OneFormerConfig](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerConfig) configuration class: [OneFormerModel](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerModel) (OneFormer model)
  + [OpenAIGPTConfig](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig) configuration class: [OpenAIGPTModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTModel) (OpenAI GPT model)
  + [OpenLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaConfig) configuration class: [OpenLlamaModel](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaModel) (OpenLlama model)
  + [Ovis2Config](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Config) configuration class: [Ovis2Model](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Model) (Ovis2 model)
  + [OwlViTConfig](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTConfig) configuration class: [OwlViTModel](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTModel) (OWL-ViT model)
  + [Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config) configuration class: [Owlv2Model](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Model) (OWLv2 model)
  + [PLBartConfig](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig) configuration class: [PLBartModel](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartModel) (PLBart model)
  + [PaliGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaConfig) configuration class: [PaliGemmaModel](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaModel) (PaliGemma model)
  + [PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig) configuration class: [PatchTSMixerModel](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerModel) (PatchTSMixer model)
  + [PatchTSTConfig](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTConfig) configuration class: [PatchTSTModel](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTModel) (PatchTST model)
  + [PegasusConfig](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusConfig) configuration class: [PegasusModel](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusModel) (Pegasus model)
  + [PegasusXConfig](/docs/transformers/v4.56.2/en/model_doc/pegasus_x#transformers.PegasusXConfig) configuration class: [PegasusXModel](/docs/transformers/v4.56.2/en/model_doc/pegasus_x#transformers.PegasusXModel) (PEGASUS-X model)
  + [PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig) configuration class: [PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel) (Perceiver model)
  + [PerceptionLMConfig](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMConfig) configuration class: [PerceptionLMModel](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMModel) (PerceptionLM model)
  + [PersimmonConfig](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonConfig) configuration class: [PersimmonModel](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonModel) (Persimmon model)
  + [Phi3Config](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Config) configuration class: [Phi3Model](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Model) (Phi3 model)
  + [Phi4MultimodalConfig](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalConfig) configuration class: [Phi4MultimodalModel](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalModel) (Phi4Multimodal model)
  + [PhiConfig](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiConfig) configuration class: [PhiModel](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiModel) (Phi model)
  + [PhimoeConfig](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeConfig) configuration class: [PhimoeModel](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeModel) (Phimoe model)
  + [PixtralVisionConfig](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionConfig) configuration class: [PixtralVisionModel](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionModel) (Pixtral model)
  + [PoolFormerConfig](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerConfig) configuration class: [PoolFormerModel](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerModel) (PoolFormer model)
  + [ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig) configuration class: [ProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetModel) (ProphetNet model)
  + [PvtConfig](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtConfig) configuration class: [PvtModel](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtModel) (PVT model)
  + [PvtV2Config](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Config) configuration class: [PvtV2Model](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Model) (PVTv2 model)
  + [QDQBertConfig](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig) configuration class: [QDQBertModel](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertModel) (QDQBert model)
  + [Qwen2AudioEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioEncoderConfig) configuration class: [Qwen2AudioEncoder](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioEncoder) (Qwen2AudioEncoder model)
  + [Qwen2Config](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Config) configuration class: [Qwen2Model](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Model) (Qwen2 model)
  + [Qwen2MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeConfig) configuration class: [Qwen2MoeModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeModel) (Qwen2MoE model)
  + [Qwen2VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLConfig) configuration class: [Qwen2VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLModel) (Qwen2VL model)
  + [Qwen2VLTextConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLTextConfig) configuration class: [Qwen2VLTextModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLTextModel) (Qwen2VL model)
  + [Qwen2\_5\_VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLConfig) configuration class: [Qwen2\_5\_VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLModel) (Qwen2\_5\_VL model)
  + [Qwen2\_5\_VLTextConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLTextConfig) configuration class: [Qwen2\_5\_VLTextModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLTextModel) (Qwen2\_5\_VL model)
  + [Qwen3Config](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Config) configuration class: [Qwen3Model](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Model) (Qwen3 model)
  + [Qwen3MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeConfig) configuration class: [Qwen3MoeModel](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeModel) (Qwen3MoE model)
  + [RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig) configuration class: [RTDetrModel](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrModel) (RT-DETR model)
  + [RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config) configuration class: [RTDetrV2Model](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Model) (RT-DETRv2 model)
  + [RecurrentGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaConfig) configuration class: [RecurrentGemmaModel](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaModel) (RecurrentGemma model)
  + [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig) configuration class: [ReformerModel](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModel) (Reformer model)
  + [RegNetConfig](/docs/transformers/v4.56.2/en/model_doc/regnet#transformers.RegNetConfig) configuration class: [RegNetModel](/docs/transformers/v4.56.2/en/model_doc/regnet#transformers.RegNetModel) (RegNet model)
  + [RemBertConfig](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertConfig) configuration class: [RemBertModel](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertModel) (RemBERT model)
  + [ResNetConfig](/docs/transformers/v4.56.2/en/model_doc/resnet#transformers.ResNetConfig) configuration class: [ResNetModel](/docs/transformers/v4.56.2/en/model_doc/resnet#transformers.ResNetModel) (ResNet model)
  + [RetriBertConfig](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertConfig) configuration class: [RetriBertModel](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertModel) (RetriBERT model)
  + [RoCBertConfig](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig) configuration class: [RoCBertModel](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertModel) (RoCBert model)
  + [RoFormerConfig](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig) configuration class: [RoFormerModel](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerModel) (RoFormer model)
  + [RobertaConfig](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig) configuration class: [RobertaModel](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaModel) (RoBERTa model)
  + [RobertaPreLayerNormConfig](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig) configuration class: [RobertaPreLayerNormModel](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormModel) (RoBERTa-PreLayerNorm model)
  + [RwkvConfig](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvConfig) configuration class: [RwkvModel](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvModel) (RWKV model)
  + [SEWConfig](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWConfig) configuration class: [SEWModel](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWModel) (SEW model)
  + [SEWDConfig](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDConfig) configuration class: [SEWDModel](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDModel) (SEW-D model)
  + [Sam2Config](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Config) configuration class: [Sam2Model](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Model) (SAM2 model)
  + [Sam2HieraDetConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2HieraDetConfig) configuration class: [Sam2HieraDetModel](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2HieraDetModel) (Sam2HieraDetModel model)
  + [Sam2VideoConfig](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoConfig) configuration class: [Sam2VideoModel](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoModel) (Sam2VideoModel model)
  + [Sam2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2VisionConfig) configuration class: [Sam2VisionModel](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2VisionModel) (Sam2VisionModel model)
  + [SamConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamConfig) configuration class: [SamModel](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamModel) (SAM model)
  + [SamHQConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQConfig) configuration class: [SamHQModel](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQModel) (SAM-HQ model)
  + [SamHQVisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQVisionConfig) configuration class: [SamHQVisionModel](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQVisionModel) (SamHQVisionModel model)
  + [SamVisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamVisionConfig) configuration class: [SamVisionModel](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamVisionModel) (SamVisionModel model)
  + [SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig) configuration class: [SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel) (SeamlessM4T model)
  + [SeamlessM4Tv2Config](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Config) configuration class: [SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model) (SeamlessM4Tv2 model)
  + [SeedOssConfig](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssConfig) configuration class: [SeedOssModel](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssModel) (SeedOss model)
  + [SegGptConfig](/docs/transformers/v4.56.2/en/model_doc/seggpt#transformers.SegGptConfig) configuration class: [SegGptModel](/docs/transformers/v4.56.2/en/model_doc/seggpt#transformers.SegGptModel) (SegGPT model)
  + [SegformerConfig](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerConfig) configuration class: [SegformerModel](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerModel) (SegFormer model)
  + [Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config) configuration class: [Siglip2Model](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Model) (SigLIP2 model)
  + [SiglipConfig](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipConfig) configuration class: [SiglipModel](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipModel) (SigLIP model)
  + [SiglipVisionConfig](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipVisionConfig) configuration class: [SiglipVisionModel](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipVisionModel) (SiglipVisionModel model)
  + [SmolLM3Config](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Config) configuration class: [SmolLM3Model](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Model) (SmolLM3 model)
  + [SmolVLMConfig](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMConfig) configuration class: [SmolVLMModel](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMModel) (SmolVLM model)
  + [SmolVLMVisionConfig](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMVisionConfig) configuration class: [SmolVLMVisionTransformer](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMVisionTransformer) (SmolVLMVisionTransformer model)
  + [Speech2TextConfig](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextConfig) configuration class: [Speech2TextModel](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextModel) (Speech2Text model)
  + [SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config) configuration class: [SpeechT5Model](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Model) (SpeechT5 model)
  + [SplinterConfig](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterConfig) configuration class: [SplinterModel](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterModel) (Splinter model)
  + [SqueezeBertConfig](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig) configuration class: [SqueezeBertModel](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertModel) (SqueezeBERT model)
  + [StableLmConfig](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmConfig) configuration class: [StableLmModel](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmModel) (StableLm model)
  + [Starcoder2Config](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2Config) configuration class: [Starcoder2Model](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2Model) (Starcoder2 model)
  + [SwiftFormerConfig](/docs/transformers/v4.56.2/en/model_doc/swiftformer#transformers.SwiftFormerConfig) configuration class: [SwiftFormerModel](/docs/transformers/v4.56.2/en/model_doc/swiftformer#transformers.SwiftFormerModel) (SwiftFormer model)
  + [Swin2SRConfig](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRConfig) configuration class: [Swin2SRModel](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRModel) (Swin2SR model)
  + [SwinConfig](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinConfig) configuration class: [SwinModel](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinModel) (Swin Transformer model)
  + [Swinv2Config](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2Config) configuration class: [Swinv2Model](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2Model) (Swin Transformer V2 model)
  + [SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig) configuration class: [SwitchTransformersModel](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersModel) (SwitchTransformers model)
  + [T5Config](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config) configuration class: [T5Model](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Model) (T5 model)
  + [T5GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig) configuration class: [T5GemmaModel](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaModel) (T5Gemma model)
  + [TableTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/table-transformer#transformers.TableTransformerConfig) configuration class: [TableTransformerModel](/docs/transformers/v4.56.2/en/model_doc/table-transformer#transformers.TableTransformerModel) (Table Transformer model)
  + [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) configuration class: [TapasModel](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasModel) (TAPAS model)
  + [TextNetConfig](/docs/transformers/v4.56.2/en/model_doc/textnet#transformers.TextNetConfig) configuration class: [TextNetModel](/docs/transformers/v4.56.2/en/model_doc/textnet#transformers.TextNetModel) (TextNet model)
  + [TimeSeriesTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerConfig) configuration class: [TimeSeriesTransformerModel](/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerModel) (Time Series Transformer model)
  + [TimesFmConfig](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmConfig) configuration class: [TimesFmModel](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmModel) (TimesFm model)
  + [TimesformerConfig](/docs/transformers/v4.56.2/en/model_doc/timesformer#transformers.TimesformerConfig) configuration class: [TimesformerModel](/docs/transformers/v4.56.2/en/model_doc/timesformer#transformers.TimesformerModel) (TimeSformer model)
  + [TimmBackboneConfig](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackboneConfig) configuration class: [TimmBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackbone) (TimmBackbone model)
  + [TimmWrapperConfig](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperConfig) configuration class: [TimmWrapperModel](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperModel) (TimmWrapperModel model)
  + [TrajectoryTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerConfig) configuration class: [TrajectoryTransformerModel](/docs/transformers/v4.56.2/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerModel) (Trajectory Transformer model)
  + [TransfoXLConfig](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLConfig) configuration class: [TransfoXLModel](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLModel) (Transformer-XL model)
  + [TvltConfig](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltConfig) configuration class: [TvltModel](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltModel) (TVLT model)
  + [TvpConfig](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpConfig) configuration class: [TvpModel](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpModel) (TVP model)
  + [UMT5Config](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config) configuration class: [UMT5Model](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Model) (UMT5 model)
  + [UdopConfig](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopConfig) configuration class: [UdopModel](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopModel) (UDOP model)
  + [UniSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechConfig) configuration class: [UniSpeechModel](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechModel) (UniSpeech model)
  + [UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig) configuration class: [UniSpeechSatModel](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel) (UniSpeechSat model)
  + [UnivNetConfig](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetConfig) configuration class: [UnivNetModel](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetModel) (UnivNet model)
  + [VJEPA2Config](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Config) configuration class: [VJEPA2Model](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Model) (VJEPA2Model model)
  + [VanConfig](/docs/transformers/v4.56.2/en/model_doc/van#transformers.VanConfig) configuration class: [VanModel](/docs/transformers/v4.56.2/en/model_doc/van#transformers.VanModel) (VAN model)
  + [ViTConfig](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTConfig) configuration class: [ViTModel](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTModel) (ViT model)
  + [ViTHybridConfig](/docs/transformers/v4.56.2/en/model_doc/vit_hybrid#transformers.ViTHybridConfig) configuration class: [ViTHybridModel](/docs/transformers/v4.56.2/en/model_doc/vit_hybrid#transformers.ViTHybridModel) (ViT Hybrid model)
  + [ViTMAEConfig](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEConfig) configuration class: [ViTMAEModel](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEModel) (ViTMAE model)
  + [ViTMSNConfig](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNConfig) configuration class: [ViTMSNModel](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNModel) (ViTMSN model)
  + [VideoLlavaConfig](/docs/transformers/v4.56.2/en/model_doc/video_llava#transformers.VideoLlavaConfig) configuration class: [VideoLlavaModel](/docs/transformers/v4.56.2/en/model_doc/video_llava#transformers.VideoLlavaModel) (VideoLlava model)
  + [VideoMAEConfig](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEConfig) configuration class: [VideoMAEModel](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEModel) (VideoMAE model)
  + [ViltConfig](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltConfig) configuration class: [ViltModel](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltModel) (ViLT model)
  + [VipLlavaConfig](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaConfig) configuration class: [VipLlavaModel](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaModel) (VipLlava model)
  + [VisionTextDualEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig) configuration class: [VisionTextDualEncoderModel](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel) (VisionTextDualEncoder model)
  + [VisualBertConfig](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig) configuration class: [VisualBertModel](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel) (VisualBERT model)
  + [VitDetConfig](/docs/transformers/v4.56.2/en/model_doc/vitdet#transformers.VitDetConfig) configuration class: [VitDetModel](/docs/transformers/v4.56.2/en/model_doc/vitdet#transformers.VitDetModel) (VitDet model)
  + [VitsConfig](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsConfig) configuration class: [VitsModel](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsModel) (VITS model)
  + [VivitConfig](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitConfig) configuration class: [VivitModel](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitModel) (ViViT model)
  + [VoxtralConfig](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralConfig) configuration class: [VoxtralForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralForConditionalGeneration) (Voxtral model)
  + [VoxtralEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralEncoderConfig) configuration class: [VoxtralEncoder](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralEncoder) (Voxtral Encoder model)
  + [Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig) configuration class: [Wav2Vec2BertModel](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertModel) (Wav2Vec2-BERT model)
  + [Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) configuration class: [Wav2Vec2Model](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Model) (Wav2Vec2 model)
  + [Wav2Vec2ConformerConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig) configuration class: [Wav2Vec2ConformerModel](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerModel) (Wav2Vec2-Conformer model)
  + [WavLMConfig](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMConfig) configuration class: [WavLMModel](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMModel) (WavLM model)
  + [WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig) configuration class: [WhisperModel](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperModel) (Whisper model)
  + [XCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/xclip#transformers.XCLIPConfig) configuration class: [XCLIPModel](/docs/transformers/v4.56.2/en/model_doc/xclip#transformers.XCLIPModel) (X-CLIP model)
  + [XGLMConfig](/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMConfig) configuration class: [XGLMModel](/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMModel) (XGLM model)
  + [XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig) configuration class: [XLMModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMModel) (XLM model)
  + [XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig) configuration class: [XLMProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel) (XLM-ProphetNet model)
  + [XLMRobertaConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig) configuration class: [XLMRobertaModel](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaModel) (XLM-RoBERTa model)
  + [XLMRobertaXLConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig) configuration class: [XLMRobertaXLModel](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLModel) (XLM-RoBERTa-XL model)
  + [XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig) configuration class: [XLNetModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel) (XLNet model)
  + [XcodecConfig](/docs/transformers/v4.56.2/en/model_doc/xcodec#transformers.XcodecConfig) configuration class: [XcodecModel](/docs/transformers/v4.56.2/en/model_doc/xcodec#transformers.XcodecModel) (X-CODEC model)
  + [XmodConfig](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig) configuration class: [XmodModel](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodModel) (X-MOD model)
  + [YolosConfig](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosConfig) configuration class: [YolosModel](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosModel) (YOLOS model)
  + [YosoConfig](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig) configuration class: [YosoModel](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoModel) (YOSO model)
  + [Zamba2Config](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2Config) configuration class: [Zamba2Model](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2Model) (Zamba2 model)
  + [ZambaConfig](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaConfig) configuration class: [ZambaModel](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaModel) (Zamba model)
  + [xLSTMConfig](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMConfig) configuration class: [xLSTMModel](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMModel) (xLSTM model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the base model classes of the library from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModel

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModel.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the base model classes of the library from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **aimv2** — [Aimv2Model](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Model) (AIMv2 model)
* **aimv2\_vision\_model** — [Aimv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionModel) (Aimv2VisionModel model)
* **albert** — `AlbertModel` (ALBERT model)
* **align** — [AlignModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignModel) (ALIGN model)
* **altclip** — [AltCLIPModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPModel) (AltCLIP model)
* **apertus** — [ApertusModel](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusModel) (Apertus model)
* **arcee** — [ArceeModel](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeModel) (Arcee model)
* **aria** — [AriaModel](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaModel) (Aria model)
* **aria\_text** — [AriaTextModel](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextModel) (AriaText model)
* **audio-spectrogram-transformer** — [ASTModel](/docs/transformers/v4.56.2/en/model_doc/audio-spectrogram-transformer#transformers.ASTModel) (Audio Spectrogram Transformer model)
* **autoformer** — [AutoformerModel](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerModel) (Autoformer model)
* **aya\_vision** — [AyaVisionModel](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionModel) (AyaVision model)
* **bamba** — [BambaModel](/docs/transformers/v4.56.2/en/model_doc/bamba#transformers.BambaModel) (Bamba model)
* **bark** — [BarkModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkModel) (Bark model)
* **bart** — [BartModel](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartModel) (BART model)
* **beit** — [BeitModel](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitModel) (BEiT model)
* **bert** — [BertModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel) (BERT model)
* **bert-generation** — [BertGenerationEncoder](/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationEncoder) (Bert Generation model)
* **big\_bird** — [BigBirdModel](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdModel) (BigBird model)
* **bigbird\_pegasus** — [BigBirdPegasusModel](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusModel) (BigBird-Pegasus model)
* **biogpt** — [BioGptModel](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptModel) (BioGpt model)
* **bit** — [BitModel](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitModel) (BiT model)
* **bitnet** — [BitNetModel](/docs/transformers/v4.56.2/en/model_doc/bitnet#transformers.BitNetModel) (BitNet model)
* **blenderbot** — [BlenderbotModel](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotModel) (Blenderbot model)
* **blenderbot-small** — [BlenderbotSmallModel](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel) (BlenderbotSmall model)
* **blip** — [BlipModel](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipModel) (BLIP model)
* **blip-2** — [Blip2Model](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Model) (BLIP-2 model)
* **blip\_2\_qformer** — [Blip2QFormerModel](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2QFormerModel) (BLIP-2 QFormer model)
* **bloom** — [BloomModel](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomModel) (BLOOM model)
* **bridgetower** — [BridgeTowerModel](/docs/transformers/v4.56.2/en/model_doc/bridgetower#transformers.BridgeTowerModel) (BridgeTower model)
* **bros** — [BrosModel](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosModel) (BROS model)
* **camembert** — [CamembertModel](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertModel) (CamemBERT model)
* **canine** — [CanineModel](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineModel) (CANINE model)
* **chameleon** — [ChameleonModel](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonModel) (Chameleon model)
* **chinese\_clip** — [ChineseCLIPModel](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPModel) (Chinese-CLIP model)
* **chinese\_clip\_vision\_model** — [ChineseCLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPVisionModel) (ChineseCLIPVisionModel model)
* **clap** — [ClapModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapModel) (CLAP model)
* **clip** — [CLIPModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPModel) (CLIP model)
* **clip\_text\_model** — [CLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTextModel) (CLIPTextModel model)
* **clip\_vision\_model** — [CLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPVisionModel) (CLIPVisionModel model)
* **clipseg** — [CLIPSegModel](/docs/transformers/v4.56.2/en/model_doc/clipseg#transformers.CLIPSegModel) (CLIPSeg model)
* **clvp** — [ClvpModelForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpModelForConditionalGeneration) (CLVP model)
* **code\_llama** — [LlamaModel](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel) (CodeLlama model)
* **codegen** — [CodeGenModel](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenModel) (CodeGen model)
* **cohere** — [CohereModel](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereModel) (Cohere model)
* **cohere2** — [Cohere2Model](/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2Model) (Cohere2 model)
* **cohere2\_vision** — [Cohere2VisionModel](/docs/transformers/v4.56.2/en/model_doc/cohere2_vision#transformers.Cohere2VisionModel) (Cohere2Vision model)
* **conditional\_detr** — [ConditionalDetrModel](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrModel) (Conditional DETR model)
* **convbert** — [ConvBertModel](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertModel) (ConvBERT model)
* **convnext** — [ConvNextModel](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextModel) (ConvNeXT model)
* **convnextv2** — [ConvNextV2Model](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Model) (ConvNeXTV2 model)
* **cpmant** — [CpmAntModel](/docs/transformers/v4.56.2/en/model_doc/cpmant#transformers.CpmAntModel) (CPM-Ant model)
* **csm** — [CsmForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmForConditionalGeneration) (CSM model)
* **ctrl** — [CTRLModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel) (CTRL model)
* **cvt** — [CvtModel](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtModel) (CvT model)
* **d\_fine** — [DFineModel](/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineModel) (D-FINE model)
* **dab-detr** — [DabDetrModel](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrModel) (DAB-DETR model)
* **dac** — [DacModel](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel) (DAC model)
* **data2vec-audio** — [Data2VecAudioModel](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioModel) (Data2VecAudio model)
* **data2vec-text** — [Data2VecTextModel](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextModel) (Data2VecText model)
* **data2vec-vision** — [Data2VecVisionModel](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionModel) (Data2VecVision model)
* **dbrx** — [DbrxModel](/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxModel) (DBRX model)
* **deberta** — [DebertaModel](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaModel) (DeBERTa model)
* **deberta-v2** — [DebertaV2Model](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Model) (DeBERTa-v2 model)
* **decision\_transformer** — [DecisionTransformerModel](/docs/transformers/v4.56.2/en/model_doc/decision_transformer#transformers.DecisionTransformerModel) (Decision Transformer model)
* **deepseek\_v2** — [DeepseekV2Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Model) (DeepSeek-V2 model)
* **deepseek\_v3** — [DeepseekV3Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Model) (DeepSeek-V3 model)
* **deepseek\_vl** — [DeepseekVLModel](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLModel) (DeepseekVL model)
* **deepseek\_vl\_hybrid** — [DeepseekVLHybridModel](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridModel) (DeepseekVLHybrid model)
* **deformable\_detr** — [DeformableDetrModel](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrModel) (Deformable DETR model)
* **deit** — [DeiTModel](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTModel) (DeiT model)
* **depth\_pro** — [DepthProModel](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProModel) (DepthPro model)
* **deta** — [DetaModel](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaModel) (DETA model)
* **detr** — [DetrModel](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrModel) (DETR model)
* **dia** — [DiaModel](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel) (Dia model)
* **diffllama** — [DiffLlamaModel](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaModel) (DiffLlama model)
* **dinat** — [DinatModel](/docs/transformers/v4.56.2/en/model_doc/dinat#transformers.DinatModel) (DiNAT model)
* **dinov2** — [Dinov2Model](/docs/transformers/v4.56.2/en/model_doc/dinov2#transformers.Dinov2Model) (DINOv2 model)
* **dinov2\_with\_registers** — [Dinov2WithRegistersModel](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersModel) (DINOv2 with Registers model)
* **dinov3\_convnext** — [DINOv3ConvNextModel](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ConvNextModel) (DINOv3 ConvNext model)
* **dinov3\_vit** — [DINOv3ViTModel](/docs/transformers/v4.56.2/en/model_doc/dinov3#transformers.DINOv3ViTModel) (DINOv3 ViT model)
* **distilbert** — [DistilBertModel](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertModel) (DistilBERT model)
* **doge** — [DogeModel](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeModel) (Doge model)
* **donut-swin** — [DonutSwinModel](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinModel) (DonutSwin model)
* **dots1** — [Dots1Model](/docs/transformers/v4.56.2/en/model_doc/dots1#transformers.Dots1Model) (dots1 model)
* **dpr** — [DPRQuestionEncoder](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoder) (DPR model)
* **dpt** — [DPTModel](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTModel) (DPT model)
* **efficientformer** — [EfficientFormerModel](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerModel) (EfficientFormer model)
* **efficientloftr** — [EfficientLoFTRModel](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRModel) (EfficientLoFTR model)
* **efficientnet** — [EfficientNetModel](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetModel) (EfficientNet model)
* **electra** — [ElectraModel](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraModel) (ELECTRA model)
* **emu3** — [Emu3Model](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Model) (Emu3 model)
* **encodec** — [EncodecModel](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel) (EnCodec model)
* **ernie** — [ErnieModel](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieModel) (ERNIE model)
* **ernie4\_5** — [Ernie4\_5Model](/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Model) (Ernie4\_5 model)
* **ernie4\_5\_moe** — [Ernie4\_5\_MoeModel](/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeModel) (Ernie4\_5\_MoE model)
* **ernie\_m** — [ErnieMModel](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMModel) (ErnieM model)
* **esm** — [EsmModel](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmModel) (ESM model)
* **evolla** — [EvollaModel](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaModel) (Evolla model)
* **exaone4** — [Exaone4Model](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Model) (EXAONE-4.0 model)
* **falcon** — [FalconModel](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconModel) (Falcon model)
* **falcon\_h1** — [FalconH1Model](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1Model) (FalconH1 model)
* **falcon\_mamba** — [FalconMambaModel](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaModel) (FalconMamba model)
* **fastspeech2\_conformer** — [FastSpeech2ConformerModel](/docs/transformers/v4.56.2/en/model_doc/fastspeech2_conformer#transformers.FastSpeech2ConformerModel) (FastSpeech2Conformer model)
* **fastspeech2\_conformer\_with\_hifigan** — [FastSpeech2ConformerWithHifiGan](/docs/transformers/v4.56.2/en/model_doc/fastspeech2_conformer#transformers.FastSpeech2ConformerWithHifiGan) (FastSpeech2ConformerWithHifiGan model)
* **flaubert** — [FlaubertModel](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertModel) (FlauBERT model)
* **flava** — [FlavaModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaModel) (FLAVA model)
* **florence2** — [Florence2Model](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Model) (Florence2 model)
* **fnet** — [FNetModel](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetModel) (FNet model)
* **focalnet** — [FocalNetModel](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetModel) (FocalNet model)
* **fsmt** — [FSMTModel](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTModel) (FairSeq Machine-Translation model)
* **funnel** — [FunnelModel](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelModel) or [FunnelBaseModel](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelBaseModel) (Funnel Transformer model)
* **fuyu** — [FuyuModel](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuModel) (Fuyu model)
* **gemma** — [GemmaModel](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaModel) (Gemma model)
* **gemma2** — [Gemma2Model](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Model) (Gemma2 model)
* **gemma3** — [Gemma3Model](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Model) (Gemma3ForConditionalGeneration model)
* **gemma3\_text** — [Gemma3TextModel](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextModel) (Gemma3ForCausalLM model)
* **gemma3n** — [Gemma3nModel](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nModel) (Gemma3nForConditionalGeneration model)
* **gemma3n\_audio** — `Gemma3nAudioEncoder` (Gemma3nAudioEncoder model)
* **gemma3n\_text** — [Gemma3nTextModel](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextModel) (Gemma3nForCausalLM model)
* **gemma3n\_vision** — [TimmWrapperModel](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperModel) (TimmWrapperModel model)
* **git** — [GitModel](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitModel) (GIT model)
* **glm** — [GlmModel](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmModel) (GLM model)
* **glm4** — [Glm4Model](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Model) (GLM4 model)
* **glm4\_moe** — [Glm4MoeModel](/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeModel) (Glm4MoE model)
* **glm4v** — [Glm4vModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vModel) (GLM4V model)
* **glm4v\_moe** — [Glm4vMoeModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeModel) (GLM4VMOE model)
* **glm4v\_moe\_text** — [Glm4vMoeTextModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeTextModel) (GLM4VMOE model)
* **glm4v\_text** — [Glm4vTextModel](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vTextModel) (GLM4V model)
* **glpn** — [GLPNModel](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNModel) (GLPN model)
* **got\_ocr2** — [GotOcr2Model](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Model) (GOT-OCR2 model)
* **gpt-sw3** — [GPT2Model](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Model) (GPT-Sw3 model)
* **gpt2** — [GPT2Model](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Model) (OpenAI GPT-2 model)
* **gpt\_bigcode** — [GPTBigCodeModel](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeModel) (GPTBigCode model)
* **gpt\_neo** — [GPTNeoModel](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoModel) (GPT Neo model)
* **gpt\_neox** — [GPTNeoXModel](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXModel) (GPT NeoX model)
* **gpt\_neox\_japanese** — [GPTNeoXJapaneseModel](/docs/transformers/v4.56.2/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseModel) (GPT NeoX Japanese model)
* **gpt\_oss** — [GptOssModel](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssModel) (GptOss model)
* **gptj** — [GPTJModel](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJModel) (GPT-J model)
* **gptsan-japanese** — [GPTSanJapaneseForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseForConditionalGeneration) (GPTSAN-japanese model)
* **granite** — [GraniteModel](/docs/transformers/v4.56.2/en/model_doc/granite#transformers.GraniteModel) (Granite model)
* **granitemoe** — [GraniteMoeModel](/docs/transformers/v4.56.2/en/model_doc/granitemoe#transformers.GraniteMoeModel) (GraniteMoeMoe model)
* **granitemoehybrid** — [GraniteMoeHybridModel](/docs/transformers/v4.56.2/en/model_doc/granitemoehybrid#transformers.GraniteMoeHybridModel) (GraniteMoeHybrid model)
* **granitemoeshared** — [GraniteMoeSharedModel](/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedModel) (GraniteMoeSharedMoe model)
* **graphormer** — [GraphormerModel](/docs/transformers/v4.56.2/en/model_doc/graphormer#transformers.GraphormerModel) (Graphormer model)
* **grounding-dino** — [GroundingDinoModel](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoModel) (Grounding DINO model)
* **groupvit** — [GroupViTModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTModel) (GroupViT model)
* **helium** — [HeliumModel](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumModel) (Helium model)
* **hgnet\_v2** — [HGNetV2Backbone](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Backbone) (HGNet-V2 model)
* **hiera** — [HieraModel](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraModel) (Hiera model)
* **hubert** — [HubertModel](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertModel) (Hubert model)
* **hunyuan\_v1\_dense** — [HunYuanDenseV1Model](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Model) (HunYuanDenseV1 model)
* **hunyuan\_v1\_moe** — [HunYuanMoEV1Model](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Model) (HunYuanMoeV1 model)
* **ibert** — [IBertModel](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertModel) (I-BERT model)
* **idefics** — [IdeficsModel](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsModel) (IDEFICS model)
* **idefics2** — [Idefics2Model](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Model) (Idefics2 model)
* **idefics3** — [Idefics3Model](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3Model) (Idefics3 model)
* **idefics3\_vision** — [Idefics3VisionTransformer](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3VisionTransformer) (Idefics3VisionTransformer model)
* **ijepa** — [IJepaModel](/docs/transformers/v4.56.2/en/model_doc/ijepa#transformers.IJepaModel) (I-JEPA model)
* **imagegpt** — [ImageGPTModel](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTModel) (ImageGPT model)
* **informer** — [InformerModel](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerModel) (Informer model)
* **instructblip** — [InstructBlipModel](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipModel) (InstructBLIP model)
* **instructblipvideo** — [InstructBlipVideoModel](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoModel) (InstructBlipVideo model)
* **internvl** — [InternVLModel](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLModel) (InternVL model)
* **internvl\_vision** — [InternVLVisionModel](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLVisionModel) (InternVLVision model)
* **jamba** — [JambaModel](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaModel) (Jamba model)
* **janus** — [JanusModel](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusModel) (Janus model)
* **jetmoe** — [JetMoeModel](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeModel) (JetMoe model)
* **jukebox** — [JukeboxModel](/docs/transformers/v4.56.2/en/model_doc/jukebox#transformers.JukeboxModel) (Jukebox model)
* **kosmos-2** — [Kosmos2Model](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Model) (KOSMOS-2 model)
* **kosmos-2.5** — [Kosmos2\_5Model](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Model) (KOSMOS-2.5 model)
* **kyutai\_speech\_to\_text** — [KyutaiSpeechToTextModel](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextModel) (KyutaiSpeechToText model)
* **layoutlm** — [LayoutLMModel](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMModel) (LayoutLM model)
* **layoutlmv2** — [LayoutLMv2Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Model) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model) (LayoutLMv3 model)
* **led** — [LEDModel](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDModel) (LED model)
* **levit** — [LevitModel](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitModel) (LeViT model)
* **lfm2** — [Lfm2Model](/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Model) (Lfm2 model)
* **lightglue** — [LightGlueForKeypointMatching](/docs/transformers/v4.56.2/en/model_doc/lightglue#transformers.LightGlueForKeypointMatching) (LightGlue model)
* **lilt** — [LiltModel](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltModel) (LiLT model)
* **llama** — [LlamaModel](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel) (LLaMA model)
* **llama4** — [Llama4ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForConditionalGeneration) (Llama4 model)
* **llama4\_text** — [Llama4TextModel](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4TextModel) (Llama4ForCausalLM model)
* **llava** — [LlavaModel](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaModel) (LLaVa model)
* **llava\_next** — [LlavaNextModel](/docs/transformers/v4.56.2/en/model_doc/llava_next#transformers.LlavaNextModel) (LLaVA-NeXT model)
* **llava\_next\_video** — [LlavaNextVideoModel](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoModel) (LLaVa-NeXT-Video model)
* **llava\_onevision** — [LlavaOnevisionModel](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionModel) (LLaVA-Onevision model)
* **longformer** — [LongformerModel](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerModel) (Longformer model)
* **longt5** — [LongT5Model](/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Model) (LongT5 model)
* **luke** — [LukeModel](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeModel) (LUKE model)
* **lxmert** — [LxmertModel](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertModel) (LXMERT model)
* **m2m\_100** — [M2M100Model](/docs/transformers/v4.56.2/en/model_doc/m2m_100#transformers.M2M100Model) (M2M100 model)
* **mamba** — [MambaModel](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaModel) (Mamba model)
* **mamba2** — [Mamba2Model](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2Model) (mamba2 model)
* **marian** — [MarianModel](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianModel) (Marian model)
* **markuplm** — [MarkupLMModel](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMModel) (MarkupLM model)
* **mask2former** — [Mask2FormerModel](/docs/transformers/v4.56.2/en/model_doc/mask2former#transformers.Mask2FormerModel) (Mask2Former model)
* **maskformer** — [MaskFormerModel](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerModel) (MaskFormer model)
* **maskformer-swin** — `MaskFormerSwinModel` (MaskFormerSwin model)
* **mbart** — [MBartModel](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartModel) (mBART model)
* **mctct** — [MCTCTModel](/docs/transformers/v4.56.2/en/model_doc/mctct#transformers.MCTCTModel) (M-CTC-T model)
* **mega** — [MegaModel](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel) (MEGA model)
* **megatron-bert** — [MegatronBertModel](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertModel) (Megatron-BERT model)
* **metaclip\_2** — [MetaClip2Model](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Model) (MetaCLIP 2 model)
* **mgp-str** — [MgpstrForSceneTextRecognition](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrForSceneTextRecognition) (MGP-STR model)
* **mimi** — [MimiModel](/docs/transformers/v4.56.2/en/model_doc/mimi#transformers.MimiModel) (Mimi model)
* **minimax** — [MiniMaxModel](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxModel) (MiniMax model)
* **mistral** — [MistralModel](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralModel) (Mistral model)
* **mistral3** — [Mistral3Model](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Model) (Mistral3 model)
* **mixtral** — [MixtralModel](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralModel) (Mixtral model)
* **mlcd** — [MLCDVisionModel](/docs/transformers/v4.56.2/en/model_doc/mlcd#transformers.MLCDVisionModel) (MLCD model)
* **mllama** — [MllamaModel](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaModel) (Mllama model)
* **mm-grounding-dino** — [MMGroundingDinoModel](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoModel) (MM Grounding DINO model)
* **mobilebert** — [MobileBertModel](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertModel) (MobileBERT model)
* **mobilenet\_v1** — [MobileNetV1Model](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1Model) (MobileNetV1 model)
* **mobilenet\_v2** — [MobileNetV2Model](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2Model) (MobileNetV2 model)
* **mobilevit** — [MobileViTModel](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTModel) (MobileViT model)
* **mobilevitv2** — [MobileViTV2Model](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Model) (MobileViTV2 model)
* **modernbert** — [ModernBertModel](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertModel) (ModernBERT model)
* **modernbert-decoder** — [ModernBertDecoderModel](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderModel) (ModernBertDecoder model)
* **moonshine** — [MoonshineModel](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineModel) (Moonshine model)
* **moshi** — [MoshiModel](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiModel) (Moshi model)
* **mpnet** — [MPNetModel](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetModel) (MPNet model)
* **mpt** — [MptModel](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptModel) (MPT model)
* **mra** — [MraModel](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraModel) (MRA model)
* **mt5** — [MT5Model](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Model) (MT5 model)
* **musicgen** — [MusicgenModel](/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenModel) (MusicGen model)
* **musicgen\_melody** — [MusicgenMelodyModel](/docs/transformers/v4.56.2/en/model_doc/musicgen_melody#transformers.MusicgenMelodyModel) (MusicGen Melody model)
* **mvp** — [MvpModel](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpModel) (MVP model)
* **nat** — [NatModel](/docs/transformers/v4.56.2/en/model_doc/nat#transformers.NatModel) (NAT model)
* **nemotron** — [NemotronModel](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronModel) (Nemotron model)
* **nezha** — [NezhaModel](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaModel) (Nezha model)
* **nllb-moe** — [NllbMoeModel](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeModel) (NLLB-MOE model)
* **nystromformer** — [NystromformerModel](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerModel) (Nyströmformer model)
* **olmo** — [OlmoModel](/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoModel) (OLMo model)
* **olmo2** — [Olmo2Model](/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Model) (OLMo2 model)
* **olmoe** — [OlmoeModel](/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeModel) (OLMoE model)
* **omdet-turbo** — [OmDetTurboForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboForObjectDetection) (OmDet-Turbo model)
* **oneformer** — [OneFormerModel](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerModel) (OneFormer model)
* **open-llama** — [OpenLlamaModel](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaModel) (OpenLlama model)
* **openai-gpt** — [OpenAIGPTModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTModel) (OpenAI GPT model)
* **opt** — [OPTModel](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTModel) (OPT model)
* **ovis2** — [Ovis2Model](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Model) (Ovis2 model)
* **owlv2** — [Owlv2Model](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Model) (OWLv2 model)
* **owlvit** — [OwlViTModel](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTModel) (OWL-ViT model)
* **paligemma** — [PaliGemmaModel](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaModel) (PaliGemma model)
* **patchtsmixer** — [PatchTSMixerModel](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerModel) (PatchTSMixer model)
* **patchtst** — [PatchTSTModel](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTModel) (PatchTST model)
* **pegasus** — [PegasusModel](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusModel) (Pegasus model)
* **pegasus\_x** — [PegasusXModel](/docs/transformers/v4.56.2/en/model_doc/pegasus_x#transformers.PegasusXModel) (PEGASUS-X model)
* **perceiver** — [PerceiverModel](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverModel) (Perceiver model)
* **perception\_encoder** — `PerceptionEncoder` (PerceptionEncoder model)
* **perception\_lm** — [PerceptionLMModel](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMModel) (PerceptionLM model)
* **persimmon** — [PersimmonModel](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonModel) (Persimmon model)
* **phi** — [PhiModel](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiModel) (Phi model)
* **phi3** — [Phi3Model](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Model) (Phi3 model)
* **phi4\_multimodal** — [Phi4MultimodalModel](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalModel) (Phi4Multimodal model)
* **phimoe** — [PhimoeModel](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeModel) (Phimoe model)
* **pixtral** — [PixtralVisionModel](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionModel) (Pixtral model)
* **plbart** — [PLBartModel](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartModel) (PLBart model)
* **poolformer** — [PoolFormerModel](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerModel) (PoolFormer model)
* **prophetnet** — [ProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetModel) (ProphetNet model)
* **pvt** — [PvtModel](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtModel) (PVT model)
* **pvt\_v2** — [PvtV2Model](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Model) (PVTv2 model)
* **qdqbert** — [QDQBertModel](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertModel) (QDQBert model)
* **qwen2** — [Qwen2Model](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Model) (Qwen2 model)
* **qwen2\_5\_vl** — [Qwen2\_5\_VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLModel) (Qwen2\_5\_VL model)
* **qwen2\_5\_vl\_text** — [Qwen2\_5\_VLTextModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLTextModel) (Qwen2\_5\_VL model)
* **qwen2\_audio\_encoder** — [Qwen2AudioEncoder](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioEncoder) (Qwen2AudioEncoder model)
* **qwen2\_moe** — [Qwen2MoeModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeModel) (Qwen2MoE model)
* **qwen2\_vl** — [Qwen2VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLModel) (Qwen2VL model)
* **qwen2\_vl\_text** — [Qwen2VLTextModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLTextModel) (Qwen2VL model)
* **qwen3** — [Qwen3Model](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Model) (Qwen3 model)
* **qwen3\_moe** — [Qwen3MoeModel](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeModel) (Qwen3MoE model)
* **recurrent\_gemma** — [RecurrentGemmaModel](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaModel) (RecurrentGemma model)
* **reformer** — [ReformerModel](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModel) (Reformer model)
* **regnet** — [RegNetModel](/docs/transformers/v4.56.2/en/model_doc/regnet#transformers.RegNetModel) (RegNet model)
* **rembert** — [RemBertModel](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertModel) (RemBERT model)
* **resnet** — [ResNetModel](/docs/transformers/v4.56.2/en/model_doc/resnet#transformers.ResNetModel) (ResNet model)
* **retribert** — [RetriBertModel](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertModel) (RetriBERT model)
* **roberta** — [RobertaModel](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaModel) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaPreLayerNormModel](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormModel) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertModel](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertModel) (RoCBert model)
* **roformer** — [RoFormerModel](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerModel) (RoFormer model)
* **rt\_detr** — [RTDetrModel](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrModel) (RT-DETR model)
* **rt\_detr\_v2** — [RTDetrV2Model](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Model) (RT-DETRv2 model)
* **rwkv** — [RwkvModel](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvModel) (RWKV model)
* **sam** — [SamModel](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamModel) (SAM model)
* **sam2** — [Sam2Model](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Model) (SAM2 model)
* **sam2\_hiera\_det\_model** — [Sam2HieraDetModel](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2HieraDetModel) (Sam2HieraDetModel model)
* **sam2\_video** — [Sam2VideoModel](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoModel) (Sam2VideoModel model)
* **sam2\_vision\_model** — [Sam2VisionModel](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2VisionModel) (Sam2VisionModel model)
* **sam\_hq** — [SamHQModel](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQModel) (SAM-HQ model)
* **sam\_hq\_vision\_model** — [SamHQVisionModel](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQVisionModel) (SamHQVisionModel model)
* **sam\_vision\_model** — [SamVisionModel](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamVisionModel) (SamVisionModel model)
* **seamless\_m4t** — [SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel) (SeamlessM4T model)
* **seamless\_m4t\_v2** — [SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model) (SeamlessM4Tv2 model)
* **seed\_oss** — [SeedOssModel](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssModel) (SeedOss model)
* **segformer** — [SegformerModel](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerModel) (SegFormer model)
* **seggpt** — [SegGptModel](/docs/transformers/v4.56.2/en/model_doc/seggpt#transformers.SegGptModel) (SegGPT model)
* **sew** — [SEWModel](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWModel) (SEW model)
* **sew-d** — [SEWDModel](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDModel) (SEW-D model)
* **siglip** — [SiglipModel](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipModel) (SigLIP model)
* **siglip2** — [Siglip2Model](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Model) (SigLIP2 model)
* **siglip\_vision\_model** — [SiglipVisionModel](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipVisionModel) (SiglipVisionModel model)
* **smollm3** — [SmolLM3Model](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Model) (SmolLM3 model)
* **smolvlm** — [SmolVLMModel](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMModel) (SmolVLM model)
* **smolvlm\_vision** — [SmolVLMVisionTransformer](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMVisionTransformer) (SmolVLMVisionTransformer model)
* **speech\_to\_text** — [Speech2TextModel](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextModel) (Speech2Text model)
* **speecht5** — [SpeechT5Model](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Model) (SpeechT5 model)
* **splinter** — [SplinterModel](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterModel) (Splinter model)
* **squeezebert** — [SqueezeBertModel](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertModel) (SqueezeBERT model)
* **stablelm** — [StableLmModel](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmModel) (StableLm model)
* **starcoder2** — [Starcoder2Model](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2Model) (Starcoder2 model)
* **swiftformer** — [SwiftFormerModel](/docs/transformers/v4.56.2/en/model_doc/swiftformer#transformers.SwiftFormerModel) (SwiftFormer model)
* **swin** — [SwinModel](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinModel) (Swin Transformer model)
* **swin2sr** — [Swin2SRModel](/docs/transformers/v4.56.2/en/model_doc/swin2sr#transformers.Swin2SRModel) (Swin2SR model)
* **swinv2** — [Swinv2Model](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2Model) (Swin Transformer V2 model)
* **switch\_transformers** — [SwitchTransformersModel](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersModel) (SwitchTransformers model)
* **t5** — [T5Model](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Model) (T5 model)
* **t5gemma** — [T5GemmaModel](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaModel) (T5Gemma model)
* **table-transformer** — [TableTransformerModel](/docs/transformers/v4.56.2/en/model_doc/table-transformer#transformers.TableTransformerModel) (Table Transformer model)
* **tapas** — [TapasModel](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasModel) (TAPAS model)
* **textnet** — [TextNetModel](/docs/transformers/v4.56.2/en/model_doc/textnet#transformers.TextNetModel) (TextNet model)
* **time\_series\_transformer** — [TimeSeriesTransformerModel](/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerModel) (Time Series Transformer model)
* **timesfm** — [TimesFmModel](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmModel) (TimesFm model)
* **timesformer** — [TimesformerModel](/docs/transformers/v4.56.2/en/model_doc/timesformer#transformers.TimesformerModel) (TimeSformer model)
* **timm\_backbone** — [TimmBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackbone) (TimmBackbone model)
* **timm\_wrapper** — [TimmWrapperModel](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperModel) (TimmWrapperModel model)
* **trajectory\_transformer** — [TrajectoryTransformerModel](/docs/transformers/v4.56.2/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerModel) (Trajectory Transformer model)
* **transfo-xl** — [TransfoXLModel](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLModel) (Transformer-XL model)
* **tvlt** — [TvltModel](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltModel) (TVLT model)
* **tvp** — [TvpModel](/docs/transformers/v4.56.2/en/model_doc/tvp#transformers.TvpModel) (TVP model)
* **udop** — [UdopModel](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopModel) (UDOP model)
* **umt5** — [UMT5Model](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Model) (UMT5 model)
* **unispeech** — [UniSpeechModel](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechModel) (UniSpeech model)
* **unispeech-sat** — [UniSpeechSatModel](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel) (UniSpeechSat model)
* **univnet** — [UnivNetModel](/docs/transformers/v4.56.2/en/model_doc/univnet#transformers.UnivNetModel) (UnivNet model)
* **van** — [VanModel](/docs/transformers/v4.56.2/en/model_doc/van#transformers.VanModel) (VAN model)
* **video\_llava** — [VideoLlavaModel](/docs/transformers/v4.56.2/en/model_doc/video_llava#transformers.VideoLlavaModel) (VideoLlava model)
* **videomae** — [VideoMAEModel](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEModel) (VideoMAE model)
* **vilt** — [ViltModel](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltModel) (ViLT model)
* **vipllava** — [VipLlavaModel](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaModel) (VipLlava model)
* **vision-text-dual-encoder** — [VisionTextDualEncoderModel](/docs/transformers/v4.56.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel) (VisionTextDualEncoder model)
* **visual\_bert** — [VisualBertModel](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel) (VisualBERT model)
* **vit** — [ViTModel](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTModel) (ViT model)
* **vit\_hybrid** — [ViTHybridModel](/docs/transformers/v4.56.2/en/model_doc/vit_hybrid#transformers.ViTHybridModel) (ViT Hybrid model)
* **vit\_mae** — [ViTMAEModel](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEModel) (ViTMAE model)
* **vit\_msn** — [ViTMSNModel](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNModel) (ViTMSN model)
* **vitdet** — [VitDetModel](/docs/transformers/v4.56.2/en/model_doc/vitdet#transformers.VitDetModel) (VitDet model)
* **vits** — [VitsModel](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsModel) (VITS model)
* **vivit** — [VivitModel](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitModel) (ViViT model)
* **vjepa2** — [VJEPA2Model](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Model) (VJEPA2Model model)
* **voxtral** — [VoxtralForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralForConditionalGeneration) (Voxtral model)
* **voxtral\_encoder** — [VoxtralEncoder](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralEncoder) (Voxtral Encoder model)
* **wav2vec2** — [Wav2Vec2Model](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Model) (Wav2Vec2 model)
* **wav2vec2-bert** — [Wav2Vec2BertModel](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertModel) (Wav2Vec2-BERT model)
* **wav2vec2-conformer** — [Wav2Vec2ConformerModel](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerModel) (Wav2Vec2-Conformer model)
* **wavlm** — [WavLMModel](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMModel) (WavLM model)
* **whisper** — [WhisperModel](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperModel) (Whisper model)
* **xclip** — [XCLIPModel](/docs/transformers/v4.56.2/en/model_doc/xclip#transformers.XCLIPModel) (X-CLIP model)
* **xcodec** — [XcodecModel](/docs/transformers/v4.56.2/en/model_doc/xcodec#transformers.XcodecModel) (X-CODEC model)
* **xglm** — [XGLMModel](/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMModel) (XGLM model)
* **xlm** — [XLMModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMModel) (XLM model)
* **xlm-prophetnet** — [XLMProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel) (XLM-ProphetNet model)
* **xlm-roberta** — [XLMRobertaModel](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaModel) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaXLModel](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLModel) (XLM-RoBERTa-XL model)
* **xlnet** — [XLNetModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel) (XLNet model)
* **xlstm** — [xLSTMModel](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMModel) (xLSTM model)
* **xmod** — [XmodModel](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodModel) (X-MOD model)
* **yolos** — [YolosModel](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosModel) (YOLOS model)
* **yoso** — [YosoModel](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoModel) (YOSO model)
* **zamba** — [ZambaModel](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaModel) (Zamba model)
* **zamba2** — [Zamba2Model](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2Model) (Zamba2 model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModel

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModel.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModel.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModel.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

## Generic pretraining classes

The following auto classes are available for instantiating a model with a pretraining head.

### AutoModelForPreTraining

### class transformers.AutoModelForPreTraining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1905)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a pretraining head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig) configuration class: `AlbertForPreTraining` (ALBERT model)
  + [BartConfig](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartConfig) configuration class: [BartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForConditionalGeneration) (BART model)
  + [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) configuration class: [BertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForPreTraining) (BERT model)
  + [BigBirdConfig](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig) configuration class: [BigBirdForPreTraining](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForPreTraining) (BigBird model)
  + [BloomConfig](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig) configuration class: [BloomForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForCausalLM) (BLOOM model)
  + [CTRLConfig](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig) configuration class: [CTRLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLLMHeadModel) (CTRL model)
  + [CamembertConfig](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig) configuration class: [CamembertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMaskedLM) (CamemBERT model)
  + [ColPaliConfig](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliConfig) configuration class: [ColPaliForRetrieval](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliForRetrieval) (ColPali model)
  + [ColQwen2Config](/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2Config) configuration class: [ColQwen2ForRetrieval](/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2ForRetrieval) (ColQwen2 model)
  + [Data2VecTextConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig) configuration class: [Data2VecTextForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMaskedLM) (Data2VecText model)
  + [DebertaConfig](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaConfig) configuration class: [DebertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForMaskedLM) (DeBERTa model)
  + [DebertaV2Config](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config) configuration class: [DebertaV2ForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMaskedLM) (DeBERTa-v2 model)
  + [DistilBertConfig](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertConfig) configuration class: [DistilBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForMaskedLM) (DistilBERT model)
  + [ElectraConfig](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig) configuration class: [ElectraForPreTraining](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForPreTraining) (ELECTRA model)
  + [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieForPreTraining](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForPreTraining) (ERNIE model)
  + [EvollaConfig](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaConfig) configuration class: [EvollaForProteinText2Text](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaForProteinText2Text) (Evolla model)
  + [Exaone4Config](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config) configuration class: [Exaone4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForCausalLM) (EXAONE-4.0 model)
  + [FNetConfig](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig) configuration class: [FNetForPreTraining](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForPreTraining) (FNet model)
  + [FSMTConfig](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTConfig) configuration class: [FSMTForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTForConditionalGeneration) (FairSeq Machine-Translation model)
  + [FalconMambaConfig](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaConfig) configuration class: [FalconMambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaForCausalLM) (FalconMamba model)
  + [FlaubertConfig](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertConfig) configuration class: [FlaubertWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertWithLMHeadModel) (FlauBERT model)
  + [FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig) configuration class: [FlavaForPreTraining](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaForPreTraining) (FLAVA model)
  + [Florence2Config](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Config) configuration class: [Florence2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2ForConditionalGeneration) (Florence2 model)
  + [FunnelConfig](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelConfig) configuration class: [FunnelForPreTraining](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForPreTraining) (Funnel Transformer model)
  + [GPT2Config](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config) configuration class: [GPT2LMHeadModel](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2LMHeadModel) (OpenAI GPT-2 model)
  + [GPTBigCodeConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeConfig) configuration class: [GPTBigCodeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeForCausalLM) (GPTBigCode model)
  + [GPTSanJapaneseConfig](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseConfig) configuration class: [GPTSanJapaneseForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseForConditionalGeneration) (GPTSAN-japanese model)
  + [Gemma3Config](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config) configuration class: [Gemma3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration) (Gemma3ForConditionalGeneration model)
  + [HieraConfig](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraConfig) configuration class: [HieraForPreTraining](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraForPreTraining) (Hiera model)
  + [IBertConfig](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertConfig) configuration class: [IBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForMaskedLM) (I-BERT model)
  + [Idefics2Config](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Config) configuration class: [Idefics2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ForConditionalGeneration) (Idefics2 model)
  + [Idefics3Config](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3Config) configuration class: [Idefics3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3ForConditionalGeneration) (Idefics3 model)
  + [IdeficsConfig](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsConfig) configuration class: [IdeficsForVisionText2Text](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsForVisionText2Text) (IDEFICS model)
  + [JanusConfig](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusConfig) configuration class: [JanusForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusForConditionalGeneration) (Janus model)
  + [LayoutLMConfig](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig) configuration class: [LayoutLMForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForMaskedLM) (LayoutLM model)
  + [LlavaConfig](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaConfig) configuration class: [LlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaForConditionalGeneration) (LLaVa model)
  + [LlavaNextConfig](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextConfig) configuration class: [LlavaNextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextForConditionalGeneration) (LLaVA-NeXT model)
  + [LlavaNextVideoConfig](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoConfig) configuration class: [LlavaNextVideoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoForConditionalGeneration) (LLaVa-NeXT-Video model)
  + [LlavaOnevisionConfig](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionConfig) configuration class: [LlavaOnevisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionForConditionalGeneration) (LLaVA-Onevision model)
  + [LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig) configuration class: [LongformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMaskedLM) (Longformer model)
  + [LukeConfig](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig) configuration class: [LukeForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMaskedLM) (LUKE model)
  + [LxmertConfig](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertConfig) configuration class: [LxmertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForPreTraining) (LXMERT model)
  + [MPNetConfig](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig) configuration class: [MPNetForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForMaskedLM) (MPNet model)
  + [Mamba2Config](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2Config) configuration class: [Mamba2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2ForCausalLM) (mamba2 model)
  + [MambaConfig](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaConfig) configuration class: [MambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaForCausalLM) (Mamba model)
  + [MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig) configuration class: [MegaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMaskedLM) (MEGA model)
  + [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForPreTraining) (Megatron-BERT model)
  + [Mistral3Config](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Config) configuration class: [Mistral3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration) (Mistral3 model)
  + [MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig) configuration class: [MllamaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForConditionalGeneration) (Mllama model)
  + [MobileBertConfig](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [MobileBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForPreTraining) (MobileBERT model)
  + [MptConfig](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptConfig) configuration class: [MptForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForCausalLM) (MPT model)
  + [MraConfig](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraConfig) configuration class: [MraForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForMaskedLM) (MRA model)
  + [MvpConfig](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig) configuration class: [MvpForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForConditionalGeneration) (MVP model)
  + [NezhaConfig](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig) configuration class: [NezhaForPreTraining](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForPreTraining) (Nezha model)
  + [NllbMoeConfig](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeConfig) configuration class: [NllbMoeForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeForConditionalGeneration) (NLLB-MOE model)
  + [OpenAIGPTConfig](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig) configuration class: [OpenAIGPTLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTLMHeadModel) (OpenAI GPT model)
  + [PaliGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaConfig) configuration class: [PaliGemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration) (PaliGemma model)
  + [Qwen2AudioConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioConfig) configuration class: [Qwen2AudioForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioForConditionalGeneration) (Qwen2Audio model)
  + [RetriBertConfig](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertConfig) configuration class: [RetriBertModel](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertModel) (RetriBERT model)
  + [RoCBertConfig](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig) configuration class: [RoCBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForPreTraining) (RoCBert model)
  + [RobertaConfig](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig) configuration class: [RobertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMaskedLM) (RoBERTa model)
  + [RobertaPreLayerNormConfig](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig) configuration class: [RobertaPreLayerNormForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMaskedLM) (RoBERTa-PreLayerNorm model)
  + [RwkvConfig](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvConfig) configuration class: [RwkvForCausalLM](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvForCausalLM) (RWKV model)
  + [SplinterConfig](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterConfig) configuration class: [SplinterForPreTraining](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterForPreTraining) (Splinter model)
  + [SqueezeBertConfig](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig) configuration class: [SqueezeBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMaskedLM) (SqueezeBERT model)
  + [SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig) configuration class: [SwitchTransformersForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersForConditionalGeneration) (SwitchTransformers model)
  + [T5Config](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config) configuration class: [T5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForConditionalGeneration) (T5 model)
  + [T5GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig) configuration class: [T5GemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForConditionalGeneration) (T5Gemma model)
  + [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) configuration class: [TapasForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForMaskedLM) (TAPAS model)
  + [TransfoXLConfig](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLConfig) configuration class: [TransfoXLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLLMHeadModel) (Transformer-XL model)
  + [TvltConfig](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltConfig) configuration class: [TvltForPreTraining](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltForPreTraining) (TVLT model)
  + [UniSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechConfig) configuration class: [UniSpeechForPreTraining](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechForPreTraining) (UniSpeech model)
  + [UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig) configuration class: [UniSpeechSatForPreTraining](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForPreTraining) (UniSpeechSat model)
  + [ViTMAEConfig](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEConfig) configuration class: [ViTMAEForPreTraining](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining) (ViTMAE model)
  + [VideoLlavaConfig](/docs/transformers/v4.56.2/en/model_doc/video_llava#transformers.VideoLlavaConfig) configuration class: [VideoLlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/video_llava#transformers.VideoLlavaForConditionalGeneration) (VideoLlava model)
  + [VideoMAEConfig](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEConfig) configuration class: [VideoMAEForPreTraining](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEForPreTraining) (VideoMAE model)
  + [VipLlavaConfig](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaConfig) configuration class: [VipLlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaForConditionalGeneration) (VipLlava model)
  + [VisualBertConfig](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig) configuration class: [VisualBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForPreTraining) (VisualBERT model)
  + [VoxtralConfig](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralConfig) configuration class: [VoxtralForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralForConditionalGeneration) (Voxtral model)
  + [Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) configuration class: [Wav2Vec2ForPreTraining](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining) (Wav2Vec2 model)
  + [Wav2Vec2ConformerConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig) configuration class: [Wav2Vec2ConformerForPreTraining](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForPreTraining) (Wav2Vec2-Conformer model)
  + [XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig) configuration class: [XLMWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMWithLMHeadModel) (XLM model)
  + [XLMRobertaConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig) configuration class: [XLMRobertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMaskedLM) (XLM-RoBERTa model)
  + [XLMRobertaXLConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig) configuration class: [XLMRobertaXLForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMaskedLM) (XLM-RoBERTa-XL model)
  + [XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig) configuration class: [XLNetLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel) (XLNet model)
  + [XmodConfig](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig) configuration class: [XmodForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMaskedLM) (X-MOD model)
  + [xLSTMConfig](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMConfig) configuration class: [xLSTMForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMForCausalLM) (xLSTM model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a pretraining head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForPreTraining

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForPreTraining.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a pretraining head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **albert** — `AlbertForPreTraining` (ALBERT model)
* **bart** — [BartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForConditionalGeneration) (BART model)
* **bert** — [BertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForPreTraining) (BERT model)
* **big\_bird** — [BigBirdForPreTraining](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForPreTraining) (BigBird model)
* **bloom** — [BloomForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForCausalLM) (BLOOM model)
* **camembert** — [CamembertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMaskedLM) (CamemBERT model)
* **colpali** — [ColPaliForRetrieval](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliForRetrieval) (ColPali model)
* **colqwen2** — [ColQwen2ForRetrieval](/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2ForRetrieval) (ColQwen2 model)
* **ctrl** — [CTRLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLLMHeadModel) (CTRL model)
* **data2vec-text** — [Data2VecTextForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMaskedLM) (Data2VecText model)
* **deberta** — [DebertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForMaskedLM) (DeBERTa model)
* **deberta-v2** — [DebertaV2ForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMaskedLM) (DeBERTa-v2 model)
* **distilbert** — [DistilBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForMaskedLM) (DistilBERT model)
* **electra** — [ElectraForPreTraining](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForPreTraining) (ELECTRA model)
* **ernie** — [ErnieForPreTraining](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForPreTraining) (ERNIE model)
* **evolla** — [EvollaForProteinText2Text](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaForProteinText2Text) (Evolla model)
* **exaone4** — [Exaone4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForCausalLM) (EXAONE-4.0 model)
* **falcon\_mamba** — [FalconMambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaForCausalLM) (FalconMamba model)
* **flaubert** — [FlaubertWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertWithLMHeadModel) (FlauBERT model)
* **flava** — [FlavaForPreTraining](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaForPreTraining) (FLAVA model)
* **florence2** — [Florence2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2ForConditionalGeneration) (Florence2 model)
* **fnet** — [FNetForPreTraining](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForPreTraining) (FNet model)
* **fsmt** — [FSMTForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTForConditionalGeneration) (FairSeq Machine-Translation model)
* **funnel** — [FunnelForPreTraining](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForPreTraining) (Funnel Transformer model)
* **gemma3** — [Gemma3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration) (Gemma3ForConditionalGeneration model)
* **gpt-sw3** — [GPT2LMHeadModel](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2LMHeadModel) (GPT-Sw3 model)
* **gpt2** — [GPT2LMHeadModel](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2LMHeadModel) (OpenAI GPT-2 model)
* **gpt\_bigcode** — [GPTBigCodeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeForCausalLM) (GPTBigCode model)
* **gptsan-japanese** — [GPTSanJapaneseForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseForConditionalGeneration) (GPTSAN-japanese model)
* **hiera** — [HieraForPreTraining](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraForPreTraining) (Hiera model)
* **ibert** — [IBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForMaskedLM) (I-BERT model)
* **idefics** — [IdeficsForVisionText2Text](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsForVisionText2Text) (IDEFICS model)
* **idefics2** — [Idefics2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ForConditionalGeneration) (Idefics2 model)
* **idefics3** — [Idefics3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3ForConditionalGeneration) (Idefics3 model)
* **janus** — [JanusForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusForConditionalGeneration) (Janus model)
* **layoutlm** — [LayoutLMForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForMaskedLM) (LayoutLM model)
* **llava** — [LlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaForConditionalGeneration) (LLaVa model)
* **llava\_next** — [LlavaNextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextForConditionalGeneration) (LLaVA-NeXT model)
* **llava\_next\_video** — [LlavaNextVideoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoForConditionalGeneration) (LLaVa-NeXT-Video model)
* **llava\_onevision** — [LlavaOnevisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionForConditionalGeneration) (LLaVA-Onevision model)
* **longformer** — [LongformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMaskedLM) (Longformer model)
* **luke** — [LukeForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMaskedLM) (LUKE model)
* **lxmert** — [LxmertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForPreTraining) (LXMERT model)
* **mamba** — [MambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaForCausalLM) (Mamba model)
* **mamba2** — [Mamba2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2ForCausalLM) (mamba2 model)
* **mega** — [MegaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMaskedLM) (MEGA model)
* **megatron-bert** — [MegatronBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForPreTraining) (Megatron-BERT model)
* **mistral3** — [Mistral3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration) (Mistral3 model)
* **mllama** — [MllamaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForConditionalGeneration) (Mllama model)
* **mobilebert** — [MobileBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForPreTraining) (MobileBERT model)
* **mpnet** — [MPNetForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForMaskedLM) (MPNet model)
* **mpt** — [MptForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForCausalLM) (MPT model)
* **mra** — [MraForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForMaskedLM) (MRA model)
* **mvp** — [MvpForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForConditionalGeneration) (MVP model)
* **nezha** — [NezhaForPreTraining](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForPreTraining) (Nezha model)
* **nllb-moe** — [NllbMoeForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeForConditionalGeneration) (NLLB-MOE model)
* **openai-gpt** — [OpenAIGPTLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTLMHeadModel) (OpenAI GPT model)
* **paligemma** — [PaliGemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration) (PaliGemma model)
* **qwen2\_audio** — [Qwen2AudioForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioForConditionalGeneration) (Qwen2Audio model)
* **retribert** — [RetriBertModel](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertModel) (RetriBERT model)
* **roberta** — [RobertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMaskedLM) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaPreLayerNormForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMaskedLM) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForPreTraining) (RoCBert model)
* **rwkv** — [RwkvForCausalLM](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvForCausalLM) (RWKV model)
* **splinter** — [SplinterForPreTraining](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterForPreTraining) (Splinter model)
* **squeezebert** — [SqueezeBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMaskedLM) (SqueezeBERT model)
* **switch\_transformers** — [SwitchTransformersForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersForConditionalGeneration) (SwitchTransformers model)
* **t5** — [T5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForConditionalGeneration) (T5 model)
* **t5gemma** — [T5GemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForConditionalGeneration) (T5Gemma model)
* **tapas** — [TapasForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForMaskedLM) (TAPAS model)
* **transfo-xl** — [TransfoXLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLLMHeadModel) (Transformer-XL model)
* **tvlt** — [TvltForPreTraining](/docs/transformers/v4.56.2/en/model_doc/tvlt#transformers.TvltForPreTraining) (TVLT model)
* **unispeech** — [UniSpeechForPreTraining](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechForPreTraining) (UniSpeech model)
* **unispeech-sat** — [UniSpeechSatForPreTraining](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForPreTraining) (UniSpeechSat model)
* **video\_llava** — [VideoLlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/video_llava#transformers.VideoLlavaForConditionalGeneration) (VideoLlava model)
* **videomae** — [VideoMAEForPreTraining](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEForPreTraining) (VideoMAE model)
* **vipllava** — [VipLlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaForConditionalGeneration) (VipLlava model)
* **visual\_bert** — [VisualBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForPreTraining) (VisualBERT model)
* **vit\_mae** — [ViTMAEForPreTraining](/docs/transformers/v4.56.2/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining) (ViTMAE model)
* **voxtral** — [VoxtralForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralForConditionalGeneration) (Voxtral model)
* **wav2vec2** — [Wav2Vec2ForPreTraining](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining) (Wav2Vec2 model)
* **wav2vec2-conformer** — [Wav2Vec2ConformerForPreTraining](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForPreTraining) (Wav2Vec2-Conformer model)
* **xlm** — [XLMWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMWithLMHeadModel) (XLM model)
* **xlm-roberta** — [XLMRobertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMaskedLM) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaXLForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMaskedLM) (XLM-RoBERTa-XL model)
* **xlnet** — [XLNetLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel) (XLNet model)
* **xlstm** — [xLSTMForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMForCausalLM) (xLSTM model)
* **xmod** — [XmodForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMaskedLM) (X-MOD model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForPreTraining

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForPreTraining.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForPreTraining.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForPreTraining.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

## Natural Language Processing

The following auto classes are available for the following natural language processing tasks.

### AutoModelForCausalLM

### class transformers.AutoModelForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1920)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a causal language modeling head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [ApertusConfig](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusConfig) configuration class: [ApertusForCausalLM](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusForCausalLM) (Apertus model)
  + [ArceeConfig](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeConfig) configuration class: [ArceeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForCausalLM) (Arcee model)
  + [AriaTextConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextConfig) configuration class: [AriaTextForCausalLM](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextForCausalLM) (AriaText model)
  + [BambaConfig](/docs/transformers/v4.56.2/en/model_doc/bamba#transformers.BambaConfig) configuration class: [BambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bamba#transformers.BambaForCausalLM) (Bamba model)
  + [BartConfig](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartConfig) configuration class: [BartForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForCausalLM) (BART model)
  + [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) configuration class: [BertLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertLMHeadModel) (BERT model)
  + [BertGenerationConfig](/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationConfig) configuration class: [BertGenerationDecoder](/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationDecoder) (Bert Generation model)
  + [BigBirdConfig](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig) configuration class: [BigBirdForCausalLM](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForCausalLM) (BigBird model)
  + [BigBirdPegasusConfig](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig) configuration class: [BigBirdPegasusForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForCausalLM) (BigBird-Pegasus model)
  + [BioGptConfig](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig) configuration class: [BioGptForCausalLM](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForCausalLM) (BioGpt model)
  + [BitNetConfig](/docs/transformers/v4.56.2/en/model_doc/bitnet#transformers.BitNetConfig) configuration class: [BitNetForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bitnet#transformers.BitNetForCausalLM) (BitNet model)
  + [BlenderbotConfig](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotConfig) configuration class: [BlenderbotForCausalLM](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotForCausalLM) (Blenderbot model)
  + [BlenderbotSmallConfig](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig) configuration class: [BlenderbotSmallForCausalLM](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForCausalLM) (BlenderbotSmall model)
  + [BloomConfig](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig) configuration class: [BloomForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForCausalLM) (BLOOM model)
  + [CTRLConfig](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig) configuration class: [CTRLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLLMHeadModel) (CTRL model)
  + [CamembertConfig](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig) configuration class: [CamembertForCausalLM](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForCausalLM) (CamemBERT model)
  + [CodeGenConfig](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenConfig) configuration class: [CodeGenForCausalLM](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenForCausalLM) (CodeGen model)
  + [Cohere2Config](/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2Config) configuration class: [Cohere2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2ForCausalLM) (Cohere2 model)
  + [CohereConfig](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereConfig) configuration class: [CohereForCausalLM](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereForCausalLM) (Cohere model)
  + [CpmAntConfig](/docs/transformers/v4.56.2/en/model_doc/cpmant#transformers.CpmAntConfig) configuration class: [CpmAntForCausalLM](/docs/transformers/v4.56.2/en/model_doc/cpmant#transformers.CpmAntForCausalLM) (CPM-Ant model)
  + [Data2VecTextConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig) configuration class: [Data2VecTextForCausalLM](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForCausalLM) (Data2VecText model)
  + [DbrxConfig](/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxConfig) configuration class: [DbrxForCausalLM](/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxForCausalLM) (DBRX model)
  + [DeepseekV2Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config) configuration class: [DeepseekV2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2ForCausalLM) (DeepSeek-V2 model)
  + [DeepseekV3Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Config) configuration class: [DeepseekV3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3ForCausalLM) (DeepSeek-V3 model)
  + [DiffLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaConfig) configuration class: [DiffLlamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForCausalLM) (DiffLlama model)
  + [DogeConfig](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeConfig) configuration class: [DogeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeForCausalLM) (Doge model)
  + [Dots1Config](/docs/transformers/v4.56.2/en/model_doc/dots1#transformers.Dots1Config) configuration class: [Dots1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/dots1#transformers.Dots1ForCausalLM) (dots1 model)
  + [ElectraConfig](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig) configuration class: [ElectraForCausalLM](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForCausalLM) (ELECTRA model)
  + [Emu3Config](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Config) configuration class: [Emu3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ForCausalLM) (Emu3 model)
  + [Ernie4\_5Config](/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Config) configuration class: [Ernie4\_5ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5ForCausalLM) (Ernie4\_5 model)
  + [Ernie4\_5\_MoeConfig](/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeConfig) configuration class: [Ernie4\_5\_MoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeForCausalLM) (Ernie4\_5\_MoE model)
  + [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieForCausalLM](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForCausalLM) (ERNIE model)
  + [Exaone4Config](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config) configuration class: [Exaone4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForCausalLM) (EXAONE-4.0 model)
  + [FalconConfig](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig) configuration class: [FalconForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForCausalLM) (Falcon model)
  + [FalconH1Config](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1Config) configuration class: [FalconH1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1ForCausalLM) (FalconH1 model)
  + [FalconMambaConfig](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaConfig) configuration class: [FalconMambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaForCausalLM) (FalconMamba model)
  + [FuyuConfig](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuConfig) configuration class: [FuyuForCausalLM](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuForCausalLM) (Fuyu model)
  + [GPT2Config](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config) configuration class: [GPT2LMHeadModel](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2LMHeadModel) (OpenAI GPT-2 model)
  + [GPTBigCodeConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeConfig) configuration class: [GPTBigCodeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeForCausalLM) (GPTBigCode model)
  + [GPTJConfig](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJConfig) configuration class: [GPTJForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJForCausalLM) (GPT-J model)
  + [GPTNeoConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig) configuration class: [GPTNeoForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForCausalLM) (GPT Neo model)
  + [GPTNeoXConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXConfig) configuration class: [GPTNeoXForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXForCausalLM) (GPT NeoX model)
  + [GPTNeoXJapaneseConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseConfig) configuration class: [GPTNeoXJapaneseForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseForCausalLM) (GPT NeoX Japanese model)
  + [Gemma2Config](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Config) configuration class: [Gemma2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2ForCausalLM) (Gemma2 model)
  + [Gemma3Config](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config) configuration class: [Gemma3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration) (Gemma3ForConditionalGeneration model)
  + [Gemma3TextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextConfig) configuration class: [Gemma3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForCausalLM) (Gemma3ForCausalLM model)
  + [Gemma3nConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nConfig) configuration class: [Gemma3nForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForConditionalGeneration) (Gemma3nForConditionalGeneration model)
  + [Gemma3nTextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextConfig) configuration class: [Gemma3nForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForCausalLM) (Gemma3nForCausalLM model)
  + [GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaConfig) configuration class: [GemmaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaForCausalLM) (Gemma model)
  + [GitConfig](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitConfig) configuration class: [GitForCausalLM](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitForCausalLM) (GIT model)
  + [Glm4Config](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Config) configuration class: [Glm4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4ForCausalLM) (GLM4 model)
  + [Glm4MoeConfig](/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeConfig) configuration class: [Glm4MoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeForCausalLM) (Glm4MoE model)
  + [GlmConfig](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmConfig) configuration class: [GlmForCausalLM](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmForCausalLM) (GLM model)
  + [GotOcr2Config](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Config) configuration class: [GotOcr2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ForConditionalGeneration) (GOT-OCR2 model)
  + [GptOssConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssConfig) configuration class: [GptOssForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssForCausalLM) (GptOss model)
  + [GraniteConfig](/docs/transformers/v4.56.2/en/model_doc/granite#transformers.GraniteConfig) configuration class: [GraniteForCausalLM](/docs/transformers/v4.56.2/en/model_doc/granite#transformers.GraniteForCausalLM) (Granite model)
  + [GraniteMoeConfig](/docs/transformers/v4.56.2/en/model_doc/granitemoe#transformers.GraniteMoeConfig) configuration class: [GraniteMoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/granitemoe#transformers.GraniteMoeForCausalLM) (GraniteMoeMoe model)
  + [GraniteMoeHybridConfig](/docs/transformers/v4.56.2/en/model_doc/granitemoehybrid#transformers.GraniteMoeHybridConfig) configuration class: [GraniteMoeHybridForCausalLM](/docs/transformers/v4.56.2/en/model_doc/granitemoehybrid#transformers.GraniteMoeHybridForCausalLM) (GraniteMoeHybrid model)
  + [GraniteMoeSharedConfig](/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedConfig) configuration class: [GraniteMoeSharedForCausalLM](/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedForCausalLM) (GraniteMoeSharedMoe model)
  + [HeliumConfig](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumConfig) configuration class: [HeliumForCausalLM](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumForCausalLM) (Helium model)
  + [HunYuanDenseV1Config](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Config) configuration class: [HunYuanDenseV1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1ForCausalLM) (HunYuanDenseV1 model)
  + [HunYuanMoEV1Config](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Config) configuration class: [HunYuanMoEV1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1ForCausalLM) (HunYuanMoeV1 model)
  + [JambaConfig](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaConfig) configuration class: [JambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaForCausalLM) (Jamba model)
  + [JetMoeConfig](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeConfig) configuration class: [JetMoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeForCausalLM) (JetMoe model)
  + [Lfm2Config](/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Config) configuration class: [Lfm2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2ForCausalLM) (Lfm2 model)
  + [Llama4Config](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Config) configuration class: [Llama4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForCausalLM) (Llama4 model)
  + [Llama4TextConfig](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4TextConfig) configuration class: [Llama4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForCausalLM) (Llama4ForCausalLM model)
  + [LlamaConfig](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig) configuration class: [LlamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForCausalLM) (LLaMA model)
  + [MBartConfig](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig) configuration class: [MBartForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForCausalLM) (mBART model)
  + [Mamba2Config](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2Config) configuration class: [Mamba2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2ForCausalLM) (mamba2 model)
  + [MambaConfig](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaConfig) configuration class: [MambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaForCausalLM) (Mamba model)
  + [MarianConfig](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianConfig) configuration class: [MarianForCausalLM](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianForCausalLM) (Marian model)
  + [MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig) configuration class: [MegaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForCausalLM) (MEGA model)
  + [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertForCausalLM](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForCausalLM) (Megatron-BERT model)
  + [MiniMaxConfig](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxConfig) configuration class: [MiniMaxForCausalLM](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxForCausalLM) (MiniMax model)
  + [MistralConfig](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig) configuration class: [MistralForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForCausalLM) (Mistral model)
  + [MixtralConfig](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralConfig) configuration class: [MixtralForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralForCausalLM) (Mixtral model)
  + [MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig) configuration class: [MllamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM) (Mllama model)
  + [ModernBertDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderConfig) configuration class: [ModernBertDecoderForCausalLM](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderForCausalLM) (ModernBertDecoder model)
  + [MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig) configuration class: [MoshiForCausalLM](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiForCausalLM) (Moshi model)
  + [MptConfig](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptConfig) configuration class: [MptForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForCausalLM) (MPT model)
  + [MusicgenConfig](/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenConfig) configuration class: [MusicgenForCausalLM](/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenForCausalLM) (MusicGen model)
  + [MusicgenMelodyConfig](/docs/transformers/v4.56.2/en/model_doc/musicgen_melody#transformers.MusicgenMelodyConfig) configuration class: [MusicgenMelodyForCausalLM](/docs/transformers/v4.56.2/en/model_doc/musicgen_melody#transformers.MusicgenMelodyForCausalLM) (MusicGen Melody model)
  + [MvpConfig](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig) configuration class: [MvpForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForCausalLM) (MVP model)
  + [NemotronConfig](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronConfig) configuration class: [NemotronForCausalLM](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronForCausalLM) (Nemotron model)
  + [OPTConfig](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig) configuration class: [OPTForCausalLM](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForCausalLM) (OPT model)
  + [Olmo2Config](/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Config) configuration class: [Olmo2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2ForCausalLM) (OLMo2 model)
  + [OlmoConfig](/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoConfig) configuration class: [OlmoForCausalLM](/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoForCausalLM) (OLMo model)
  + [OlmoeConfig](/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeConfig) configuration class: [OlmoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeForCausalLM) (OLMoE model)
  + [OpenAIGPTConfig](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig) configuration class: [OpenAIGPTLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTLMHeadModel) (OpenAI GPT model)
  + [OpenLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaConfig) configuration class: [OpenLlamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaForCausalLM) (OpenLlama model)
  + [PLBartConfig](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig) configuration class: [PLBartForCausalLM](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForCausalLM) (PLBart model)
  + [PegasusConfig](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusConfig) configuration class: [PegasusForCausalLM](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusForCausalLM) (Pegasus model)
  + [PersimmonConfig](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonConfig) configuration class: [PersimmonForCausalLM](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonForCausalLM) (Persimmon model)
  + [Phi3Config](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Config) configuration class: [Phi3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3ForCausalLM) (Phi3 model)
  + [Phi4MultimodalConfig](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalConfig) configuration class: [Phi4MultimodalForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalForCausalLM) (Phi4Multimodal model)
  + [PhiConfig](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiConfig) configuration class: [PhiForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiForCausalLM) (Phi model)
  + [PhimoeConfig](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeConfig) configuration class: [PhimoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeForCausalLM) (Phimoe model)
  + [ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig) configuration class: [ProphetNetForCausalLM](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForCausalLM) (ProphetNet model)
  + [QDQBertConfig](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig) configuration class: [QDQBertLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertLMHeadModel) (QDQBert model)
  + [Qwen2Config](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Config) configuration class: [Qwen2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForCausalLM) (Qwen2 model)
  + [Qwen2MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeConfig) configuration class: [Qwen2MoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeForCausalLM) (Qwen2MoE model)
  + [Qwen3Config](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Config) configuration class: [Qwen3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForCausalLM) (Qwen3 model)
  + [Qwen3MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeConfig) configuration class: [Qwen3MoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForCausalLM) (Qwen3MoE model)
  + [RecurrentGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaConfig) configuration class: [RecurrentGemmaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaForCausalLM) (RecurrentGemma model)
  + [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig) configuration class: [ReformerModelWithLMHead](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModelWithLMHead) (Reformer model)
  + [RemBertConfig](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertConfig) configuration class: [RemBertForCausalLM](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForCausalLM) (RemBERT model)
  + [RoCBertConfig](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig) configuration class: [RoCBertForCausalLM](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForCausalLM) (RoCBert model)
  + [RoFormerConfig](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig) configuration class: [RoFormerForCausalLM](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForCausalLM) (RoFormer model)
  + [RobertaConfig](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig) configuration class: [RobertaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForCausalLM) (RoBERTa model)
  + [RobertaPreLayerNormConfig](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig) configuration class: [RobertaPreLayerNormForCausalLM](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForCausalLM) (RoBERTa-PreLayerNorm model)
  + [RwkvConfig](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvConfig) configuration class: [RwkvForCausalLM](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvForCausalLM) (RWKV model)
  + [SeedOssConfig](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssConfig) configuration class: [SeedOssForCausalLM](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForCausalLM) (SeedOss model)
  + [SmolLM3Config](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Config) configuration class: [SmolLM3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForCausalLM) (SmolLM3 model)
  + [Speech2Text2Config](/docs/transformers/v4.56.2/en/model_doc/speech_to_text_2#transformers.Speech2Text2Config) configuration class: [Speech2Text2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/speech_to_text_2#transformers.Speech2Text2ForCausalLM) (Speech2Text2 model)
  + [StableLmConfig](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmConfig) configuration class: [StableLmForCausalLM](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmForCausalLM) (StableLm model)
  + [Starcoder2Config](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2Config) configuration class: [Starcoder2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2ForCausalLM) (Starcoder2 model)
  + [TrOCRConfig](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRConfig) configuration class: [TrOCRForCausalLM](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRForCausalLM) (TrOCR model)
  + [TransfoXLConfig](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLConfig) configuration class: [TransfoXLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLLMHeadModel) (Transformer-XL model)
  + [WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig) configuration class: [WhisperForCausalLM](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForCausalLM) (Whisper model)
  + [XGLMConfig](/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMConfig) configuration class: [XGLMForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMForCausalLM) (XGLM model)
  + [XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig) configuration class: [XLMWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMWithLMHeadModel) (XLM model)
  + [XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig) configuration class: [XLMProphetNetForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForCausalLM) (XLM-ProphetNet model)
  + [XLMRobertaConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig) configuration class: [XLMRobertaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForCausalLM) (XLM-RoBERTa model)
  + [XLMRobertaXLConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig) configuration class: [XLMRobertaXLForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForCausalLM) (XLM-RoBERTa-XL model)
  + [XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig) configuration class: [XLNetLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel) (XLNet model)
  + [XmodConfig](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig) configuration class: [XmodForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForCausalLM) (X-MOD model)
  + [Zamba2Config](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2Config) configuration class: [Zamba2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2ForCausalLM) (Zamba2 model)
  + [ZambaConfig](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaConfig) configuration class: [ZambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaForCausalLM) (Zamba model)
  + [xLSTMConfig](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMConfig) configuration class: [xLSTMForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMForCausalLM) (xLSTM model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a causal language modeling head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForCausalLM

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForCausalLM.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a causal language modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **apertus** — [ApertusForCausalLM](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusForCausalLM) (Apertus model)
* **arcee** — [ArceeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForCausalLM) (Arcee model)
* **aria\_text** — [AriaTextForCausalLM](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextForCausalLM) (AriaText model)
* **bamba** — [BambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bamba#transformers.BambaForCausalLM) (Bamba model)
* **bart** — [BartForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForCausalLM) (BART model)
* **bert** — [BertLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertLMHeadModel) (BERT model)
* **bert-generation** — [BertGenerationDecoder](/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationDecoder) (Bert Generation model)
* **big\_bird** — [BigBirdForCausalLM](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForCausalLM) (BigBird model)
* **bigbird\_pegasus** — [BigBirdPegasusForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForCausalLM) (BigBird-Pegasus model)
* **biogpt** — [BioGptForCausalLM](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForCausalLM) (BioGpt model)
* **bitnet** — [BitNetForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bitnet#transformers.BitNetForCausalLM) (BitNet model)
* **blenderbot** — [BlenderbotForCausalLM](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotForCausalLM) (Blenderbot model)
* **blenderbot-small** — [BlenderbotSmallForCausalLM](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForCausalLM) (BlenderbotSmall model)
* **bloom** — [BloomForCausalLM](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForCausalLM) (BLOOM model)
* **camembert** — [CamembertForCausalLM](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForCausalLM) (CamemBERT model)
* **code\_llama** — [LlamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForCausalLM) (CodeLlama model)
* **codegen** — [CodeGenForCausalLM](/docs/transformers/v4.56.2/en/model_doc/codegen#transformers.CodeGenForCausalLM) (CodeGen model)
* **cohere** — [CohereForCausalLM](/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereForCausalLM) (Cohere model)
* **cohere2** — [Cohere2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2ForCausalLM) (Cohere2 model)
* **cpmant** — [CpmAntForCausalLM](/docs/transformers/v4.56.2/en/model_doc/cpmant#transformers.CpmAntForCausalLM) (CPM-Ant model)
* **ctrl** — [CTRLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLLMHeadModel) (CTRL model)
* **data2vec-text** — [Data2VecTextForCausalLM](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForCausalLM) (Data2VecText model)
* **dbrx** — [DbrxForCausalLM](/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxForCausalLM) (DBRX model)
* **deepseek\_v2** — [DeepseekV2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2ForCausalLM) (DeepSeek-V2 model)
* **deepseek\_v3** — [DeepseekV3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3ForCausalLM) (DeepSeek-V3 model)
* **diffllama** — [DiffLlamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForCausalLM) (DiffLlama model)
* **doge** — [DogeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeForCausalLM) (Doge model)
* **dots1** — [Dots1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/dots1#transformers.Dots1ForCausalLM) (dots1 model)
* **electra** — [ElectraForCausalLM](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForCausalLM) (ELECTRA model)
* **emu3** — [Emu3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ForCausalLM) (Emu3 model)
* **ernie** — [ErnieForCausalLM](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForCausalLM) (ERNIE model)
* **ernie4\_5** — [Ernie4\_5ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5ForCausalLM) (Ernie4\_5 model)
* **ernie4\_5\_moe** — [Ernie4\_5\_MoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeForCausalLM) (Ernie4\_5\_MoE model)
* **exaone4** — [Exaone4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForCausalLM) (EXAONE-4.0 model)
* **falcon** — [FalconForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForCausalLM) (Falcon model)
* **falcon\_h1** — [FalconH1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1ForCausalLM) (FalconH1 model)
* **falcon\_mamba** — [FalconMambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_mamba#transformers.FalconMambaForCausalLM) (FalconMamba model)
* **fuyu** — [FuyuForCausalLM](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuForCausalLM) (Fuyu model)
* **gemma** — [GemmaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaForCausalLM) (Gemma model)
* **gemma2** — [Gemma2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2ForCausalLM) (Gemma2 model)
* **gemma3** — [Gemma3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration) (Gemma3ForConditionalGeneration model)
* **gemma3\_text** — [Gemma3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForCausalLM) (Gemma3ForCausalLM model)
* **gemma3n** — [Gemma3nForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForConditionalGeneration) (Gemma3nForConditionalGeneration model)
* **gemma3n\_text** — [Gemma3nForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForCausalLM) (Gemma3nForCausalLM model)
* **git** — [GitForCausalLM](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitForCausalLM) (GIT model)
* **glm** — [GlmForCausalLM](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmForCausalLM) (GLM model)
* **glm4** — [Glm4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4ForCausalLM) (GLM4 model)
* **glm4\_moe** — [Glm4MoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeForCausalLM) (Glm4MoE model)
* **got\_ocr2** — [GotOcr2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ForConditionalGeneration) (GOT-OCR2 model)
* **gpt-sw3** — [GPT2LMHeadModel](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2LMHeadModel) (GPT-Sw3 model)
* **gpt2** — [GPT2LMHeadModel](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2LMHeadModel) (OpenAI GPT-2 model)
* **gpt\_bigcode** — [GPTBigCodeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeForCausalLM) (GPTBigCode model)
* **gpt\_neo** — [GPTNeoForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForCausalLM) (GPT Neo model)
* **gpt\_neox** — [GPTNeoXForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXForCausalLM) (GPT NeoX model)
* **gpt\_neox\_japanese** — [GPTNeoXJapaneseForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseForCausalLM) (GPT NeoX Japanese model)
* **gpt\_oss** — [GptOssForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssForCausalLM) (GptOss model)
* **gptj** — [GPTJForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJForCausalLM) (GPT-J model)
* **granite** — [GraniteForCausalLM](/docs/transformers/v4.56.2/en/model_doc/granite#transformers.GraniteForCausalLM) (Granite model)
* **granitemoe** — [GraniteMoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/granitemoe#transformers.GraniteMoeForCausalLM) (GraniteMoeMoe model)
* **granitemoehybrid** — [GraniteMoeHybridForCausalLM](/docs/transformers/v4.56.2/en/model_doc/granitemoehybrid#transformers.GraniteMoeHybridForCausalLM) (GraniteMoeHybrid model)
* **granitemoeshared** — [GraniteMoeSharedForCausalLM](/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedForCausalLM) (GraniteMoeSharedMoe model)
* **helium** — [HeliumForCausalLM](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumForCausalLM) (Helium model)
* **hunyuan\_v1\_dense** — [HunYuanDenseV1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1ForCausalLM) (HunYuanDenseV1 model)
* **hunyuan\_v1\_moe** — [HunYuanMoEV1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1ForCausalLM) (HunYuanMoeV1 model)
* **jamba** — [JambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaForCausalLM) (Jamba model)
* **jetmoe** — [JetMoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeForCausalLM) (JetMoe model)
* **lfm2** — [Lfm2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2ForCausalLM) (Lfm2 model)
* **llama** — [LlamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForCausalLM) (LLaMA model)
* **llama4** — [Llama4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForCausalLM) (Llama4 model)
* **llama4\_text** — [Llama4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForCausalLM) (Llama4ForCausalLM model)
* **mamba** — [MambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba#transformers.MambaForCausalLM) (Mamba model)
* **mamba2** — [Mamba2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mamba2#transformers.Mamba2ForCausalLM) (mamba2 model)
* **marian** — [MarianForCausalLM](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianForCausalLM) (Marian model)
* **mbart** — [MBartForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForCausalLM) (mBART model)
* **mega** — [MegaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForCausalLM) (MEGA model)
* **megatron-bert** — [MegatronBertForCausalLM](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForCausalLM) (Megatron-BERT model)
* **minimax** — [MiniMaxForCausalLM](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxForCausalLM) (MiniMax model)
* **mistral** — [MistralForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForCausalLM) (Mistral model)
* **mixtral** — [MixtralForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralForCausalLM) (Mixtral model)
* **mllama** — [MllamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM) (Mllama model)
* **modernbert-decoder** — [ModernBertDecoderForCausalLM](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderForCausalLM) (ModernBertDecoder model)
* **moshi** — [MoshiForCausalLM](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiForCausalLM) (Moshi model)
* **mpt** — [MptForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForCausalLM) (MPT model)
* **musicgen** — [MusicgenForCausalLM](/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenForCausalLM) (MusicGen model)
* **musicgen\_melody** — [MusicgenMelodyForCausalLM](/docs/transformers/v4.56.2/en/model_doc/musicgen_melody#transformers.MusicgenMelodyForCausalLM) (MusicGen Melody model)
* **mvp** — [MvpForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForCausalLM) (MVP model)
* **nemotron** — [NemotronForCausalLM](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronForCausalLM) (Nemotron model)
* **olmo** — [OlmoForCausalLM](/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoForCausalLM) (OLMo model)
* **olmo2** — [Olmo2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2ForCausalLM) (OLMo2 model)
* **olmoe** — [OlmoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeForCausalLM) (OLMoE model)
* **open-llama** — [OpenLlamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaForCausalLM) (OpenLlama model)
* **openai-gpt** — [OpenAIGPTLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTLMHeadModel) (OpenAI GPT model)
* **opt** — [OPTForCausalLM](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForCausalLM) (OPT model)
* **pegasus** — [PegasusForCausalLM](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusForCausalLM) (Pegasus model)
* **persimmon** — [PersimmonForCausalLM](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonForCausalLM) (Persimmon model)
* **phi** — [PhiForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiForCausalLM) (Phi model)
* **phi3** — [Phi3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3ForCausalLM) (Phi3 model)
* **phi4\_multimodal** — [Phi4MultimodalForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalForCausalLM) (Phi4Multimodal model)
* **phimoe** — [PhimoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeForCausalLM) (Phimoe model)
* **plbart** — [PLBartForCausalLM](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForCausalLM) (PLBart model)
* **prophetnet** — [ProphetNetForCausalLM](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForCausalLM) (ProphetNet model)
* **qdqbert** — [QDQBertLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertLMHeadModel) (QDQBert model)
* **qwen2** — [Qwen2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForCausalLM) (Qwen2 model)
* **qwen2\_moe** — [Qwen2MoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeForCausalLM) (Qwen2MoE model)
* **qwen3** — [Qwen3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForCausalLM) (Qwen3 model)
* **qwen3\_moe** — [Qwen3MoeForCausalLM](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForCausalLM) (Qwen3MoE model)
* **recurrent\_gemma** — [RecurrentGemmaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaForCausalLM) (RecurrentGemma model)
* **reformer** — [ReformerModelWithLMHead](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModelWithLMHead) (Reformer model)
* **rembert** — [RemBertForCausalLM](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForCausalLM) (RemBERT model)
* **roberta** — [RobertaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForCausalLM) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaPreLayerNormForCausalLM](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForCausalLM) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertForCausalLM](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForCausalLM) (RoCBert model)
* **roformer** — [RoFormerForCausalLM](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForCausalLM) (RoFormer model)
* **rwkv** — [RwkvForCausalLM](/docs/transformers/v4.56.2/en/model_doc/rwkv#transformers.RwkvForCausalLM) (RWKV model)
* **seed\_oss** — [SeedOssForCausalLM](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForCausalLM) (SeedOss model)
* **smollm3** — [SmolLM3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForCausalLM) (SmolLM3 model)
* **speech\_to\_text\_2** — [Speech2Text2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/speech_to_text_2#transformers.Speech2Text2ForCausalLM) (Speech2Text2 model)
* **stablelm** — [StableLmForCausalLM](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmForCausalLM) (StableLm model)
* **starcoder2** — [Starcoder2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2ForCausalLM) (Starcoder2 model)
* **transfo-xl** — [TransfoXLLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLLMHeadModel) (Transformer-XL model)
* **trocr** — [TrOCRForCausalLM](/docs/transformers/v4.56.2/en/model_doc/trocr#transformers.TrOCRForCausalLM) (TrOCR model)
* **whisper** — [WhisperForCausalLM](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForCausalLM) (Whisper model)
* **xglm** — [XGLMForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMForCausalLM) (XGLM model)
* **xlm** — [XLMWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMWithLMHeadModel) (XLM model)
* **xlm-prophetnet** — [XLMProphetNetForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForCausalLM) (XLM-ProphetNet model)
* **xlm-roberta** — [XLMRobertaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForCausalLM) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaXLForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForCausalLM) (XLM-RoBERTa-XL model)
* **xlnet** — [XLNetLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel) (XLNet model)
* **xlstm** — [xLSTMForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMForCausalLM) (xLSTM model)
* **xmod** — [XmodForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForCausalLM) (X-MOD model)
* **zamba** — [ZambaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaForCausalLM) (Zamba model)
* **zamba2** — [Zamba2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2ForCausalLM) (Zamba2 model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForCausalLM

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForCausalLM.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForMaskedLM

### class transformers.AutoModelForMaskedLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1937)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a masked language modeling head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig) configuration class: `AlbertForMaskedLM` (ALBERT model)
  + [BartConfig](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartConfig) configuration class: [BartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForConditionalGeneration) (BART model)
  + [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) configuration class: [BertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForMaskedLM) (BERT model)
  + [BigBirdConfig](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig) configuration class: [BigBirdForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForMaskedLM) (BigBird model)
  + [CamembertConfig](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig) configuration class: [CamembertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMaskedLM) (CamemBERT model)
  + [ConvBertConfig](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertConfig) configuration class: [ConvBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForMaskedLM) (ConvBERT model)
  + [Data2VecTextConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig) configuration class: [Data2VecTextForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMaskedLM) (Data2VecText model)
  + [DebertaConfig](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaConfig) configuration class: [DebertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForMaskedLM) (DeBERTa model)
  + [DebertaV2Config](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config) configuration class: [DebertaV2ForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMaskedLM) (DeBERTa-v2 model)
  + [DistilBertConfig](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertConfig) configuration class: [DistilBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForMaskedLM) (DistilBERT model)
  + [ElectraConfig](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig) configuration class: [ElectraForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForMaskedLM) (ELECTRA model)
  + [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForMaskedLM) (ERNIE model)
  + [EsmConfig](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig) configuration class: [EsmForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForMaskedLM) (ESM model)
  + [FNetConfig](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig) configuration class: [FNetForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForMaskedLM) (FNet model)
  + [FlaubertConfig](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertConfig) configuration class: [FlaubertWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertWithLMHeadModel) (FlauBERT model)
  + [FunnelConfig](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelConfig) configuration class: [FunnelForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForMaskedLM) (Funnel Transformer model)
  + [IBertConfig](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertConfig) configuration class: [IBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForMaskedLM) (I-BERT model)
  + [LayoutLMConfig](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig) configuration class: [LayoutLMForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForMaskedLM) (LayoutLM model)
  + [LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig) configuration class: [LongformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMaskedLM) (Longformer model)
  + [LukeConfig](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig) configuration class: [LukeForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMaskedLM) (LUKE model)
  + [MBartConfig](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig) configuration class: [MBartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForConditionalGeneration) (mBART model)
  + [MPNetConfig](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig) configuration class: [MPNetForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForMaskedLM) (MPNet model)
  + [MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig) configuration class: [MegaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMaskedLM) (MEGA model)
  + [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForMaskedLM) (Megatron-BERT model)
  + [MobileBertConfig](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [MobileBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForMaskedLM) (MobileBERT model)
  + [ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig) configuration class: [ModernBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForMaskedLM) (ModernBERT model)
  + [MraConfig](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraConfig) configuration class: [MraForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForMaskedLM) (MRA model)
  + [MvpConfig](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig) configuration class: [MvpForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForConditionalGeneration) (MVP model)
  + [NezhaConfig](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig) configuration class: [NezhaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForMaskedLM) (Nezha model)
  + [NystromformerConfig](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig) configuration class: [NystromformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForMaskedLM) (Nyströmformer model)
  + [PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig) configuration class: [PerceiverForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForMaskedLM) (Perceiver model)
  + [QDQBertConfig](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig) configuration class: [QDQBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForMaskedLM) (QDQBert model)
  + [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig) configuration class: [ReformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForMaskedLM) (Reformer model)
  + [RemBertConfig](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertConfig) configuration class: [RemBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForMaskedLM) (RemBERT model)
  + [RoCBertConfig](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig) configuration class: [RoCBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForMaskedLM) (RoCBert model)
  + [RoFormerConfig](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig) configuration class: [RoFormerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForMaskedLM) (RoFormer model)
  + [RobertaConfig](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig) configuration class: [RobertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMaskedLM) (RoBERTa model)
  + [RobertaPreLayerNormConfig](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig) configuration class: [RobertaPreLayerNormForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMaskedLM) (RoBERTa-PreLayerNorm model)
  + [SqueezeBertConfig](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig) configuration class: [SqueezeBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMaskedLM) (SqueezeBERT model)
  + [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) configuration class: [TapasForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForMaskedLM) (TAPAS model)
  + [Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) configuration class: `Wav2Vec2ForMaskedLM` (Wav2Vec2 model)
  + [XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig) configuration class: [XLMWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMWithLMHeadModel) (XLM model)
  + [XLMRobertaConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig) configuration class: [XLMRobertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMaskedLM) (XLM-RoBERTa model)
  + [XLMRobertaXLConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig) configuration class: [XLMRobertaXLForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMaskedLM) (XLM-RoBERTa-XL model)
  + [XmodConfig](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig) configuration class: [XmodForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMaskedLM) (X-MOD model)
  + [YosoConfig](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig) configuration class: [YosoForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForMaskedLM) (YOSO model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a masked language modeling head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForMaskedLM

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForMaskedLM.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a masked language modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **albert** — `AlbertForMaskedLM` (ALBERT model)
* **bart** — [BartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForConditionalGeneration) (BART model)
* **bert** — [BertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForMaskedLM) (BERT model)
* **big\_bird** — [BigBirdForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForMaskedLM) (BigBird model)
* **camembert** — [CamembertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMaskedLM) (CamemBERT model)
* **convbert** — [ConvBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForMaskedLM) (ConvBERT model)
* **data2vec-text** — [Data2VecTextForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMaskedLM) (Data2VecText model)
* **deberta** — [DebertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForMaskedLM) (DeBERTa model)
* **deberta-v2** — [DebertaV2ForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMaskedLM) (DeBERTa-v2 model)
* **distilbert** — [DistilBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForMaskedLM) (DistilBERT model)
* **electra** — [ElectraForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForMaskedLM) (ELECTRA model)
* **ernie** — [ErnieForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForMaskedLM) (ERNIE model)
* **esm** — [EsmForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForMaskedLM) (ESM model)
* **flaubert** — [FlaubertWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertWithLMHeadModel) (FlauBERT model)
* **fnet** — [FNetForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForMaskedLM) (FNet model)
* **funnel** — [FunnelForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForMaskedLM) (Funnel Transformer model)
* **ibert** — [IBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForMaskedLM) (I-BERT model)
* **layoutlm** — [LayoutLMForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForMaskedLM) (LayoutLM model)
* **longformer** — [LongformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMaskedLM) (Longformer model)
* **luke** — [LukeForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMaskedLM) (LUKE model)
* **mbart** — [MBartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForConditionalGeneration) (mBART model)
* **mega** — [MegaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMaskedLM) (MEGA model)
* **megatron-bert** — [MegatronBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForMaskedLM) (Megatron-BERT model)
* **mobilebert** — [MobileBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForMaskedLM) (MobileBERT model)
* **modernbert** — [ModernBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForMaskedLM) (ModernBERT model)
* **mpnet** — [MPNetForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForMaskedLM) (MPNet model)
* **mra** — [MraForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForMaskedLM) (MRA model)
* **mvp** — [MvpForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForConditionalGeneration) (MVP model)
* **nezha** — [NezhaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForMaskedLM) (Nezha model)
* **nystromformer** — [NystromformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForMaskedLM) (Nyströmformer model)
* **perceiver** — [PerceiverForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForMaskedLM) (Perceiver model)
* **qdqbert** — [QDQBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForMaskedLM) (QDQBert model)
* **reformer** — [ReformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForMaskedLM) (Reformer model)
* **rembert** — [RemBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForMaskedLM) (RemBERT model)
* **roberta** — [RobertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMaskedLM) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaPreLayerNormForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMaskedLM) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForMaskedLM) (RoCBert model)
* **roformer** — [RoFormerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForMaskedLM) (RoFormer model)
* **squeezebert** — [SqueezeBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMaskedLM) (SqueezeBERT model)
* **tapas** — [TapasForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForMaskedLM) (TAPAS model)
* **wav2vec2** — `Wav2Vec2ForMaskedLM` (Wav2Vec2 model)
* **xlm** — [XLMWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMWithLMHeadModel) (XLM model)
* **xlm-roberta** — [XLMRobertaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMaskedLM) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaXLForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMaskedLM) (XLM-RoBERTa-XL model)
* **xmod** — [XmodForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMaskedLM) (X-MOD model)
* **yoso** — [YosoForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForMaskedLM) (YOSO model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForMaskedLM

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForMaskedLM.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForMaskGeneration

### class transformers.AutoModelForMaskGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1878)

( \*args \*\*kwargs  )

### AutoModelForSeq2SeqLM

### class transformers.AutoModelForSeq2SeqLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1944)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence language modeling head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [BartConfig](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartConfig) configuration class: [BartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForConditionalGeneration) (BART model)
  + [BigBirdPegasusConfig](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig) configuration class: [BigBirdPegasusForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForConditionalGeneration) (BigBird-Pegasus model)
  + [BlenderbotConfig](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotConfig) configuration class: [BlenderbotForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotForConditionalGeneration) (Blenderbot model)
  + [BlenderbotSmallConfig](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig) configuration class: [BlenderbotSmallForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForConditionalGeneration) (BlenderbotSmall model)
  + [EncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) configuration class: [EncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) (Encoder decoder model)
  + [FSMTConfig](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTConfig) configuration class: [FSMTForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTForConditionalGeneration) (FairSeq Machine-Translation model)
  + [GPTSanJapaneseConfig](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseConfig) configuration class: [GPTSanJapaneseForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseForConditionalGeneration) (GPTSAN-japanese model)
  + [GraniteSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechConfig) configuration class: [GraniteSpeechForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechForConditionalGeneration) (GraniteSpeech model)
  + [LEDConfig](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDConfig) configuration class: [LEDForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDForConditionalGeneration) (LED model)
  + [LongT5Config](/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Config) configuration class: [LongT5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5ForConditionalGeneration) (LongT5 model)
  + [M2M100Config](/docs/transformers/v4.56.2/en/model_doc/m2m_100#transformers.M2M100Config) configuration class: [M2M100ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/m2m_100#transformers.M2M100ForConditionalGeneration) (M2M100 model)
  + [MBartConfig](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig) configuration class: [MBartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForConditionalGeneration) (mBART model)
  + [MT5Config](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config) configuration class: [MT5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForConditionalGeneration) (MT5 model)
  + [MarianConfig](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianConfig) configuration class: [MarianMTModel](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianMTModel) (Marian model)
  + [MvpConfig](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig) configuration class: [MvpForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForConditionalGeneration) (MVP model)
  + [NllbMoeConfig](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeConfig) configuration class: [NllbMoeForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeForConditionalGeneration) (NLLB-MOE model)
  + [PLBartConfig](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig) configuration class: [PLBartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForConditionalGeneration) (PLBart model)
  + [PegasusConfig](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusConfig) configuration class: [PegasusForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusForConditionalGeneration) (Pegasus model)
  + [PegasusXConfig](/docs/transformers/v4.56.2/en/model_doc/pegasus_x#transformers.PegasusXConfig) configuration class: [PegasusXForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pegasus_x#transformers.PegasusXForConditionalGeneration) (PEGASUS-X model)
  + [ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig) configuration class: [ProphetNetForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForConditionalGeneration) (ProphetNet model)
  + [Qwen2AudioConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioConfig) configuration class: [Qwen2AudioForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioForConditionalGeneration) (Qwen2Audio model)
  + [SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig) configuration class: [SeamlessM4TForTextToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForTextToText) (SeamlessM4T model)
  + [SeamlessM4Tv2Config](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Config) configuration class: [SeamlessM4Tv2ForTextToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToText) (SeamlessM4Tv2 model)
  + [SwitchTransformersConfig](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig) configuration class: [SwitchTransformersForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersForConditionalGeneration) (SwitchTransformers model)
  + [T5Config](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config) configuration class: [T5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForConditionalGeneration) (T5 model)
  + [T5GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig) configuration class: [T5GemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForConditionalGeneration) (T5Gemma model)
  + [UMT5Config](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config) configuration class: [UMT5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForConditionalGeneration) (UMT5 model)
  + [VoxtralConfig](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralConfig) configuration class: [VoxtralForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralForConditionalGeneration) (Voxtral model)
  + [XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig) configuration class: [XLMProphetNetForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForConditionalGeneration) (XLM-ProphetNet model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a sequence-to-sequence language modeling head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForSeq2SeqLM

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-t5/t5-base")
>>> model = AutoModelForSeq2SeqLM.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a sequence-to-sequence language modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **bart** — [BartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForConditionalGeneration) (BART model)
* **bigbird\_pegasus** — [BigBirdPegasusForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForConditionalGeneration) (BigBird-Pegasus model)
* **blenderbot** — [BlenderbotForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotForConditionalGeneration) (Blenderbot model)
* **blenderbot-small** — [BlenderbotSmallForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForConditionalGeneration) (BlenderbotSmall model)
* **encoder-decoder** — [EncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) (Encoder decoder model)
* **fsmt** — [FSMTForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTForConditionalGeneration) (FairSeq Machine-Translation model)
* **gptsan-japanese** — [GPTSanJapaneseForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gptsan-japanese#transformers.GPTSanJapaneseForConditionalGeneration) (GPTSAN-japanese model)
* **granite\_speech** — [GraniteSpeechForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechForConditionalGeneration) (GraniteSpeech model)
* **led** — [LEDForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDForConditionalGeneration) (LED model)
* **longt5** — [LongT5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5ForConditionalGeneration) (LongT5 model)
* **m2m\_100** — [M2M100ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/m2m_100#transformers.M2M100ForConditionalGeneration) (M2M100 model)
* **marian** — [MarianMTModel](/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianMTModel) (Marian model)
* **mbart** — [MBartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForConditionalGeneration) (mBART model)
* **mt5** — [MT5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForConditionalGeneration) (MT5 model)
* **mvp** — [MvpForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForConditionalGeneration) (MVP model)
* **nllb-moe** — [NllbMoeForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/nllb-moe#transformers.NllbMoeForConditionalGeneration) (NLLB-MOE model)
* **pegasus** — [PegasusForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusForConditionalGeneration) (Pegasus model)
* **pegasus\_x** — [PegasusXForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pegasus_x#transformers.PegasusXForConditionalGeneration) (PEGASUS-X model)
* **plbart** — [PLBartForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForConditionalGeneration) (PLBart model)
* **prophetnet** — [ProphetNetForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForConditionalGeneration) (ProphetNet model)
* **qwen2\_audio** — [Qwen2AudioForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioForConditionalGeneration) (Qwen2Audio model)
* **seamless\_m4t** — [SeamlessM4TForTextToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForTextToText) (SeamlessM4T model)
* **seamless\_m4t\_v2** — [SeamlessM4Tv2ForTextToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToText) (SeamlessM4Tv2 model)
* **switch\_transformers** — [SwitchTransformersForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersForConditionalGeneration) (SwitchTransformers model)
* **t5** — [T5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForConditionalGeneration) (T5 model)
* **t5gemma** — [T5GemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForConditionalGeneration) (T5Gemma model)
* **umt5** — [UMT5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForConditionalGeneration) (UMT5 model)
* **voxtral** — [VoxtralForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralForConditionalGeneration) (Voxtral model)
* **xlm-prophetnet** — [XLMProphetNetForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForConditionalGeneration) (XLM-ProphetNet model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForSeq2SeqLM

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

>>> # Update configuration during loading
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/t5_tf_model_config.json")
>>> model = AutoModelForSeq2SeqLM.from_pretrained(
...     "./tf_model/t5_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForSequenceClassification

### class transformers.AutoModelForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1955)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence classification head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig) configuration class: `AlbertForSequenceClassification` (ALBERT model)
  + [ArceeConfig](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeConfig) configuration class: [ArceeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForSequenceClassification) (Arcee model)
  + [BartConfig](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartConfig) configuration class: [BartForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForSequenceClassification) (BART model)
  + [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) configuration class: [BertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForSequenceClassification) (BERT model)
  + [BigBirdConfig](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig) configuration class: [BigBirdForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForSequenceClassification) (BigBird model)
  + [BigBirdPegasusConfig](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig) configuration class: [BigBirdPegasusForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForSequenceClassification) (BigBird-Pegasus model)
  + [BioGptConfig](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig) configuration class: [BioGptForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForSequenceClassification) (BioGpt model)
  + [BloomConfig](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig) configuration class: [BloomForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForSequenceClassification) (BLOOM model)
  + [CTRLConfig](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig) configuration class: [CTRLForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLForSequenceClassification) (CTRL model)
  + [CamembertConfig](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig) configuration class: [CamembertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForSequenceClassification) (CamemBERT model)
  + [CanineConfig](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineConfig) configuration class: [CanineForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineForSequenceClassification) (CANINE model)
  + [ConvBertConfig](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertConfig) configuration class: [ConvBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForSequenceClassification) (ConvBERT model)
  + [Data2VecTextConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig) configuration class: [Data2VecTextForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForSequenceClassification) (Data2VecText model)
  + [DebertaConfig](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaConfig) configuration class: [DebertaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForSequenceClassification) (DeBERTa model)
  + [DebertaV2Config](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config) configuration class: [DebertaV2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForSequenceClassification) (DeBERTa-v2 model)
  + [DeepseekV2Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config) configuration class: [DeepseekV2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2ForSequenceClassification) (DeepSeek-V2 model)
  + [DeepseekV3Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Config) configuration class: [DeepseekV3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3ForSequenceClassification) (DeepSeek-V3 model)
  + [DiffLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaConfig) configuration class: [DiffLlamaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForSequenceClassification) (DiffLlama model)
  + [DistilBertConfig](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertConfig) configuration class: [DistilBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification) (DistilBERT model)
  + [DogeConfig](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeConfig) configuration class: [DogeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeForSequenceClassification) (Doge model)
  + [ElectraConfig](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig) configuration class: [ElectraForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForSequenceClassification) (ELECTRA model)
  + [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForSequenceClassification) (ERNIE model)
  + [ErnieMConfig](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMConfig) configuration class: [ErnieMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMForSequenceClassification) (ErnieM model)
  + [EsmConfig](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig) configuration class: [EsmForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForSequenceClassification) (ESM model)
  + [Exaone4Config](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config) configuration class: [Exaone4ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForSequenceClassification) (EXAONE-4.0 model)
  + [FNetConfig](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig) configuration class: [FNetForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForSequenceClassification) (FNet model)
  + [FalconConfig](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig) configuration class: [FalconForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForSequenceClassification) (Falcon model)
  + [FlaubertConfig](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertConfig) configuration class: [FlaubertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertForSequenceClassification) (FlauBERT model)
  + [FunnelConfig](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelConfig) configuration class: [FunnelForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForSequenceClassification) (Funnel Transformer model)
  + [GPT2Config](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config) configuration class: [GPT2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification) (OpenAI GPT-2 model)
  + [GPTBigCodeConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeConfig) configuration class: [GPTBigCodeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeForSequenceClassification) (GPTBigCode model)
  + [GPTJConfig](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJConfig) configuration class: [GPTJForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJForSequenceClassification) (GPT-J model)
  + [GPTNeoConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig) configuration class: [GPTNeoForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForSequenceClassification) (GPT Neo model)
  + [GPTNeoXConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXConfig) configuration class: [GPTNeoXForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXForSequenceClassification) (GPT NeoX model)
  + [Gemma2Config](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Config) configuration class: [Gemma2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2ForSequenceClassification) (Gemma2 model)
  + [Gemma3Config](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config) configuration class: [Gemma3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForSequenceClassification) (Gemma3ForConditionalGeneration model)
  + [GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaConfig) configuration class: [GemmaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaForSequenceClassification) (Gemma model)
  + [Glm4Config](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Config) configuration class: [Glm4ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4ForSequenceClassification) (GLM4 model)
  + [GlmConfig](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmConfig) configuration class: [GlmForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmForSequenceClassification) (GLM model)
  + [GptOssConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssConfig) configuration class: [GptOssForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssForSequenceClassification) (GptOss model)
  + [HeliumConfig](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumConfig) configuration class: [HeliumForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumForSequenceClassification) (Helium model)
  + [HunYuanDenseV1Config](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Config) configuration class: [HunYuanDenseV1ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1ForSequenceClassification) (HunYuanDenseV1 model)
  + [HunYuanMoEV1Config](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Config) configuration class: [HunYuanMoEV1ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1ForSequenceClassification) (HunYuanMoeV1 model)
  + [IBertConfig](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertConfig) configuration class: [IBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForSequenceClassification) (I-BERT model)
  + [JambaConfig](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaConfig) configuration class: [JambaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaForSequenceClassification) (Jamba model)
  + [JetMoeConfig](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeConfig) configuration class: [JetMoeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeForSequenceClassification) (JetMoe model)
  + [LEDConfig](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDConfig) configuration class: [LEDForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDForSequenceClassification) (LED model)
  + [LayoutLMConfig](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig) configuration class: [LayoutLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForSequenceClassification) (LayoutLM model)
  + [LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config) configuration class: [LayoutLMv2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForSequenceClassification) (LayoutLMv2 model)
  + [LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config) configuration class: [LayoutLMv3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForSequenceClassification) (LayoutLMv3 model)
  + [LiltConfig](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltConfig) configuration class: [LiltForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForSequenceClassification) (LiLT model)
  + [LlamaConfig](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig) configuration class: [LlamaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForSequenceClassification) (LLaMA model)
  + [LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig) configuration class: [LongformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForSequenceClassification) (Longformer model)
  + [LukeConfig](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig) configuration class: [LukeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForSequenceClassification) (LUKE model)
  + [MBartConfig](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig) configuration class: [MBartForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForSequenceClassification) (mBART model)
  + [MPNetConfig](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig) configuration class: [MPNetForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForSequenceClassification) (MPNet model)
  + [MT5Config](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config) configuration class: [MT5ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForSequenceClassification) (MT5 model)
  + [MarkupLMConfig](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMConfig) configuration class: [MarkupLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForSequenceClassification) (MarkupLM model)
  + [MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig) configuration class: [MegaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForSequenceClassification) (MEGA model)
  + [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForSequenceClassification) (Megatron-BERT model)
  + [MiniMaxConfig](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxConfig) configuration class: [MiniMaxForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxForSequenceClassification) (MiniMax model)
  + [MistralConfig](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig) configuration class: [MistralForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForSequenceClassification) (Mistral model)
  + [MixtralConfig](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralConfig) configuration class: [MixtralForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralForSequenceClassification) (Mixtral model)
  + [MobileBertConfig](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [MobileBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForSequenceClassification) (MobileBERT model)
  + [ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig) configuration class: [ModernBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForSequenceClassification) (ModernBERT model)
  + [ModernBertDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderConfig) configuration class: [ModernBertDecoderForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderForSequenceClassification) (ModernBertDecoder model)
  + [MptConfig](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptConfig) configuration class: [MptForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForSequenceClassification) (MPT model)
  + [MraConfig](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraConfig) configuration class: [MraForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForSequenceClassification) (MRA model)
  + [MvpConfig](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig) configuration class: [MvpForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForSequenceClassification) (MVP model)
  + [NemotronConfig](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronConfig) configuration class: [NemotronForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronForSequenceClassification) (Nemotron model)
  + [NezhaConfig](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig) configuration class: [NezhaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForSequenceClassification) (Nezha model)
  + [NystromformerConfig](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig) configuration class: [NystromformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForSequenceClassification) (Nyströmformer model)
  + [OPTConfig](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig) configuration class: [OPTForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForSequenceClassification) (OPT model)
  + [OpenAIGPTConfig](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig) configuration class: [OpenAIGPTForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTForSequenceClassification) (OpenAI GPT model)
  + [OpenLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaConfig) configuration class: [OpenLlamaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaForSequenceClassification) (OpenLlama model)
  + [PLBartConfig](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig) configuration class: [PLBartForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForSequenceClassification) (PLBart model)
  + [PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig) configuration class: [PerceiverForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForSequenceClassification) (Perceiver model)
  + [PersimmonConfig](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonConfig) configuration class: [PersimmonForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonForSequenceClassification) (Persimmon model)
  + [Phi3Config](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Config) configuration class: [Phi3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3ForSequenceClassification) (Phi3 model)
  + [PhiConfig](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiConfig) configuration class: [PhiForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiForSequenceClassification) (Phi model)
  + [PhimoeConfig](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeConfig) configuration class: [PhimoeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeForSequenceClassification) (Phimoe model)
  + [QDQBertConfig](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig) configuration class: [QDQBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForSequenceClassification) (QDQBert model)
  + [Qwen2Config](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Config) configuration class: [Qwen2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForSequenceClassification) (Qwen2 model)
  + [Qwen2MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeConfig) configuration class: [Qwen2MoeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeForSequenceClassification) (Qwen2MoE model)
  + [Qwen3Config](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Config) configuration class: [Qwen3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForSequenceClassification) (Qwen3 model)
  + [Qwen3MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeConfig) configuration class: [Qwen3MoeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForSequenceClassification) (Qwen3MoE model)
  + [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig) configuration class: [ReformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForSequenceClassification) (Reformer model)
  + [RemBertConfig](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertConfig) configuration class: [RemBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForSequenceClassification) (RemBERT model)
  + [RoCBertConfig](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig) configuration class: [RoCBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForSequenceClassification) (RoCBert model)
  + [RoFormerConfig](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig) configuration class: [RoFormerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForSequenceClassification) (RoFormer model)
  + [RobertaConfig](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig) configuration class: [RobertaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForSequenceClassification) (RoBERTa model)
  + [RobertaPreLayerNormConfig](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig) configuration class: [RobertaPreLayerNormForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForSequenceClassification) (RoBERTa-PreLayerNorm model)
  + [SeedOssConfig](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssConfig) configuration class: [SeedOssForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForSequenceClassification) (SeedOss model)
  + [SmolLM3Config](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Config) configuration class: [SmolLM3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForSequenceClassification) (SmolLM3 model)
  + [SqueezeBertConfig](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig) configuration class: [SqueezeBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForSequenceClassification) (SqueezeBERT model)
  + [StableLmConfig](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmConfig) configuration class: [StableLmForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmForSequenceClassification) (StableLm model)
  + [Starcoder2Config](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2Config) configuration class: [Starcoder2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2ForSequenceClassification) (Starcoder2 model)
  + [T5Config](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config) configuration class: [T5ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForSequenceClassification) (T5 model)
  + [T5GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig) configuration class: [T5GemmaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForSequenceClassification) (T5Gemma model)
  + [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) configuration class: [TapasForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForSequenceClassification) (TAPAS model)
  + [TransfoXLConfig](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLConfig) configuration class: [TransfoXLForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLForSequenceClassification) (Transformer-XL model)
  + [UMT5Config](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config) configuration class: [UMT5ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForSequenceClassification) (UMT5 model)
  + [XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig) configuration class: [XLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForSequenceClassification) (XLM model)
  + [XLMRobertaConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig) configuration class: [XLMRobertaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForSequenceClassification) (XLM-RoBERTa model)
  + [XLMRobertaXLConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig) configuration class: [XLMRobertaXLForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForSequenceClassification) (XLM-RoBERTa-XL model)
  + [XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig) configuration class: [XLNetForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForSequenceClassification) (XLNet model)
  + [XmodConfig](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig) configuration class: [XmodForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForSequenceClassification) (X-MOD model)
  + [YosoConfig](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig) configuration class: [YosoForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForSequenceClassification) (YOSO model)
  + [Zamba2Config](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2Config) configuration class: [Zamba2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2ForSequenceClassification) (Zamba2 model)
  + [ZambaConfig](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaConfig) configuration class: [ZambaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaForSequenceClassification) (Zamba model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a sequence classification head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForSequenceClassification

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForSequenceClassification.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a sequence classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **albert** — `AlbertForSequenceClassification` (ALBERT model)
* **arcee** — [ArceeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForSequenceClassification) (Arcee model)
* **bart** — [BartForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForSequenceClassification) (BART model)
* **bert** — [BertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForSequenceClassification) (BERT model)
* **big\_bird** — [BigBirdForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForSequenceClassification) (BigBird model)
* **bigbird\_pegasus** — [BigBirdPegasusForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForSequenceClassification) (BigBird-Pegasus model)
* **biogpt** — [BioGptForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForSequenceClassification) (BioGpt model)
* **bloom** — [BloomForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForSequenceClassification) (BLOOM model)
* **camembert** — [CamembertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForSequenceClassification) (CamemBERT model)
* **canine** — [CanineForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineForSequenceClassification) (CANINE model)
* **code\_llama** — [LlamaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForSequenceClassification) (CodeLlama model)
* **convbert** — [ConvBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForSequenceClassification) (ConvBERT model)
* **ctrl** — [CTRLForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLForSequenceClassification) (CTRL model)
* **data2vec-text** — [Data2VecTextForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForSequenceClassification) (Data2VecText model)
* **deberta** — [DebertaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForSequenceClassification) (DeBERTa model)
* **deberta-v2** — [DebertaV2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForSequenceClassification) (DeBERTa-v2 model)
* **deepseek\_v2** — [DeepseekV2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2ForSequenceClassification) (DeepSeek-V2 model)
* **deepseek\_v3** — [DeepseekV3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3ForSequenceClassification) (DeepSeek-V3 model)
* **diffllama** — [DiffLlamaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForSequenceClassification) (DiffLlama model)
* **distilbert** — [DistilBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification) (DistilBERT model)
* **doge** — [DogeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeForSequenceClassification) (Doge model)
* **electra** — [ElectraForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForSequenceClassification) (ELECTRA model)
* **ernie** — [ErnieForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForSequenceClassification) (ERNIE model)
* **ernie\_m** — [ErnieMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMForSequenceClassification) (ErnieM model)
* **esm** — [EsmForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForSequenceClassification) (ESM model)
* **exaone4** — [Exaone4ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForSequenceClassification) (EXAONE-4.0 model)
* **falcon** — [FalconForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForSequenceClassification) (Falcon model)
* **flaubert** — [FlaubertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertForSequenceClassification) (FlauBERT model)
* **fnet** — [FNetForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForSequenceClassification) (FNet model)
* **funnel** — [FunnelForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForSequenceClassification) (Funnel Transformer model)
* **gemma** — [GemmaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaForSequenceClassification) (Gemma model)
* **gemma2** — [Gemma2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2ForSequenceClassification) (Gemma2 model)
* **gemma3** — [Gemma3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForSequenceClassification) (Gemma3ForConditionalGeneration model)
* **glm** — [GlmForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmForSequenceClassification) (GLM model)
* **glm4** — [Glm4ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4ForSequenceClassification) (GLM4 model)
* **gpt-sw3** — [GPT2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification) (GPT-Sw3 model)
* **gpt2** — [GPT2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification) (OpenAI GPT-2 model)
* **gpt\_bigcode** — [GPTBigCodeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeForSequenceClassification) (GPTBigCode model)
* **gpt\_neo** — [GPTNeoForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForSequenceClassification) (GPT Neo model)
* **gpt\_neox** — [GPTNeoXForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXForSequenceClassification) (GPT NeoX model)
* **gpt\_oss** — [GptOssForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssForSequenceClassification) (GptOss model)
* **gptj** — [GPTJForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJForSequenceClassification) (GPT-J model)
* **helium** — [HeliumForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumForSequenceClassification) (Helium model)
* **hunyuan\_v1\_dense** — [HunYuanDenseV1ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1ForSequenceClassification) (HunYuanDenseV1 model)
* **hunyuan\_v1\_moe** — [HunYuanMoEV1ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1ForSequenceClassification) (HunYuanMoeV1 model)
* **ibert** — [IBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForSequenceClassification) (I-BERT model)
* **jamba** — [JambaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/jamba#transformers.JambaForSequenceClassification) (Jamba model)
* **jetmoe** — [JetMoeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeForSequenceClassification) (JetMoe model)
* **layoutlm** — [LayoutLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForSequenceClassification) (LayoutLM model)
* **layoutlmv2** — [LayoutLMv2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForSequenceClassification) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForSequenceClassification) (LayoutLMv3 model)
* **led** — [LEDForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDForSequenceClassification) (LED model)
* **lilt** — [LiltForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForSequenceClassification) (LiLT model)
* **llama** — [LlamaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForSequenceClassification) (LLaMA model)
* **longformer** — [LongformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForSequenceClassification) (Longformer model)
* **luke** — [LukeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForSequenceClassification) (LUKE model)
* **markuplm** — [MarkupLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForSequenceClassification) (MarkupLM model)
* **mbart** — [MBartForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForSequenceClassification) (mBART model)
* **mega** — [MegaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForSequenceClassification) (MEGA model)
* **megatron-bert** — [MegatronBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForSequenceClassification) (Megatron-BERT model)
* **minimax** — [MiniMaxForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxForSequenceClassification) (MiniMax model)
* **mistral** — [MistralForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForSequenceClassification) (Mistral model)
* **mixtral** — [MixtralForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralForSequenceClassification) (Mixtral model)
* **mobilebert** — [MobileBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForSequenceClassification) (MobileBERT model)
* **modernbert** — [ModernBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForSequenceClassification) (ModernBERT model)
* **modernbert-decoder** — [ModernBertDecoderForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/modernbert-decoder#transformers.ModernBertDecoderForSequenceClassification) (ModernBertDecoder model)
* **mpnet** — [MPNetForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForSequenceClassification) (MPNet model)
* **mpt** — [MptForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForSequenceClassification) (MPT model)
* **mra** — [MraForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForSequenceClassification) (MRA model)
* **mt5** — [MT5ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForSequenceClassification) (MT5 model)
* **mvp** — [MvpForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForSequenceClassification) (MVP model)
* **nemotron** — [NemotronForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronForSequenceClassification) (Nemotron model)
* **nezha** — [NezhaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForSequenceClassification) (Nezha model)
* **nystromformer** — [NystromformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForSequenceClassification) (Nyströmformer model)
* **open-llama** — [OpenLlamaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/open-llama#transformers.OpenLlamaForSequenceClassification) (OpenLlama model)
* **openai-gpt** — [OpenAIGPTForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/openai-gpt#transformers.OpenAIGPTForSequenceClassification) (OpenAI GPT model)
* **opt** — [OPTForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForSequenceClassification) (OPT model)
* **perceiver** — [PerceiverForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForSequenceClassification) (Perceiver model)
* **persimmon** — [PersimmonForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonForSequenceClassification) (Persimmon model)
* **phi** — [PhiForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiForSequenceClassification) (Phi model)
* **phi3** — [Phi3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3ForSequenceClassification) (Phi3 model)
* **phimoe** — [PhimoeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeForSequenceClassification) (Phimoe model)
* **plbart** — [PLBartForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForSequenceClassification) (PLBart model)
* **qdqbert** — [QDQBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForSequenceClassification) (QDQBert model)
* **qwen2** — [Qwen2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForSequenceClassification) (Qwen2 model)
* **qwen2\_moe** — [Qwen2MoeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeForSequenceClassification) (Qwen2MoE model)
* **qwen3** — [Qwen3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForSequenceClassification) (Qwen3 model)
* **qwen3\_moe** — [Qwen3MoeForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForSequenceClassification) (Qwen3MoE model)
* **reformer** — [ReformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForSequenceClassification) (Reformer model)
* **rembert** — [RemBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForSequenceClassification) (RemBERT model)
* **roberta** — [RobertaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForSequenceClassification) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaPreLayerNormForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForSequenceClassification) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForSequenceClassification) (RoCBert model)
* **roformer** — [RoFormerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForSequenceClassification) (RoFormer model)
* **seed\_oss** — [SeedOssForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForSequenceClassification) (SeedOss model)
* **smollm3** — [SmolLM3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForSequenceClassification) (SmolLM3 model)
* **squeezebert** — [SqueezeBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForSequenceClassification) (SqueezeBERT model)
* **stablelm** — [StableLmForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmForSequenceClassification) (StableLm model)
* **starcoder2** — [Starcoder2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2ForSequenceClassification) (Starcoder2 model)
* **t5** — [T5ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForSequenceClassification) (T5 model)
* **t5gemma** — [T5GemmaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForSequenceClassification) (T5Gemma model)
* **tapas** — [TapasForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForSequenceClassification) (TAPAS model)
* **transfo-xl** — [TransfoXLForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/transfo-xl#transformers.TransfoXLForSequenceClassification) (Transformer-XL model)
* **umt5** — [UMT5ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForSequenceClassification) (UMT5 model)
* **xlm** — [XLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForSequenceClassification) (XLM model)
* **xlm-roberta** — [XLMRobertaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForSequenceClassification) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaXLForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForSequenceClassification) (XLM-RoBERTa-XL model)
* **xlnet** — [XLNetForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForSequenceClassification) (XLNet model)
* **xmod** — [XmodForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForSequenceClassification) (X-MOD model)
* **yoso** — [YosoForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForSequenceClassification) (YOSO model)
* **zamba** — [ZambaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/zamba#transformers.ZambaForSequenceClassification) (Zamba model)
* **zamba2** — [Zamba2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/zamba2#transformers.Zamba2ForSequenceClassification) (Zamba2 model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForSequenceClassification

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForSequenceClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForMultipleChoice

### class transformers.AutoModelForMultipleChoice

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2011)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a multiple choice head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig) configuration class: [AlbertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertForMultipleChoice) (ALBERT model)
  + [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) configuration class: [BertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForMultipleChoice) (BERT model)
  + [BigBirdConfig](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig) configuration class: [BigBirdForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForMultipleChoice) (BigBird model)
  + [CamembertConfig](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig) configuration class: [CamembertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMultipleChoice) (CamemBERT model)
  + [CanineConfig](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineConfig) configuration class: [CanineForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineForMultipleChoice) (CANINE model)
  + [ConvBertConfig](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertConfig) configuration class: [ConvBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForMultipleChoice) (ConvBERT model)
  + [Data2VecTextConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig) configuration class: [Data2VecTextForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMultipleChoice) (Data2VecText model)
  + [DebertaV2Config](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config) configuration class: [DebertaV2ForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMultipleChoice) (DeBERTa-v2 model)
  + [DistilBertConfig](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertConfig) configuration class: [DistilBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForMultipleChoice) (DistilBERT model)
  + [ElectraConfig](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig) configuration class: [ElectraForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForMultipleChoice) (ELECTRA model)
  + [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForMultipleChoice) (ERNIE model)
  + [ErnieMConfig](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMConfig) configuration class: [ErnieMForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMForMultipleChoice) (ErnieM model)
  + [FNetConfig](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig) configuration class: [FNetForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForMultipleChoice) (FNet model)
  + [FlaubertConfig](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertConfig) configuration class: [FlaubertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertForMultipleChoice) (FlauBERT model)
  + [FunnelConfig](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelConfig) configuration class: [FunnelForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForMultipleChoice) (Funnel Transformer model)
  + [IBertConfig](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertConfig) configuration class: [IBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForMultipleChoice) (I-BERT model)
  + [LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig) configuration class: [LongformerForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMultipleChoice) (Longformer model)
  + [LukeConfig](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig) configuration class: [LukeForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMultipleChoice) (LUKE model)
  + [MPNetConfig](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig) configuration class: [MPNetForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForMultipleChoice) (MPNet model)
  + [MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig) configuration class: [MegaForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMultipleChoice) (MEGA model)
  + [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForMultipleChoice) (Megatron-BERT model)
  + [MobileBertConfig](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [MobileBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForMultipleChoice) (MobileBERT model)
  + [ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig) configuration class: [ModernBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForMultipleChoice) (ModernBERT model)
  + [MraConfig](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraConfig) configuration class: [MraForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForMultipleChoice) (MRA model)
  + [NezhaConfig](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig) configuration class: [NezhaForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForMultipleChoice) (Nezha model)
  + [NystromformerConfig](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig) configuration class: [NystromformerForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForMultipleChoice) (Nyströmformer model)
  + [QDQBertConfig](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig) configuration class: [QDQBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForMultipleChoice) (QDQBert model)
  + [RemBertConfig](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertConfig) configuration class: [RemBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForMultipleChoice) (RemBERT model)
  + [RoCBertConfig](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig) configuration class: [RoCBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForMultipleChoice) (RoCBert model)
  + [RoFormerConfig](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig) configuration class: [RoFormerForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForMultipleChoice) (RoFormer model)
  + [RobertaConfig](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig) configuration class: [RobertaForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMultipleChoice) (RoBERTa model)
  + [RobertaPreLayerNormConfig](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig) configuration class: [RobertaPreLayerNormForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMultipleChoice) (RoBERTa-PreLayerNorm model)
  + [SqueezeBertConfig](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig) configuration class: [SqueezeBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMultipleChoice) (SqueezeBERT model)
  + [XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig) configuration class: [XLMForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForMultipleChoice) (XLM model)
  + [XLMRobertaConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig) configuration class: [XLMRobertaForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMultipleChoice) (XLM-RoBERTa model)
  + [XLMRobertaXLConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig) configuration class: [XLMRobertaXLForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMultipleChoice) (XLM-RoBERTa-XL model)
  + [XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig) configuration class: [XLNetForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForMultipleChoice) (XLNet model)
  + [XmodConfig](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig) configuration class: [XmodForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMultipleChoice) (X-MOD model)
  + [YosoConfig](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig) configuration class: [YosoForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForMultipleChoice) (YOSO model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a multiple choice head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForMultipleChoice

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForMultipleChoice.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a multiple choice head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **albert** — [AlbertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertForMultipleChoice) (ALBERT model)
* **bert** — [BertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForMultipleChoice) (BERT model)
* **big\_bird** — [BigBirdForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForMultipleChoice) (BigBird model)
* **camembert** — [CamembertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMultipleChoice) (CamemBERT model)
* **canine** — [CanineForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineForMultipleChoice) (CANINE model)
* **convbert** — [ConvBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForMultipleChoice) (ConvBERT model)
* **data2vec-text** — [Data2VecTextForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMultipleChoice) (Data2VecText model)
* **deberta-v2** — [DebertaV2ForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMultipleChoice) (DeBERTa-v2 model)
* **distilbert** — [DistilBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForMultipleChoice) (DistilBERT model)
* **electra** — [ElectraForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForMultipleChoice) (ELECTRA model)
* **ernie** — [ErnieForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForMultipleChoice) (ERNIE model)
* **ernie\_m** — [ErnieMForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMForMultipleChoice) (ErnieM model)
* **flaubert** — [FlaubertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertForMultipleChoice) (FlauBERT model)
* **fnet** — [FNetForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForMultipleChoice) (FNet model)
* **funnel** — [FunnelForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForMultipleChoice) (Funnel Transformer model)
* **ibert** — [IBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForMultipleChoice) (I-BERT model)
* **longformer** — [LongformerForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForMultipleChoice) (Longformer model)
* **luke** — [LukeForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMultipleChoice) (LUKE model)
* **mega** — [MegaForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMultipleChoice) (MEGA model)
* **megatron-bert** — [MegatronBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForMultipleChoice) (Megatron-BERT model)
* **mobilebert** — [MobileBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForMultipleChoice) (MobileBERT model)
* **modernbert** — [ModernBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForMultipleChoice) (ModernBERT model)
* **mpnet** — [MPNetForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForMultipleChoice) (MPNet model)
* **mra** — [MraForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForMultipleChoice) (MRA model)
* **nezha** — [NezhaForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForMultipleChoice) (Nezha model)
* **nystromformer** — [NystromformerForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForMultipleChoice) (Nyströmformer model)
* **qdqbert** — [QDQBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForMultipleChoice) (QDQBert model)
* **rembert** — [RemBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForMultipleChoice) (RemBERT model)
* **roberta** — [RobertaForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMultipleChoice) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaPreLayerNormForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMultipleChoice) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForMultipleChoice) (RoCBert model)
* **roformer** — [RoFormerForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForMultipleChoice) (RoFormer model)
* **squeezebert** — [SqueezeBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMultipleChoice) (SqueezeBERT model)
* **xlm** — [XLMForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForMultipleChoice) (XLM model)
* **xlm-roberta** — [XLMRobertaForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMultipleChoice) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaXLForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMultipleChoice) (XLM-RoBERTa-XL model)
* **xlnet** — [XLNetForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForMultipleChoice) (XLNet model)
* **xmod** — [XmodForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMultipleChoice) (X-MOD model)
* **yoso** — [YosoForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForMultipleChoice) (YOSO model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForMultipleChoice

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForMultipleChoice.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForNextSentencePrediction

### class transformers.AutoModelForNextSentencePrediction

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2018)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a next sentence prediction head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) configuration class: [BertForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForNextSentencePrediction) (BERT model)
  + [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForNextSentencePrediction) (ERNIE model)
  + [FNetConfig](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig) configuration class: [FNetForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForNextSentencePrediction) (FNet model)
  + [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForNextSentencePrediction) (Megatron-BERT model)
  + [MobileBertConfig](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [MobileBertForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForNextSentencePrediction) (MobileBERT model)
  + [NezhaConfig](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig) configuration class: [NezhaForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForNextSentencePrediction) (Nezha model)
  + [QDQBertConfig](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig) configuration class: [QDQBertForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForNextSentencePrediction) (QDQBert model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a next sentence prediction head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForNextSentencePrediction

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForNextSentencePrediction.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a next sentence prediction head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **bert** — [BertForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForNextSentencePrediction) (BERT model)
* **ernie** — [ErnieForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForNextSentencePrediction) (ERNIE model)
* **fnet** — [FNetForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForNextSentencePrediction) (FNet model)
* **megatron-bert** — [MegatronBertForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForNextSentencePrediction) (Megatron-BERT model)
* **mobilebert** — [MobileBertForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForNextSentencePrediction) (MobileBERT model)
* **nezha** — [NezhaForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForNextSentencePrediction) (Nezha model)
* **qdqbert** — [QDQBertForNextSentencePrediction](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForNextSentencePrediction) (QDQBert model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForNextSentencePrediction

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForNextSentencePrediction.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForNextSentencePrediction.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForNextSentencePrediction.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForTokenClassification

### class transformers.AutoModelForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2004)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a token classification head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig) configuration class: `AlbertForTokenClassification` (ALBERT model)
  + [ApertusConfig](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusConfig) configuration class: [ApertusForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusForTokenClassification) (Apertus model)
  + [ArceeConfig](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeConfig) configuration class: [ArceeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForTokenClassification) (Arcee model)
  + [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) configuration class: [BertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForTokenClassification) (BERT model)
  + [BigBirdConfig](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig) configuration class: [BigBirdForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForTokenClassification) (BigBird model)
  + [BioGptConfig](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig) configuration class: [BioGptForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForTokenClassification) (BioGpt model)
  + [BloomConfig](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig) configuration class: [BloomForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForTokenClassification) (BLOOM model)
  + [BrosConfig](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosConfig) configuration class: [BrosForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosForTokenClassification) (BROS model)
  + [CamembertConfig](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig) configuration class: [CamembertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForTokenClassification) (CamemBERT model)
  + [CanineConfig](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineConfig) configuration class: [CanineForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineForTokenClassification) (CANINE model)
  + [ConvBertConfig](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertConfig) configuration class: [ConvBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForTokenClassification) (ConvBERT model)
  + [Data2VecTextConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig) configuration class: [Data2VecTextForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForTokenClassification) (Data2VecText model)
  + [DebertaConfig](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaConfig) configuration class: [DebertaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForTokenClassification) (DeBERTa model)
  + [DebertaV2Config](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config) configuration class: [DebertaV2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForTokenClassification) (DeBERTa-v2 model)
  + [DiffLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaConfig) configuration class: [DiffLlamaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForTokenClassification) (DiffLlama model)
  + [DistilBertConfig](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertConfig) configuration class: [DistilBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForTokenClassification) (DistilBERT model)
  + [ElectraConfig](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig) configuration class: [ElectraForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForTokenClassification) (ELECTRA model)
  + [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForTokenClassification) (ERNIE model)
  + [ErnieMConfig](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMConfig) configuration class: [ErnieMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMForTokenClassification) (ErnieM model)
  + [EsmConfig](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig) configuration class: [EsmForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForTokenClassification) (ESM model)
  + [Exaone4Config](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config) configuration class: [Exaone4ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForTokenClassification) (EXAONE-4.0 model)
  + [FNetConfig](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig) configuration class: [FNetForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForTokenClassification) (FNet model)
  + [FalconConfig](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig) configuration class: [FalconForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForTokenClassification) (Falcon model)
  + [FlaubertConfig](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertConfig) configuration class: [FlaubertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertForTokenClassification) (FlauBERT model)
  + [FunnelConfig](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelConfig) configuration class: [FunnelForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForTokenClassification) (Funnel Transformer model)
  + [GPT2Config](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config) configuration class: [GPT2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForTokenClassification) (OpenAI GPT-2 model)
  + [GPTBigCodeConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeConfig) configuration class: [GPTBigCodeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeForTokenClassification) (GPTBigCode model)
  + [GPTNeoConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig) configuration class: [GPTNeoForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForTokenClassification) (GPT Neo model)
  + [GPTNeoXConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXConfig) configuration class: [GPTNeoXForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXForTokenClassification) (GPT NeoX model)
  + [Gemma2Config](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Config) configuration class: [Gemma2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2ForTokenClassification) (Gemma2 model)
  + [GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaConfig) configuration class: [GemmaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaForTokenClassification) (Gemma model)
  + [Glm4Config](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Config) configuration class: [Glm4ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4ForTokenClassification) (GLM4 model)
  + [GlmConfig](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmConfig) configuration class: [GlmForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmForTokenClassification) (GLM model)
  + [GptOssConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssConfig) configuration class: [GptOssForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssForTokenClassification) (GptOss model)
  + [HeliumConfig](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumConfig) configuration class: [HeliumForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumForTokenClassification) (Helium model)
  + [IBertConfig](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertConfig) configuration class: [IBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForTokenClassification) (I-BERT model)
  + [LayoutLMConfig](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig) configuration class: [LayoutLMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForTokenClassification) (LayoutLM model)
  + [LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config) configuration class: [LayoutLMv2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification) (LayoutLMv2 model)
  + [LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config) configuration class: [LayoutLMv3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForTokenClassification) (LayoutLMv3 model)
  + [LiltConfig](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltConfig) configuration class: [LiltForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForTokenClassification) (LiLT model)
  + [LlamaConfig](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig) configuration class: [LlamaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForTokenClassification) (LLaMA model)
  + [LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig) configuration class: [LongformerForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForTokenClassification) (Longformer model)
  + [LukeConfig](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig) configuration class: [LukeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForTokenClassification) (LUKE model)
  + [MPNetConfig](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig) configuration class: [MPNetForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForTokenClassification) (MPNet model)
  + [MT5Config](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config) configuration class: [MT5ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForTokenClassification) (MT5 model)
  + [MarkupLMConfig](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMConfig) configuration class: [MarkupLMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForTokenClassification) (MarkupLM model)
  + [MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig) configuration class: [MegaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForTokenClassification) (MEGA model)
  + [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForTokenClassification) (Megatron-BERT model)
  + [MiniMaxConfig](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxConfig) configuration class: [MiniMaxForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxForTokenClassification) (MiniMax model)
  + [MistralConfig](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig) configuration class: [MistralForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForTokenClassification) (Mistral model)
  + [MixtralConfig](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralConfig) configuration class: [MixtralForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralForTokenClassification) (Mixtral model)
  + [MobileBertConfig](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [MobileBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForTokenClassification) (MobileBERT model)
  + [ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig) configuration class: [ModernBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForTokenClassification) (ModernBERT model)
  + [MptConfig](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptConfig) configuration class: [MptForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForTokenClassification) (MPT model)
  + [MraConfig](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraConfig) configuration class: [MraForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForTokenClassification) (MRA model)
  + [NemotronConfig](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronConfig) configuration class: [NemotronForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronForTokenClassification) (Nemotron model)
  + [NezhaConfig](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig) configuration class: [NezhaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForTokenClassification) (Nezha model)
  + [NystromformerConfig](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig) configuration class: [NystromformerForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForTokenClassification) (Nyströmformer model)
  + [PersimmonConfig](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonConfig) configuration class: [PersimmonForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonForTokenClassification) (Persimmon model)
  + [Phi3Config](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Config) configuration class: [Phi3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3ForTokenClassification) (Phi3 model)
  + [PhiConfig](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiConfig) configuration class: [PhiForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiForTokenClassification) (Phi model)
  + [QDQBertConfig](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig) configuration class: [QDQBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForTokenClassification) (QDQBert model)
  + [Qwen2Config](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Config) configuration class: [Qwen2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForTokenClassification) (Qwen2 model)
  + [Qwen2MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeConfig) configuration class: [Qwen2MoeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeForTokenClassification) (Qwen2MoE model)
  + [Qwen3Config](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Config) configuration class: [Qwen3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForTokenClassification) (Qwen3 model)
  + [Qwen3MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeConfig) configuration class: [Qwen3MoeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForTokenClassification) (Qwen3MoE model)
  + [RemBertConfig](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertConfig) configuration class: [RemBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForTokenClassification) (RemBERT model)
  + [RoCBertConfig](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig) configuration class: [RoCBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForTokenClassification) (RoCBert model)
  + [RoFormerConfig](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig) configuration class: [RoFormerForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForTokenClassification) (RoFormer model)
  + [RobertaConfig](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig) configuration class: [RobertaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForTokenClassification) (RoBERTa model)
  + [RobertaPreLayerNormConfig](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig) configuration class: [RobertaPreLayerNormForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForTokenClassification) (RoBERTa-PreLayerNorm model)
  + [SeedOssConfig](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssConfig) configuration class: [SeedOssForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForTokenClassification) (SeedOss model)
  + [SmolLM3Config](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Config) configuration class: [SmolLM3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForTokenClassification) (SmolLM3 model)
  + [SqueezeBertConfig](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig) configuration class: [SqueezeBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForTokenClassification) (SqueezeBERT model)
  + [StableLmConfig](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmConfig) configuration class: [StableLmForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmForTokenClassification) (StableLm model)
  + [Starcoder2Config](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2Config) configuration class: [Starcoder2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2ForTokenClassification) (Starcoder2 model)
  + [T5Config](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config) configuration class: [T5ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForTokenClassification) (T5 model)
  + [T5GemmaConfig](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig) configuration class: [T5GemmaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForTokenClassification) (T5Gemma model)
  + [UMT5Config](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config) configuration class: [UMT5ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForTokenClassification) (UMT5 model)
  + [XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig) configuration class: [XLMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForTokenClassification) (XLM model)
  + [XLMRobertaConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig) configuration class: [XLMRobertaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForTokenClassification) (XLM-RoBERTa model)
  + [XLMRobertaXLConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig) configuration class: [XLMRobertaXLForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForTokenClassification) (XLM-RoBERTa-XL model)
  + [XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig) configuration class: [XLNetForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForTokenClassification) (XLNet model)
  + [XmodConfig](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig) configuration class: [XmodForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForTokenClassification) (X-MOD model)
  + [YosoConfig](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig) configuration class: [YosoForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForTokenClassification) (YOSO model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a token classification head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForTokenClassification

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForTokenClassification.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a token classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **albert** — `AlbertForTokenClassification` (ALBERT model)
* **apertus** — [ApertusForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusForTokenClassification) (Apertus model)
* **arcee** — [ArceeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForTokenClassification) (Arcee model)
* **bert** — [BertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForTokenClassification) (BERT model)
* **big\_bird** — [BigBirdForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForTokenClassification) (BigBird model)
* **biogpt** — [BioGptForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForTokenClassification) (BioGpt model)
* **bloom** — [BloomForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForTokenClassification) (BLOOM model)
* **bros** — [BrosForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/bros#transformers.BrosForTokenClassification) (BROS model)
* **camembert** — [CamembertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForTokenClassification) (CamemBERT model)
* **canine** — [CanineForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineForTokenClassification) (CANINE model)
* **convbert** — [ConvBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForTokenClassification) (ConvBERT model)
* **data2vec-text** — [Data2VecTextForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForTokenClassification) (Data2VecText model)
* **deberta** — [DebertaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForTokenClassification) (DeBERTa model)
* **deberta-v2** — [DebertaV2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForTokenClassification) (DeBERTa-v2 model)
* **diffllama** — [DiffLlamaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForTokenClassification) (DiffLlama model)
* **distilbert** — [DistilBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForTokenClassification) (DistilBERT model)
* **electra** — [ElectraForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForTokenClassification) (ELECTRA model)
* **ernie** — [ErnieForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForTokenClassification) (ERNIE model)
* **ernie\_m** — [ErnieMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMForTokenClassification) (ErnieM model)
* **esm** — [EsmForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForTokenClassification) (ESM model)
* **exaone4** — [Exaone4ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForTokenClassification) (EXAONE-4.0 model)
* **falcon** — [FalconForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForTokenClassification) (Falcon model)
* **flaubert** — [FlaubertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertForTokenClassification) (FlauBERT model)
* **fnet** — [FNetForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForTokenClassification) (FNet model)
* **funnel** — [FunnelForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForTokenClassification) (Funnel Transformer model)
* **gemma** — [GemmaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gemma#transformers.GemmaForTokenClassification) (Gemma model)
* **gemma2** — [Gemma2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2ForTokenClassification) (Gemma2 model)
* **glm** — [GlmForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmForTokenClassification) (GLM model)
* **glm4** — [Glm4ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4ForTokenClassification) (GLM4 model)
* **gpt-sw3** — [GPT2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForTokenClassification) (GPT-Sw3 model)
* **gpt2** — [GPT2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForTokenClassification) (OpenAI GPT-2 model)
* **gpt\_bigcode** — [GPTBigCodeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeForTokenClassification) (GPTBigCode model)
* **gpt\_neo** — [GPTNeoForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForTokenClassification) (GPT Neo model)
* **gpt\_neox** — [GPTNeoXForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXForTokenClassification) (GPT NeoX model)
* **gpt\_oss** — [GptOssForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssForTokenClassification) (GptOss model)
* **helium** — [HeliumForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/helium#transformers.HeliumForTokenClassification) (Helium model)
* **ibert** — [IBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForTokenClassification) (I-BERT model)
* **layoutlm** — [LayoutLMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForTokenClassification) (LayoutLM model)
* **layoutlmv2** — [LayoutLMv2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForTokenClassification) (LayoutLMv3 model)
* **lilt** — [LiltForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForTokenClassification) (LiLT model)
* **llama** — [LlamaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForTokenClassification) (LLaMA model)
* **longformer** — [LongformerForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForTokenClassification) (Longformer model)
* **luke** — [LukeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForTokenClassification) (LUKE model)
* **markuplm** — [MarkupLMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForTokenClassification) (MarkupLM model)
* **mega** — [MegaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForTokenClassification) (MEGA model)
* **megatron-bert** — [MegatronBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForTokenClassification) (Megatron-BERT model)
* **minimax** — [MiniMaxForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxForTokenClassification) (MiniMax model)
* **mistral** — [MistralForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForTokenClassification) (Mistral model)
* **mixtral** — [MixtralForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralForTokenClassification) (Mixtral model)
* **mobilebert** — [MobileBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForTokenClassification) (MobileBERT model)
* **modernbert** — [ModernBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForTokenClassification) (ModernBERT model)
* **mpnet** — [MPNetForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForTokenClassification) (MPNet model)
* **mpt** — [MptForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForTokenClassification) (MPT model)
* **mra** — [MraForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForTokenClassification) (MRA model)
* **mt5** — [MT5ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForTokenClassification) (MT5 model)
* **nemotron** — [NemotronForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronForTokenClassification) (Nemotron model)
* **nezha** — [NezhaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForTokenClassification) (Nezha model)
* **nystromformer** — [NystromformerForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForTokenClassification) (Nyströmformer model)
* **persimmon** — [PersimmonForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonForTokenClassification) (Persimmon model)
* **phi** — [PhiForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiForTokenClassification) (Phi model)
* **phi3** — [Phi3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3ForTokenClassification) (Phi3 model)
* **qdqbert** — [QDQBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForTokenClassification) (QDQBert model)
* **qwen2** — [Qwen2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForTokenClassification) (Qwen2 model)
* **qwen2\_moe** — [Qwen2MoeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeForTokenClassification) (Qwen2MoE model)
* **qwen3** — [Qwen3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForTokenClassification) (Qwen3 model)
* **qwen3\_moe** — [Qwen3MoeForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForTokenClassification) (Qwen3MoE model)
* **rembert** — [RemBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForTokenClassification) (RemBERT model)
* **roberta** — [RobertaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForTokenClassification) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaPreLayerNormForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForTokenClassification) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForTokenClassification) (RoCBert model)
* **roformer** — [RoFormerForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForTokenClassification) (RoFormer model)
* **seed\_oss** — [SeedOssForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForTokenClassification) (SeedOss model)
* **smollm3** — [SmolLM3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForTokenClassification) (SmolLM3 model)
* **squeezebert** — [SqueezeBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForTokenClassification) (SqueezeBERT model)
* **stablelm** — [StableLmForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmForTokenClassification) (StableLm model)
* **starcoder2** — [Starcoder2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/starcoder2#transformers.Starcoder2ForTokenClassification) (Starcoder2 model)
* **t5** — [T5ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForTokenClassification) (T5 model)
* **t5gemma** — [T5GemmaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForTokenClassification) (T5Gemma model)
* **umt5** — [UMT5ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForTokenClassification) (UMT5 model)
* **xlm** — [XLMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForTokenClassification) (XLM model)
* **xlm-roberta** — [XLMRobertaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForTokenClassification) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaXLForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForTokenClassification) (XLM-RoBERTa-XL model)
* **xlnet** — [XLNetForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForTokenClassification) (XLNet model)
* **xmod** — [XmodForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForTokenClassification) (X-MOD model)
* **yoso** — [YosoForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForTokenClassification) (YOSO model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForTokenClassification

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForTokenClassification.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForTokenClassification.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForTokenClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForQuestionAnswering

### class transformers.AutoModelForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1964)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a question answering head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [AlbertConfig](/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertConfig) configuration class: `AlbertForQuestionAnswering` (ALBERT model)
  + [ArceeConfig](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeConfig) configuration class: [ArceeForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForQuestionAnswering) (Arcee model)
  + [BartConfig](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartConfig) configuration class: [BartForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForQuestionAnswering) (BART model)
  + [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig) configuration class: [BertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForQuestionAnswering) (BERT model)
  + [BigBirdConfig](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig) configuration class: [BigBirdForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForQuestionAnswering) (BigBird model)
  + [BigBirdPegasusConfig](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig) configuration class: [BigBirdPegasusForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForQuestionAnswering) (BigBird-Pegasus model)
  + [BloomConfig](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig) configuration class: [BloomForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForQuestionAnswering) (BLOOM model)
  + [CamembertConfig](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig) configuration class: [CamembertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForQuestionAnswering) (CamemBERT model)
  + [CanineConfig](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineConfig) configuration class: [CanineForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineForQuestionAnswering) (CANINE model)
  + [ConvBertConfig](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertConfig) configuration class: [ConvBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForQuestionAnswering) (ConvBERT model)
  + [Data2VecTextConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig) configuration class: [Data2VecTextForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForQuestionAnswering) (Data2VecText model)
  + [DebertaConfig](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaConfig) configuration class: [DebertaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForQuestionAnswering) (DeBERTa model)
  + [DebertaV2Config](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config) configuration class: [DebertaV2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForQuestionAnswering) (DeBERTa-v2 model)
  + [DiffLlamaConfig](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaConfig) configuration class: [DiffLlamaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForQuestionAnswering) (DiffLlama model)
  + [DistilBertConfig](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertConfig) configuration class: [DistilBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForQuestionAnswering) (DistilBERT model)
  + [ElectraConfig](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig) configuration class: [ElectraForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForQuestionAnswering) (ELECTRA model)
  + [ErnieConfig](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig) configuration class: [ErnieForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForQuestionAnswering) (ERNIE model)
  + [ErnieMConfig](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMConfig) configuration class: [ErnieMForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMForQuestionAnswering) (ErnieM model)
  + [Exaone4Config](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config) configuration class: [Exaone4ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForQuestionAnswering) (EXAONE-4.0 model)
  + [FNetConfig](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig) configuration class: [FNetForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForQuestionAnswering) (FNet model)
  + [FalconConfig](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig) configuration class: [FalconForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForQuestionAnswering) (Falcon model)
  + [FlaubertConfig](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertConfig) configuration class: [FlaubertForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertForQuestionAnsweringSimple) (FlauBERT model)
  + [FunnelConfig](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelConfig) configuration class: [FunnelForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForQuestionAnswering) (Funnel Transformer model)
  + [GPT2Config](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config) configuration class: [GPT2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForQuestionAnswering) (OpenAI GPT-2 model)
  + [GPTJConfig](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJConfig) configuration class: [GPTJForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJForQuestionAnswering) (GPT-J model)
  + [GPTNeoConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig) configuration class: [GPTNeoForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForQuestionAnswering) (GPT Neo model)
  + [GPTNeoXConfig](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXConfig) configuration class: [GPTNeoXForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXForQuestionAnswering) (GPT NeoX model)
  + [IBertConfig](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertConfig) configuration class: [IBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForQuestionAnswering) (I-BERT model)
  + [LEDConfig](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDConfig) configuration class: [LEDForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDForQuestionAnswering) (LED model)
  + [LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config) configuration class: [LayoutLMv2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) (LayoutLMv2 model)
  + [LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config) configuration class: [LayoutLMv3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering) (LayoutLMv3 model)
  + [LiltConfig](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltConfig) configuration class: [LiltForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForQuestionAnswering) (LiLT model)
  + [LlamaConfig](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig) configuration class: [LlamaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForQuestionAnswering) (LLaMA model)
  + [LongformerConfig](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerConfig) configuration class: [LongformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForQuestionAnswering) (Longformer model)
  + [LukeConfig](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig) configuration class: [LukeForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForQuestionAnswering) (LUKE model)
  + [LxmertConfig](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertConfig) configuration class: [LxmertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering) (LXMERT model)
  + [MBartConfig](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig) configuration class: [MBartForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForQuestionAnswering) (mBART model)
  + [MPNetConfig](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig) configuration class: [MPNetForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForQuestionAnswering) (MPNet model)
  + [MT5Config](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config) configuration class: [MT5ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForQuestionAnswering) (MT5 model)
  + [MarkupLMConfig](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMConfig) configuration class: [MarkupLMForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForQuestionAnswering) (MarkupLM model)
  + [MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig) configuration class: [MegaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForQuestionAnswering) (MEGA model)
  + [MegatronBertConfig](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig) configuration class: [MegatronBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForQuestionAnswering) (Megatron-BERT model)
  + [MiniMaxConfig](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxConfig) configuration class: [MiniMaxForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxForQuestionAnswering) (MiniMax model)
  + [MistralConfig](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig) configuration class: [MistralForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForQuestionAnswering) (Mistral model)
  + [MixtralConfig](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralConfig) configuration class: [MixtralForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralForQuestionAnswering) (Mixtral model)
  + [MobileBertConfig](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig) configuration class: [MobileBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForQuestionAnswering) (MobileBERT model)
  + [ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig) configuration class: [ModernBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForQuestionAnswering) (ModernBERT model)
  + [MptConfig](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptConfig) configuration class: [MptForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForQuestionAnswering) (MPT model)
  + [MraConfig](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraConfig) configuration class: [MraForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForQuestionAnswering) (MRA model)
  + [MvpConfig](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig) configuration class: [MvpForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForQuestionAnswering) (MVP model)
  + [NemotronConfig](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronConfig) configuration class: [NemotronForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronForQuestionAnswering) (Nemotron model)
  + [NezhaConfig](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig) configuration class: [NezhaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForQuestionAnswering) (Nezha model)
  + [NystromformerConfig](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig) configuration class: [NystromformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForQuestionAnswering) (Nyströmformer model)
  + [OPTConfig](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig) configuration class: [OPTForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForQuestionAnswering) (OPT model)
  + [QDQBertConfig](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig) configuration class: [QDQBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForQuestionAnswering) (QDQBert model)
  + [Qwen2Config](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Config) configuration class: [Qwen2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForQuestionAnswering) (Qwen2 model)
  + [Qwen2MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeConfig) configuration class: [Qwen2MoeForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeForQuestionAnswering) (Qwen2MoE model)
  + [Qwen3Config](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Config) configuration class: [Qwen3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForQuestionAnswering) (Qwen3 model)
  + [Qwen3MoeConfig](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeConfig) configuration class: [Qwen3MoeForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForQuestionAnswering) (Qwen3MoE model)
  + [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig) configuration class: [ReformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForQuestionAnswering) (Reformer model)
  + [RemBertConfig](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertConfig) configuration class: [RemBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForQuestionAnswering) (RemBERT model)
  + [RoCBertConfig](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig) configuration class: [RoCBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForQuestionAnswering) (RoCBert model)
  + [RoFormerConfig](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig) configuration class: [RoFormerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForQuestionAnswering) (RoFormer model)
  + [RobertaConfig](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig) configuration class: [RobertaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForQuestionAnswering) (RoBERTa model)
  + [RobertaPreLayerNormConfig](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig) configuration class: [RobertaPreLayerNormForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForQuestionAnswering) (RoBERTa-PreLayerNorm model)
  + [SeedOssConfig](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssConfig) configuration class: [SeedOssForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForQuestionAnswering) (SeedOss model)
  + [SmolLM3Config](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Config) configuration class: [SmolLM3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForQuestionAnswering) (SmolLM3 model)
  + [SplinterConfig](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterConfig) configuration class: [SplinterForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterForQuestionAnswering) (Splinter model)
  + [SqueezeBertConfig](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig) configuration class: [SqueezeBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForQuestionAnswering) (SqueezeBERT model)
  + [T5Config](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config) configuration class: [T5ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForQuestionAnswering) (T5 model)
  + [UMT5Config](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config) configuration class: [UMT5ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForQuestionAnswering) (UMT5 model)
  + [XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig) configuration class: [XLMForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForQuestionAnsweringSimple) (XLM model)
  + [XLMRobertaConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig) configuration class: [XLMRobertaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForQuestionAnswering) (XLM-RoBERTa model)
  + [XLMRobertaXLConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig) configuration class: [XLMRobertaXLForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForQuestionAnswering) (XLM-RoBERTa-XL model)
  + [XLNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig) configuration class: [XLNetForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple) (XLNet model)
  + [XmodConfig](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig) configuration class: [XmodForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForQuestionAnswering) (X-MOD model)
  + [YosoConfig](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig) configuration class: [YosoForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForQuestionAnswering) (YOSO model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a question answering head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForQuestionAnswering

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForQuestionAnswering.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a question answering head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **albert** — `AlbertForQuestionAnswering` (ALBERT model)
* **arcee** — [ArceeForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForQuestionAnswering) (Arcee model)
* **bart** — [BartForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForQuestionAnswering) (BART model)
* **bert** — [BertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForQuestionAnswering) (BERT model)
* **big\_bird** — [BigBirdForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForQuestionAnswering) (BigBird model)
* **bigbird\_pegasus** — [BigBirdPegasusForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForQuestionAnswering) (BigBird-Pegasus model)
* **bloom** — [BloomForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForQuestionAnswering) (BLOOM model)
* **camembert** — [CamembertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForQuestionAnswering) (CamemBERT model)
* **canine** — [CanineForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/canine#transformers.CanineForQuestionAnswering) (CANINE model)
* **convbert** — [ConvBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/convbert#transformers.ConvBertForQuestionAnswering) (ConvBERT model)
* **data2vec-text** — [Data2VecTextForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForQuestionAnswering) (Data2VecText model)
* **deberta** — [DebertaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaForQuestionAnswering) (DeBERTa model)
* **deberta-v2** — [DebertaV2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForQuestionAnswering) (DeBERTa-v2 model)
* **diffllama** — [DiffLlamaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForQuestionAnswering) (DiffLlama model)
* **distilbert** — [DistilBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/distilbert#transformers.DistilBertForQuestionAnswering) (DistilBERT model)
* **electra** — [ElectraForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForQuestionAnswering) (ELECTRA model)
* **ernie** — [ErnieForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForQuestionAnswering) (ERNIE model)
* **ernie\_m** — [ErnieMForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/ernie_m#transformers.ErnieMForQuestionAnswering) (ErnieM model)
* **exaone4** — [Exaone4ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForQuestionAnswering) (EXAONE-4.0 model)
* **falcon** — [FalconForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForQuestionAnswering) (Falcon model)
* **flaubert** — [FlaubertForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/flaubert#transformers.FlaubertForQuestionAnsweringSimple) (FlauBERT model)
* **fnet** — [FNetForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForQuestionAnswering) (FNet model)
* **funnel** — [FunnelForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/funnel#transformers.FunnelForQuestionAnswering) (Funnel Transformer model)
* **gpt2** — [GPT2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForQuestionAnswering) (OpenAI GPT-2 model)
* **gpt\_neo** — [GPTNeoForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForQuestionAnswering) (GPT Neo model)
* **gpt\_neox** — [GPTNeoXForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXForQuestionAnswering) (GPT NeoX model)
* **gptj** — [GPTJForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/gptj#transformers.GPTJForQuestionAnswering) (GPT-J model)
* **ibert** — [IBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/ibert#transformers.IBertForQuestionAnswering) (I-BERT model)
* **layoutlmv2** — [LayoutLMv2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering) (LayoutLMv3 model)
* **led** — [LEDForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/led#transformers.LEDForQuestionAnswering) (LED model)
* **lilt** — [LiltForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForQuestionAnswering) (LiLT model)
* **llama** — [LlamaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForQuestionAnswering) (LLaMA model)
* **longformer** — [LongformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/longformer#transformers.LongformerForQuestionAnswering) (Longformer model)
* **luke** — [LukeForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForQuestionAnswering) (LUKE model)
* **lxmert** — [LxmertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering) (LXMERT model)
* **markuplm** — [MarkupLMForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForQuestionAnswering) (MarkupLM model)
* **mbart** — [MBartForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForQuestionAnswering) (mBART model)
* **mega** — [MegaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForQuestionAnswering) (MEGA model)
* **megatron-bert** — [MegatronBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForQuestionAnswering) (Megatron-BERT model)
* **minimax** — [MiniMaxForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/minimax#transformers.MiniMaxForQuestionAnswering) (MiniMax model)
* **mistral** — [MistralForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForQuestionAnswering) (Mistral model)
* **mixtral** — [MixtralForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mixtral#transformers.MixtralForQuestionAnswering) (Mixtral model)
* **mobilebert** — [MobileBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForQuestionAnswering) (MobileBERT model)
* **modernbert** — [ModernBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForQuestionAnswering) (ModernBERT model)
* **mpnet** — [MPNetForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForQuestionAnswering) (MPNet model)
* **mpt** — [MptForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mpt#transformers.MptForQuestionAnswering) (MPT model)
* **mra** — [MraForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mra#transformers.MraForQuestionAnswering) (MRA model)
* **mt5** — [MT5ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForQuestionAnswering) (MT5 model)
* **mvp** — [MvpForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForQuestionAnswering) (MVP model)
* **nemotron** — [NemotronForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronForQuestionAnswering) (Nemotron model)
* **nezha** — [NezhaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForQuestionAnswering) (Nezha model)
* **nystromformer** — [NystromformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForQuestionAnswering) (Nyströmformer model)
* **opt** — [OPTForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForQuestionAnswering) (OPT model)
* **qdqbert** — [QDQBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForQuestionAnswering) (QDQBert model)
* **qwen2** — [Qwen2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForQuestionAnswering) (Qwen2 model)
* **qwen2\_moe** — [Qwen2MoeForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeForQuestionAnswering) (Qwen2MoE model)
* **qwen3** — [Qwen3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForQuestionAnswering) (Qwen3 model)
* **qwen3\_moe** — [Qwen3MoeForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForQuestionAnswering) (Qwen3MoE model)
* **reformer** — [ReformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForQuestionAnswering) (Reformer model)
* **rembert** — [RemBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/rembert#transformers.RemBertForQuestionAnswering) (RemBERT model)
* **roberta** — [RobertaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForQuestionAnswering) (RoBERTa model)
* **roberta-prelayernorm** — [RobertaPreLayerNormForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForQuestionAnswering) (RoBERTa-PreLayerNorm model)
* **roc\_bert** — [RoCBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForQuestionAnswering) (RoCBert model)
* **roformer** — [RoFormerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForQuestionAnswering) (RoFormer model)
* **seed\_oss** — [SeedOssForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForQuestionAnswering) (SeedOss model)
* **smollm3** — [SmolLM3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForQuestionAnswering) (SmolLM3 model)
* **splinter** — [SplinterForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/splinter#transformers.SplinterForQuestionAnswering) (Splinter model)
* **squeezebert** — [SqueezeBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForQuestionAnswering) (SqueezeBERT model)
* **t5** — [T5ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForQuestionAnswering) (T5 model)
* **umt5** — [UMT5ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForQuestionAnswering) (UMT5 model)
* **xlm** — [XLMForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForQuestionAnsweringSimple) (XLM model)
* **xlm-roberta** — [XLMRobertaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForQuestionAnswering) (XLM-RoBERTa model)
* **xlm-roberta-xl** — [XLMRobertaXLForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForQuestionAnswering) (XLM-RoBERTa-XL model)
* **xlnet** — [XLNetForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple) (XLNet model)
* **xmod** — [XmodForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForQuestionAnswering) (X-MOD model)
* **yoso** — [YosoForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForQuestionAnswering) (YOSO model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForQuestionAnswering

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForQuestionAnswering.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForTextEncoding

### class transformers.AutoModelForTextEncoding

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1890)

( \*args \*\*kwargs  )

## Computer vision

The following auto classes are available for the following computer vision tasks.

### AutoModelForDepthEstimation

### class transformers.AutoModelForDepthEstimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2102)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a depth estimation head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [DPTConfig](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTConfig) configuration class: [DPTForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForDepthEstimation) (DPT model)
  + [DepthAnythingConfig](/docs/transformers/v4.56.2/en/model_doc/depth_anything#transformers.DepthAnythingConfig) configuration class: [DepthAnythingForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/depth_anything#transformers.DepthAnythingForDepthEstimation) (Depth Anything model)
  + [DepthProConfig](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProConfig) configuration class: [DepthProForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProForDepthEstimation) (DepthPro model)
  + [GLPNConfig](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNConfig) configuration class: [GLPNForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNForDepthEstimation) (GLPN model)
  + [PromptDepthAnythingConfig](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingConfig) configuration class: [PromptDepthAnythingForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingForDepthEstimation) (PromptDepthAnything model)
  + [ZoeDepthConfig](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthConfig) configuration class: [ZoeDepthForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthForDepthEstimation) (ZoeDepth model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a depth estimation head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForDepthEstimation

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForDepthEstimation.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a depth estimation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **depth\_anything** — [DepthAnythingForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/depth_anything#transformers.DepthAnythingForDepthEstimation) (Depth Anything model)
* **depth\_pro** — [DepthProForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/depth_pro#transformers.DepthProForDepthEstimation) (DepthPro model)
* **dpt** — [DPTForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForDepthEstimation) (DPT model)
* **glpn** — [GLPNForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNForDepthEstimation) (GLPN model)
* **prompt\_depth\_anything** — [PromptDepthAnythingForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingForDepthEstimation) (PromptDepthAnything model)
* **zoedepth** — [ZoeDepthForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthForDepthEstimation) (ZoeDepth model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForDepthEstimation

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForDepthEstimation.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForDepthEstimation.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForDepthEstimation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForImageClassification

### class transformers.AutoModelForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2027)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a image classification head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [BeitConfig](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitConfig) configuration class: [BeitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitForImageClassification) (BEiT model)
  + [BitConfig](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitConfig) configuration class: [BitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitForImageClassification) (BiT model)
  + [CLIPConfig](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPConfig) configuration class: [CLIPForImageClassification](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPForImageClassification) (CLIP model)
  + [ConvNextConfig](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextConfig) configuration class: [ConvNextForImageClassification](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextForImageClassification) (ConvNeXT model)
  + [ConvNextV2Config](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Config) configuration class: [ConvNextV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2ForImageClassification) (ConvNeXTV2 model)
  + [CvtConfig](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtConfig) configuration class: [CvtForImageClassification](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtForImageClassification) (CvT model)
  + [Data2VecVisionConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig) configuration class: [Data2VecVisionForImageClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionForImageClassification) (Data2VecVision model)
  + [DeiTConfig](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTConfig) configuration class: [DeiTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTForImageClassification) or [DeiTForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTForImageClassificationWithTeacher) (DeiT model)
  + [DinatConfig](/docs/transformers/v4.56.2/en/model_doc/dinat#transformers.DinatConfig) configuration class: [DinatForImageClassification](/docs/transformers/v4.56.2/en/model_doc/dinat#transformers.DinatForImageClassification) (DiNAT model)
  + [Dinov2Config](/docs/transformers/v4.56.2/en/model_doc/dinov2#transformers.Dinov2Config) configuration class: [Dinov2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/dinov2#transformers.Dinov2ForImageClassification) (DINOv2 model)
  + [Dinov2WithRegistersConfig](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersConfig) configuration class: [Dinov2WithRegistersForImageClassification](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersForImageClassification) (DINOv2 with Registers model)
  + [DonutSwinConfig](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinConfig) configuration class: [DonutSwinForImageClassification](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinForImageClassification) (DonutSwin model)
  + [EfficientFormerConfig](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerConfig) configuration class: [EfficientFormerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerForImageClassification) or [EfficientFormerForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerForImageClassificationWithTeacher) (EfficientFormer model)
  + [EfficientNetConfig](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetConfig) configuration class: [EfficientNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetForImageClassification) (EfficientNet model)
  + [FocalNetConfig](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetConfig) configuration class: [FocalNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetForImageClassification) (FocalNet model)
  + [HGNetV2Config](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Config) configuration class: [HGNetV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2ForImageClassification) (HGNet-V2 model)
  + [HieraConfig](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraConfig) configuration class: [HieraForImageClassification](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraForImageClassification) (Hiera model)
  + [IJepaConfig](/docs/transformers/v4.56.2/en/model_doc/ijepa#transformers.IJepaConfig) configuration class: [IJepaForImageClassification](/docs/transformers/v4.56.2/en/model_doc/ijepa#transformers.IJepaForImageClassification) (I-JEPA model)
  + [ImageGPTConfig](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig) configuration class: [ImageGPTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification) (ImageGPT model)
  + [LevitConfig](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitConfig) configuration class: [LevitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassification) or [LevitForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher) (LeViT model)
  + [MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config) configuration class: [MetaClip2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2ForImageClassification) (MetaCLIP 2 model)
  + [MobileNetV1Config](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1Config) configuration class: [MobileNetV1ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1ForImageClassification) (MobileNetV1 model)
  + [MobileNetV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2Config) configuration class: [MobileNetV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2ForImageClassification) (MobileNetV2 model)
  + [MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig) configuration class: [MobileViTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForImageClassification) (MobileViT model)
  + [MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config) configuration class: [MobileViTV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2ForImageClassification) (MobileViTV2 model)
  + [NatConfig](/docs/transformers/v4.56.2/en/model_doc/nat#transformers.NatConfig) configuration class: [NatForImageClassification](/docs/transformers/v4.56.2/en/model_doc/nat#transformers.NatForImageClassification) (NAT model)
  + [PerceiverConfig](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverConfig) configuration class: [PerceiverForImageClassificationLearned](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) or [PerceiverForImageClassificationFourier](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationFourier) or [PerceiverForImageClassificationConvProcessing](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationConvProcessing) (Perceiver model)
  + [PoolFormerConfig](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerConfig) configuration class: [PoolFormerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerForImageClassification) (PoolFormer model)
  + [PvtConfig](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtConfig) configuration class: [PvtForImageClassification](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtForImageClassification) (PVT model)
  + [PvtV2Config](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Config) configuration class: [PvtV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2ForImageClassification) (PVTv2 model)
  + [RegNetConfig](/docs/transformers/v4.56.2/en/model_doc/regnet#transformers.RegNetConfig) configuration class: [RegNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/regnet#transformers.RegNetForImageClassification) (RegNet model)
  + [ResNetConfig](/docs/transformers/v4.56.2/en/model_doc/resnet#transformers.ResNetConfig) configuration class: [ResNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/resnet#transformers.ResNetForImageClassification) (ResNet model)
  + [SegformerConfig](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerConfig) configuration class: [SegformerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForImageClassification) (SegFormer model)
  + [ShieldGemma2Config](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2Config) configuration class: [ShieldGemma2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2ForImageClassification) (Shieldgemma2 model)
  + [Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config) configuration class: [Siglip2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ForImageClassification) (SigLIP2 model)
  + [SiglipConfig](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipConfig) configuration class: [SiglipForImageClassification](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipForImageClassification) (SigLIP model)
  + [SwiftFormerConfig](/docs/transformers/v4.56.2/en/model_doc/swiftformer#transformers.SwiftFormerConfig) configuration class: [SwiftFormerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/swiftformer#transformers.SwiftFormerForImageClassification) (SwiftFormer model)
  + [SwinConfig](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinConfig) configuration class: [SwinForImageClassification](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinForImageClassification) (Swin Transformer model)
  + [Swinv2Config](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2Config) configuration class: [Swinv2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2ForImageClassification) (Swin Transformer V2 model)
  + [TextNetConfig](/docs/transformers/v4.56.2/en/model_doc/textnet#transformers.TextNetConfig) configuration class: [TextNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/textnet#transformers.TextNetForImageClassification) (TextNet model)
  + [TimmWrapperConfig](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperConfig) configuration class: [TimmWrapperForImageClassification](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperForImageClassification) (TimmWrapperModel model)
  + [VanConfig](/docs/transformers/v4.56.2/en/model_doc/van#transformers.VanConfig) configuration class: [VanForImageClassification](/docs/transformers/v4.56.2/en/model_doc/van#transformers.VanForImageClassification) (VAN model)
  + [ViTConfig](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTConfig) configuration class: [ViTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTForImageClassification) (ViT model)
  + [ViTHybridConfig](/docs/transformers/v4.56.2/en/model_doc/vit_hybrid#transformers.ViTHybridConfig) configuration class: [ViTHybridForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit_hybrid#transformers.ViTHybridForImageClassification) (ViT Hybrid model)
  + [ViTMSNConfig](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNConfig) configuration class: [ViTMSNForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNForImageClassification) (ViTMSN model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a image classification head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForImageClassification

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForImageClassification.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a image classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **beit** — [BeitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitForImageClassification) (BEiT model)
* **bit** — [BitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/bit#transformers.BitForImageClassification) (BiT model)
* **clip** — [CLIPForImageClassification](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPForImageClassification) (CLIP model)
* **convnext** — [ConvNextForImageClassification](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextForImageClassification) (ConvNeXT model)
* **convnextv2** — [ConvNextV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2ForImageClassification) (ConvNeXTV2 model)
* **cvt** — [CvtForImageClassification](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtForImageClassification) (CvT model)
* **data2vec-vision** — [Data2VecVisionForImageClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionForImageClassification) (Data2VecVision model)
* **deit** — [DeiTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTForImageClassification) or [DeiTForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTForImageClassificationWithTeacher) (DeiT model)
* **dinat** — [DinatForImageClassification](/docs/transformers/v4.56.2/en/model_doc/dinat#transformers.DinatForImageClassification) (DiNAT model)
* **dinov2** — [Dinov2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/dinov2#transformers.Dinov2ForImageClassification) (DINOv2 model)
* **dinov2\_with\_registers** — [Dinov2WithRegistersForImageClassification](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersForImageClassification) (DINOv2 with Registers model)
* **donut-swin** — [DonutSwinForImageClassification](/docs/transformers/v4.56.2/en/model_doc/donut#transformers.DonutSwinForImageClassification) (DonutSwin model)
* **efficientformer** — [EfficientFormerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerForImageClassification) or [EfficientFormerForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/efficientformer#transformers.EfficientFormerForImageClassificationWithTeacher) (EfficientFormer model)
* **efficientnet** — [EfficientNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetForImageClassification) (EfficientNet model)
* **focalnet** — [FocalNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetForImageClassification) (FocalNet model)
* **hgnet\_v2** — [HGNetV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2ForImageClassification) (HGNet-V2 model)
* **hiera** — [HieraForImageClassification](/docs/transformers/v4.56.2/en/model_doc/hiera#transformers.HieraForImageClassification) (Hiera model)
* **ijepa** — [IJepaForImageClassification](/docs/transformers/v4.56.2/en/model_doc/ijepa#transformers.IJepaForImageClassification) (I-JEPA model)
* **imagegpt** — [ImageGPTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification) (ImageGPT model)
* **levit** — [LevitForImageClassification](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassification) or [LevitForImageClassificationWithTeacher](/docs/transformers/v4.56.2/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher) (LeViT model)
* **metaclip\_2** — [MetaClip2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2ForImageClassification) (MetaCLIP 2 model)
* **mobilenet\_v1** — [MobileNetV1ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1ForImageClassification) (MobileNetV1 model)
* **mobilenet\_v2** — [MobileNetV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2ForImageClassification) (MobileNetV2 model)
* **mobilevit** — [MobileViTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForImageClassification) (MobileViT model)
* **mobilevitv2** — [MobileViTV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2ForImageClassification) (MobileViTV2 model)
* **nat** — [NatForImageClassification](/docs/transformers/v4.56.2/en/model_doc/nat#transformers.NatForImageClassification) (NAT model)
* **perceiver** — [PerceiverForImageClassificationLearned](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) or [PerceiverForImageClassificationFourier](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationFourier) or [PerceiverForImageClassificationConvProcessing](/docs/transformers/v4.56.2/en/model_doc/perceiver#transformers.PerceiverForImageClassificationConvProcessing) (Perceiver model)
* **poolformer** — [PoolFormerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/poolformer#transformers.PoolFormerForImageClassification) (PoolFormer model)
* **pvt** — [PvtForImageClassification](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtForImageClassification) (PVT model)
* **pvt\_v2** — [PvtV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2ForImageClassification) (PVTv2 model)
* **regnet** — [RegNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/regnet#transformers.RegNetForImageClassification) (RegNet model)
* **resnet** — [ResNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/resnet#transformers.ResNetForImageClassification) (ResNet model)
* **segformer** — [SegformerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForImageClassification) (SegFormer model)
* **shieldgemma2** — [ShieldGemma2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2ForImageClassification) (Shieldgemma2 model)
* **siglip** — [SiglipForImageClassification](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipForImageClassification) (SigLIP model)
* **siglip2** — [Siglip2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2ForImageClassification) (SigLIP2 model)
* **swiftformer** — [SwiftFormerForImageClassification](/docs/transformers/v4.56.2/en/model_doc/swiftformer#transformers.SwiftFormerForImageClassification) (SwiftFormer model)
* **swin** — [SwinForImageClassification](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinForImageClassification) (Swin Transformer model)
* **swinv2** — [Swinv2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2ForImageClassification) (Swin Transformer V2 model)
* **textnet** — [TextNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/textnet#transformers.TextNetForImageClassification) (TextNet model)
* **timm\_wrapper** — [TimmWrapperForImageClassification](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperForImageClassification) (TimmWrapperModel model)
* **van** — [VanForImageClassification](/docs/transformers/v4.56.2/en/model_doc/van#transformers.VanForImageClassification) (VAN model)
* **vit** — [ViTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTForImageClassification) (ViT model)
* **vit\_hybrid** — [ViTHybridForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit_hybrid#transformers.ViTHybridForImageClassification) (ViT Hybrid model)
* **vit\_msn** — [ViTMSNForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNForImageClassification) (ViTMSN model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForImageClassification

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForImageClassification.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForImageClassification.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForImageClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForVideoClassification

### class transformers.AutoModelForVideoClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2109)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a video classification head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [TimesformerConfig](/docs/transformers/v4.56.2/en/model_doc/timesformer#transformers.TimesformerConfig) configuration class: [TimesformerForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/timesformer#transformers.TimesformerForVideoClassification) (TimeSformer model)
  + [VJEPA2Config](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2Config) configuration class: [VJEPA2ForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2ForVideoClassification) (VJEPA2Model model)
  + [VideoMAEConfig](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEConfig) configuration class: [VideoMAEForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEForVideoClassification) (VideoMAE model)
  + [VivitConfig](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitConfig) configuration class: [VivitForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitForVideoClassification) (ViViT model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a video classification head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForVideoClassification

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForVideoClassification.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a video classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **timesformer** — [TimesformerForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/timesformer#transformers.TimesformerForVideoClassification) (TimeSformer model)
* **videomae** — [VideoMAEForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/videomae#transformers.VideoMAEForVideoClassification) (VideoMAE model)
* **vivit** — [VivitForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/vivit#transformers.VivitForVideoClassification) (ViViT model)
* **vjepa2** — [VJEPA2ForVideoClassification](/docs/transformers/v4.56.2/en/model_doc/vjepa2#transformers.VJEPA2ForVideoClassification) (VJEPA2Model model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForVideoClassification

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForVideoClassification.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForVideoClassification.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForVideoClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForKeypointDetection

### class transformers.AutoModelForKeypointDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1882)

( \*args \*\*kwargs  )

### AutoModelForKeypointMatching

### class transformers.AutoModelForKeypointMatching

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1886)

( \*args \*\*kwargs  )

### AutoModelForMaskedImageModeling

### class transformers.AutoModelForMaskedImageModeling

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2192)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a masked image modeling head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [DeiTConfig](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTConfig) configuration class: [DeiTForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTForMaskedImageModeling) (DeiT model)
  + [FocalNetConfig](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetConfig) configuration class: [FocalNetForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetForMaskedImageModeling) (FocalNet model)
  + [SwinConfig](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinConfig) configuration class: [SwinForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinForMaskedImageModeling) (Swin Transformer model)
  + [Swinv2Config](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2Config) configuration class: [Swinv2ForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2ForMaskedImageModeling) (Swin Transformer V2 model)
  + [ViTConfig](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTConfig) configuration class: [ViTForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTForMaskedImageModeling) (ViT model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a masked image modeling head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForMaskedImageModeling

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForMaskedImageModeling.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a masked image modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **deit** — [DeiTForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/deit#transformers.DeiTForMaskedImageModeling) (DeiT model)
* **focalnet** — [FocalNetForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/focalnet#transformers.FocalNetForMaskedImageModeling) (FocalNet model)
* **swin** — [SwinForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/swin#transformers.SwinForMaskedImageModeling) (Swin Transformer model)
* **swinv2** — [Swinv2ForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/swinv2#transformers.Swinv2ForMaskedImageModeling) (Swin Transformer V2 model)
* **vit** — [ViTForMaskedImageModeling](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTForMaskedImageModeling) (ViT model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForMaskedImageModeling

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForMaskedImageModeling.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForMaskedImageModeling.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForMaskedImageModeling.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForObjectDetection

### class transformers.AutoModelForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2086)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a object detection head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [ConditionalDetrConfig](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig) configuration class: [ConditionalDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection) (Conditional DETR model)
  + [DFineConfig](/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineConfig) configuration class: [DFineForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineForObjectDetection) (D-FINE model)
  + [DabDetrConfig](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrConfig) configuration class: [DabDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrForObjectDetection) (DAB-DETR model)
  + [DeformableDetrConfig](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrConfig) configuration class: [DeformableDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) (Deformable DETR model)
  + [DetaConfig](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaConfig) configuration class: [DetaForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaForObjectDetection) (DETA model)
  + [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig) configuration class: [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) (DETR model)
  + [RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig) configuration class: [RTDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrForObjectDetection) (RT-DETR model)
  + [RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config) configuration class: [RTDetrV2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2ForObjectDetection) (RT-DETRv2 model)
  + [TableTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/table-transformer#transformers.TableTransformerConfig) configuration class: [TableTransformerForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/table-transformer#transformers.TableTransformerForObjectDetection) (Table Transformer model)
  + [YolosConfig](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosConfig) configuration class: [YolosForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosForObjectDetection) (YOLOS model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a object detection head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForObjectDetection

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForObjectDetection.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a object detection head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **conditional\_detr** — [ConditionalDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection) (Conditional DETR model)
* **d\_fine** — [DFineForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineForObjectDetection) (D-FINE model)
* **dab-detr** — [DabDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrForObjectDetection) (DAB-DETR model)
* **deformable\_detr** — [DeformableDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) (Deformable DETR model)
* **deta** — [DetaForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaForObjectDetection) (DETA model)
* **detr** — [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) (DETR model)
* **rt\_detr** — [RTDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrForObjectDetection) (RT-DETR model)
* **rt\_detr\_v2** — [RTDetrV2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2ForObjectDetection) (RT-DETRv2 model)
* **table-transformer** — [TableTransformerForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/table-transformer#transformers.TableTransformerForObjectDetection) (Table Transformer model)
* **yolos** — [YolosForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosForObjectDetection) (YOLOS model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForObjectDetection

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForObjectDetection.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForObjectDetection.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForObjectDetection.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForImageSegmentation

### class transformers.AutoModelForImageSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2043)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a image segmentation head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig) configuration class: [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) (DETR model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a image segmentation head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForImageSegmentation

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForImageSegmentation.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a image segmentation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **detr** — [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) (DETR model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForImageSegmentation

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForImageSegmentation.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForImageSegmentation.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForImageSegmentation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForImageToImage

### class transformers.AutoModelForImageToImage

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1894)

( \*args \*\*kwargs  )

### AutoModelForSemanticSegmentation

### class transformers.AutoModelForSemanticSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2050)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a semantic segmentation head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [BeitConfig](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitConfig) configuration class: [BeitForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitForSemanticSegmentation) (BEiT model)
  + [DPTConfig](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTConfig) configuration class: [DPTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) (DPT model)
  + [Data2VecVisionConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig) configuration class: [Data2VecVisionForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionForSemanticSegmentation) (Data2VecVision model)
  + [MobileNetV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2Config) configuration class: [MobileNetV2ForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2ForSemanticSegmentation) (MobileNetV2 model)
  + [MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig) configuration class: [MobileViTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) (MobileViT model)
  + [MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config) configuration class: [MobileViTV2ForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2ForSemanticSegmentation) (MobileViTV2 model)
  + [SegformerConfig](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerConfig) configuration class: [SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) (SegFormer model)
  + [UperNetConfig](/docs/transformers/v4.56.2/en/model_doc/upernet#transformers.UperNetConfig) configuration class: [UperNetForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation) (UPerNet model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a semantic segmentation head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForSemanticSegmentation

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForSemanticSegmentation.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a semantic segmentation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **beit** — [BeitForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitForSemanticSegmentation) (BEiT model)
* **data2vec-vision** — [Data2VecVisionForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionForSemanticSegmentation) (Data2VecVision model)
* **dpt** — [DPTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) (DPT model)
* **mobilenet\_v2** — [MobileNetV2ForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2ForSemanticSegmentation) (MobileNetV2 model)
* **mobilevit** — [MobileViTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) (MobileViT model)
* **mobilevitv2** — [MobileViTV2ForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2ForSemanticSegmentation) (MobileViTV2 model)
* **segformer** — [SegformerForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) (SegFormer model)
* **upernet** — [UperNetForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation) (UPerNet model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForSemanticSegmentation

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForSemanticSegmentation.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForSemanticSegmentation.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForSemanticSegmentation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForInstanceSegmentation

### class transformers.AutoModelForInstanceSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2077)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a instance segmentation head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [MaskFormerConfig](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig) configuration class: [MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) (MaskFormer model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a instance segmentation head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForInstanceSegmentation

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForInstanceSegmentation.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a instance segmentation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **maskformer** — [MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) (MaskFormer model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForInstanceSegmentation

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForInstanceSegmentation.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForInstanceSegmentation.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForInstanceSegmentation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForUniversalSegmentation

### class transformers.AutoModelForUniversalSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2068)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a universal image segmentation head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig) configuration class: [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) (DETR model)
  + [EomtConfig](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtConfig) configuration class: [EomtForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtForUniversalSegmentation) (EoMT model)
  + [Mask2FormerConfig](/docs/transformers/v4.56.2/en/model_doc/mask2former#transformers.Mask2FormerConfig) configuration class: [Mask2FormerForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/mask2former#transformers.Mask2FormerForUniversalSegmentation) (Mask2Former model)
  + [MaskFormerConfig](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig) configuration class: [MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) (MaskFormer model)
  + [OneFormerConfig](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerConfig) configuration class: [OneFormerForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerForUniversalSegmentation) (OneFormer model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a universal image segmentation head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForUniversalSegmentation

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForUniversalSegmentation.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a universal image segmentation head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **detr** — [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) (DETR model)
* **eomt** — [EomtForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/eomt#transformers.EomtForUniversalSegmentation) (EoMT model)
* **mask2former** — [Mask2FormerForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/mask2former#transformers.Mask2FormerForUniversalSegmentation) (Mask2Former model)
* **maskformer** — [MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) (MaskFormer model)
* **oneformer** — [OneFormerForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerForUniversalSegmentation) (OneFormer model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForUniversalSegmentation

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForUniversalSegmentation.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForUniversalSegmentation.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForUniversalSegmentation.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForZeroShotImageClassification

### class transformers.AutoModelForZeroShotImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2034)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a zero-shot image classification head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [AlignConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignConfig) configuration class: [AlignModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignModel) (ALIGN model)
  + [AltCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPConfig) configuration class: [AltCLIPModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPModel) (AltCLIP model)
  + [Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config) configuration class: [Blip2ForImageTextRetrieval](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2ForImageTextRetrieval) (BLIP-2 model)
  + [BlipConfig](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipConfig) configuration class: [BlipModel](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipModel) (BLIP model)
  + [CLIPConfig](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPConfig) configuration class: [CLIPModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPModel) (CLIP model)
  + [CLIPSegConfig](/docs/transformers/v4.56.2/en/model_doc/clipseg#transformers.CLIPSegConfig) configuration class: [CLIPSegModel](/docs/transformers/v4.56.2/en/model_doc/clipseg#transformers.CLIPSegModel) (CLIPSeg model)
  + [ChineseCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig) configuration class: [ChineseCLIPModel](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPModel) (Chinese-CLIP model)
  + [MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config) configuration class: [MetaClip2Model](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Model) (MetaCLIP 2 model)
  + [Siglip2Config](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Config) configuration class: [Siglip2Model](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Model) (SigLIP2 model)
  + [SiglipConfig](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipConfig) configuration class: [SiglipModel](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipModel) (SigLIP model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a zero-shot image classification head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForZeroShotImageClassification

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForZeroShotImageClassification.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a zero-shot image classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **align** — [AlignModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignModel) (ALIGN model)
* **altclip** — [AltCLIPModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPModel) (AltCLIP model)
* **blip** — [BlipModel](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipModel) (BLIP model)
* **blip-2** — [Blip2ForImageTextRetrieval](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2ForImageTextRetrieval) (BLIP-2 model)
* **chinese\_clip** — [ChineseCLIPModel](/docs/transformers/v4.56.2/en/model_doc/chinese_clip#transformers.ChineseCLIPModel) (Chinese-CLIP model)
* **clip** — [CLIPModel](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPModel) (CLIP model)
* **clipseg** — [CLIPSegModel](/docs/transformers/v4.56.2/en/model_doc/clipseg#transformers.CLIPSegModel) (CLIPSeg model)
* **metaclip\_2** — [MetaClip2Model](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Model) (MetaCLIP 2 model)
* **siglip** — [SiglipModel](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipModel) (SigLIP model)
* **siglip2** — [Siglip2Model](/docs/transformers/v4.56.2/en/model_doc/siglip2#transformers.Siglip2Model) (SigLIP2 model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForZeroShotImageClassification

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForZeroShotImageClassification.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForZeroShotImageClassification.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForZeroShotImageClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForZeroShotObjectDetection

### class transformers.AutoModelForZeroShotObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2093)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a zero-shot object detection head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [GroundingDinoConfig](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoConfig) configuration class: [GroundingDinoForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoForObjectDetection) (Grounding DINO model)
  + [MMGroundingDinoConfig](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoConfig) configuration class: [MMGroundingDinoForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoForObjectDetection) (MM Grounding DINO model)
  + [OmDetTurboConfig](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboConfig) configuration class: [OmDetTurboForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboForObjectDetection) (OmDet-Turbo model)
  + [OwlViTConfig](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTConfig) configuration class: [OwlViTForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) (OWL-ViT model)
  + [Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config) configuration class: [Owlv2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ForObjectDetection) (OWLv2 model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a zero-shot object detection head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForZeroShotObjectDetection

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForZeroShotObjectDetection.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a zero-shot object detection head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **grounding-dino** — [GroundingDinoForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoForObjectDetection) (Grounding DINO model)
* **mm-grounding-dino** — [MMGroundingDinoForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoForObjectDetection) (MM Grounding DINO model)
* **omdet-turbo** — [OmDetTurboForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboForObjectDetection) (OmDet-Turbo model)
* **owlv2** — [Owlv2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ForObjectDetection) (OWLv2 model)
* **owlvit** — [OwlViTForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) (OWL-ViT model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForZeroShotObjectDetection

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

## Audio

The following auto classes are available for the following audio tasks.

### AutoModelForAudioClassification

### class transformers.AutoModelForAudioClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2141)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio classification head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [ASTConfig](/docs/transformers/v4.56.2/en/model_doc/audio-spectrogram-transformer#transformers.ASTConfig) configuration class: [ASTForAudioClassification](/docs/transformers/v4.56.2/en/model_doc/audio-spectrogram-transformer#transformers.ASTForAudioClassification) (Audio Spectrogram Transformer model)
  + [Data2VecAudioConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig) configuration class: [Data2VecAudioForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForSequenceClassification) (Data2VecAudio model)
  + [HubertConfig](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertConfig) configuration class: [HubertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertForSequenceClassification) (Hubert model)
  + [SEWConfig](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWConfig) configuration class: [SEWForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWForSequenceClassification) (SEW model)
  + [SEWDConfig](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDConfig) configuration class: [SEWDForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDForSequenceClassification) (SEW-D model)
  + [UniSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechConfig) configuration class: [UniSpeechForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechForSequenceClassification) (UniSpeech model)
  + [UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig) configuration class: [UniSpeechSatForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForSequenceClassification) (UniSpeechSat model)
  + [Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig) configuration class: [Wav2Vec2BertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForSequenceClassification) (Wav2Vec2-BERT model)
  + [Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) configuration class: [Wav2Vec2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification) (Wav2Vec2 model)
  + [Wav2Vec2ConformerConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig) configuration class: [Wav2Vec2ConformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForSequenceClassification) (Wav2Vec2-Conformer model)
  + [WavLMConfig](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMConfig) configuration class: [WavLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMForSequenceClassification) (WavLM model)
  + [WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig) configuration class: [WhisperForAudioClassification](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForAudioClassification) (Whisper model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a audio classification head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForAudioClassification

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForAudioClassification.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a audio classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **audio-spectrogram-transformer** — [ASTForAudioClassification](/docs/transformers/v4.56.2/en/model_doc/audio-spectrogram-transformer#transformers.ASTForAudioClassification) (Audio Spectrogram Transformer model)
* **data2vec-audio** — [Data2VecAudioForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForSequenceClassification) (Data2VecAudio model)
* **hubert** — [HubertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertForSequenceClassification) (Hubert model)
* **sew** — [SEWForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWForSequenceClassification) (SEW model)
* **sew-d** — [SEWDForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDForSequenceClassification) (SEW-D model)
* **unispeech** — [UniSpeechForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechForSequenceClassification) (UniSpeech model)
* **unispeech-sat** — [UniSpeechSatForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForSequenceClassification) (UniSpeechSat model)
* **wav2vec2** — [Wav2Vec2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification) (Wav2Vec2 model)
* **wav2vec2-bert** — [Wav2Vec2BertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForSequenceClassification) (Wav2Vec2-BERT model)
* **wav2vec2-conformer** — [Wav2Vec2ConformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForSequenceClassification) (Wav2Vec2-Conformer model)
* **wavlm** — [WavLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMForSequenceClassification) (WavLM model)
* **whisper** — [WhisperForAudioClassification](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForAudioClassification) (Whisper model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForAudioClassification

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForAudioClassification.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForAudioClassification.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForAudioFrameClassification

### class transformers.AutoModelForAudioFrameClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2164)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio frame (token) classification head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [Data2VecAudioConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig) configuration class: [Data2VecAudioForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForAudioFrameClassification) (Data2VecAudio model)
  + [UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig) configuration class: [UniSpeechSatForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForAudioFrameClassification) (UniSpeechSat model)
  + [Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig) configuration class: [Wav2Vec2BertForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForAudioFrameClassification) (Wav2Vec2-BERT model)
  + [Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) configuration class: [Wav2Vec2ForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification) (Wav2Vec2 model)
  + [Wav2Vec2ConformerConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig) configuration class: [Wav2Vec2ConformerForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForAudioFrameClassification) (Wav2Vec2-Conformer model)
  + [WavLMConfig](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMConfig) configuration class: [WavLMForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMForAudioFrameClassification) (WavLM model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a audio frame (token) classification head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForAudioFrameClassification

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForAudioFrameClassification.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a audio frame (token) classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **data2vec-audio** — [Data2VecAudioForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForAudioFrameClassification) (Data2VecAudio model)
* **unispeech-sat** — [UniSpeechSatForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForAudioFrameClassification) (UniSpeechSat model)
* **wav2vec2** — [Wav2Vec2ForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification) (Wav2Vec2 model)
* **wav2vec2-bert** — [Wav2Vec2BertForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForAudioFrameClassification) (Wav2Vec2-BERT model)
* **wav2vec2-conformer** — [Wav2Vec2ConformerForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForAudioFrameClassification) (Wav2Vec2-Conformer model)
* **wavlm** — [WavLMForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMForAudioFrameClassification) (WavLM model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForAudioFrameClassification

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForAudioFrameClassification.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForAudioFrameClassification.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForAudioFrameClassification.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForCTC

### class transformers.AutoModelForCTC

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2148)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a connectionist temporal classification head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [Data2VecAudioConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig) configuration class: [Data2VecAudioForCTC](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForCTC) (Data2VecAudio model)
  + [HubertConfig](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertConfig) configuration class: [HubertForCTC](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertForCTC) (Hubert model)
  + [MCTCTConfig](/docs/transformers/v4.56.2/en/model_doc/mctct#transformers.MCTCTConfig) configuration class: [MCTCTForCTC](/docs/transformers/v4.56.2/en/model_doc/mctct#transformers.MCTCTForCTC) (M-CTC-T model)
  + [SEWConfig](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWConfig) configuration class: [SEWForCTC](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWForCTC) (SEW model)
  + [SEWDConfig](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDConfig) configuration class: [SEWDForCTC](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDForCTC) (SEW-D model)
  + [UniSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechConfig) configuration class: [UniSpeechForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechForCTC) (UniSpeech model)
  + [UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig) configuration class: [UniSpeechSatForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC) (UniSpeechSat model)
  + [Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig) configuration class: [Wav2Vec2BertForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForCTC) (Wav2Vec2-BERT model)
  + [Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) configuration class: [Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) (Wav2Vec2 model)
  + [Wav2Vec2ConformerConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig) configuration class: [Wav2Vec2ConformerForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForCTC) (Wav2Vec2-Conformer model)
  + [WavLMConfig](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMConfig) configuration class: [WavLMForCTC](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMForCTC) (WavLM model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a connectionist temporal classification head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForCTC

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForCTC.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a connectionist temporal classification head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **data2vec-audio** — [Data2VecAudioForCTC](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForCTC) (Data2VecAudio model)
* **hubert** — [HubertForCTC](/docs/transformers/v4.56.2/en/model_doc/hubert#transformers.HubertForCTC) (Hubert model)
* **mctct** — [MCTCTForCTC](/docs/transformers/v4.56.2/en/model_doc/mctct#transformers.MCTCTForCTC) (M-CTC-T model)
* **sew** — [SEWForCTC](/docs/transformers/v4.56.2/en/model_doc/sew#transformers.SEWForCTC) (SEW model)
* **sew-d** — [SEWDForCTC](/docs/transformers/v4.56.2/en/model_doc/sew-d#transformers.SEWDForCTC) (SEW-D model)
* **unispeech** — [UniSpeechForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech#transformers.UniSpeechForCTC) (UniSpeech model)
* **unispeech-sat** — [UniSpeechSatForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC) (UniSpeechSat model)
* **wav2vec2** — [Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) (Wav2Vec2 model)
* **wav2vec2-bert** — [Wav2Vec2BertForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForCTC) (Wav2Vec2-BERT model)
* **wav2vec2-conformer** — [Wav2Vec2ConformerForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForCTC) (Wav2Vec2-Conformer model)
* **wavlm** — [WavLMForCTC](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMForCTC) (WavLM model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForCTC

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForCTC.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForCTC.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForCTC.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForSpeechSeq2Seq

### class transformers.AutoModelForSpeechSeq2Seq

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2155)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [DiaConfig](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaConfig) configuration class: [DiaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaForConditionalGeneration) (Dia model)
  + [GraniteSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechConfig) configuration class: [GraniteSpeechForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechForConditionalGeneration) (GraniteSpeech model)
  + [KyutaiSpeechToTextConfig](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextConfig) configuration class: [KyutaiSpeechToTextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextForConditionalGeneration) (KyutaiSpeechToText model)
  + [MoonshineConfig](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig) configuration class: [MoonshineForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineForConditionalGeneration) (Moonshine model)
  + [Pop2PianoConfig](/docs/transformers/v4.56.2/en/model_doc/pop2piano#transformers.Pop2PianoConfig) configuration class: [Pop2PianoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration) (Pop2Piano model)
  + [SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig) configuration class: [SeamlessM4TForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForSpeechToText) (SeamlessM4T model)
  + [SeamlessM4Tv2Config](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Config) configuration class: [SeamlessM4Tv2ForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForSpeechToText) (SeamlessM4Tv2 model)
  + [Speech2TextConfig](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextConfig) configuration class: [Speech2TextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextForConditionalGeneration) (Speech2Text model)
  + [SpeechEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig) configuration class: [SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) (Speech Encoder decoder model)
  + [SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config) configuration class: [SpeechT5ForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToText) (SpeechT5 model)
  + [WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig) configuration class: [WhisperForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) (Whisper model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForSpeechSeq2Seq.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **dia** — [DiaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaForConditionalGeneration) (Dia model)
* **granite\_speech** — [GraniteSpeechForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechForConditionalGeneration) (GraniteSpeech model)
* **kyutai\_speech\_to\_text** — [KyutaiSpeechToTextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextForConditionalGeneration) (KyutaiSpeechToText model)
* **moonshine** — [MoonshineForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineForConditionalGeneration) (Moonshine model)
* **pop2piano** — [Pop2PianoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration) (Pop2Piano model)
* **seamless\_m4t** — [SeamlessM4TForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForSpeechToText) (SeamlessM4T model)
* **seamless\_m4t\_v2** — [SeamlessM4Tv2ForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForSpeechToText) (SeamlessM4Tv2 model)
* **speech-encoder-decoder** — [SpeechEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) (Speech Encoder decoder model)
* **speech\_to\_text** — [Speech2TextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextForConditionalGeneration) (Speech2Text model)
* **speecht5** — [SpeechT5ForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToText) (SpeechT5 model)
* **whisper** — [WhisperForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) (Whisper model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForSpeechSeq2Seq.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForSpeechSeq2Seq.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForSpeechSeq2Seq.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForAudioXVector

### class transformers.AutoModelForAudioXVector

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2173)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio retrieval via x-vector head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [Data2VecAudioConfig](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig) configuration class: [Data2VecAudioForXVector](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForXVector) (Data2VecAudio model)
  + [UniSpeechSatConfig](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig) configuration class: [UniSpeechSatForXVector](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForXVector) (UniSpeechSat model)
  + [Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig) configuration class: [Wav2Vec2BertForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForXVector) (Wav2Vec2-BERT model)
  + [Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config) configuration class: [Wav2Vec2ForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector) (Wav2Vec2 model)
  + [Wav2Vec2ConformerConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerConfig) configuration class: [Wav2Vec2ConformerForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForXVector) (Wav2Vec2-Conformer model)
  + [WavLMConfig](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMConfig) configuration class: [WavLMForXVector](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMForXVector) (WavLM model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a audio retrieval via x-vector head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForAudioXVector

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForAudioXVector.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a audio retrieval via x-vector head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **data2vec-audio** — [Data2VecAudioForXVector](/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForXVector) (Data2VecAudio model)
* **unispeech-sat** — [UniSpeechSatForXVector](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForXVector) (UniSpeechSat model)
* **wav2vec2** — [Wav2Vec2ForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector) (Wav2Vec2 model)
* **wav2vec2-bert** — [Wav2Vec2BertForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForXVector) (Wav2Vec2-BERT model)
* **wav2vec2-conformer** — [Wav2Vec2ConformerForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-conformer#transformers.Wav2Vec2ConformerForXVector) (Wav2Vec2-Conformer model)
* **wavlm** — [WavLMForXVector](/docs/transformers/v4.56.2/en/model_doc/wavlm#transformers.WavLMForXVector) (WavLM model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForAudioXVector

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForAudioXVector.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForAudioXVector.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForAudioXVector.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForTextToSpectrogram

### class transformers.AutoModelForTextToSpectrogram

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2177)

( \*args \*\*kwargs  )

### AutoModelForTextToWaveform

### class transformers.AutoModelForTextToWaveform

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2181)

( \*args \*\*kwargs  )

### AutoModelForAudioTokenization

### class transformers.AutoModelForAudioTokenization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2199)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio tokenization through codebooks head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [DacConfig](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacConfig) configuration class: [DacModel](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel) (DAC model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a audio tokenization through codebooks head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForAudioTokenization

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForAudioTokenization.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a audio tokenization through codebooks head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **dac** — [DacModel](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel) (DAC model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForAudioTokenization

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForAudioTokenization.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForAudioTokenization.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForAudioTokenization.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

## Multimodal

The following auto classes are available for the following multimodal tasks.

### AutoModelForTableQuestionAnswering

### class transformers.AutoModelForTableQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1971)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a table question answering head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) configuration class: [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering) (TAPAS model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a table question answering head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForTableQuestionAnswering

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = AutoModelForTableQuestionAnswering.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a table question answering head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **tapas** — [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering) (TAPAS model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForTableQuestionAnswering

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

>>> # Update configuration during loading
>>> model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/tapas_tf_model_config.json")
>>> model = AutoModelForTableQuestionAnswering.from_pretrained(
...     "./tf_model/tapas_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForDocumentQuestionAnswering

### class transformers.AutoModelForDocumentQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1993)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a document question answering head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [LayoutLMConfig](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig) configuration class: [LayoutLMForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForQuestionAnswering) (LayoutLM model)
  + [LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config) configuration class: [LayoutLMv2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) (LayoutLMv2 model)
  + [LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config) configuration class: [LayoutLMv3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering) (LayoutLMv3 model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a document question answering head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForDocumentQuestionAnswering

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3")
>>> model = AutoModelForDocumentQuestionAnswering.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a document question answering head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **layoutlm** — [LayoutLMForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForQuestionAnswering) (LayoutLM model)
* **layoutlmv2** — [LayoutLMv2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) (LayoutLMv2 model)
* **layoutlmv3** — [LayoutLMv3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering) (LayoutLMv3 model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForDocumentQuestionAnswering

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3")

>>> # Update configuration during loading
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/layoutlm_tf_model_config.json")
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained(
...     "./tf_model/layoutlm_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForVisualQuestionAnswering

### class transformers.AutoModelForVisualQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L1982)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a visual question answering head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config) configuration class: [Blip2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) (BLIP-2 model)
  + [BlipConfig](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipConfig) configuration class: [BlipForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipForQuestionAnswering) (BLIP model)
  + [ViltConfig](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltConfig) configuration class: [ViltForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForQuestionAnswering) (ViLT model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a visual question answering head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForVisualQuestionAnswering

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
>>> model = AutoModelForVisualQuestionAnswering.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a visual question answering head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **blip** — [BlipForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipForQuestionAnswering) (BLIP model)
* **blip-2** — [Blip2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) (BLIP-2 model)
* **vilt** — [ViltForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForQuestionAnswering) (ViLT model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForVisualQuestionAnswering

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForVisualQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

>>> # Update configuration during loading
>>> model = AutoModelForVisualQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/vilt_tf_model_config.json")
>>> model = AutoModelForVisualQuestionAnswering.from_pretrained(
...     "./tf_model/vilt_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

### AutoModelForVision2Seq

### class transformers.AutoModelForVision2Seq

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2230)

( \*args \*\*kwargs  )

### AutoModelForImageTextToText

### class transformers.AutoModelForImageTextToText

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2124)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a image-text-to-text modeling head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [AriaConfig](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig) configuration class: [AriaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaForConditionalGeneration) (Aria model)
  + [AyaVisionConfig](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig) configuration class: [AyaVisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionForConditionalGeneration) (AyaVision model)
  + [Blip2Config](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2Config) configuration class: [Blip2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) (BLIP-2 model)
  + [BlipConfig](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipConfig) configuration class: [BlipForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipForConditionalGeneration) (BLIP model)
  + [ChameleonConfig](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonConfig) configuration class: [ChameleonForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonForConditionalGeneration) (Chameleon model)
  + [Cohere2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/cohere2_vision#transformers.Cohere2VisionConfig) configuration class: [Cohere2VisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/cohere2_vision#transformers.Cohere2VisionForConditionalGeneration) (Cohere2Vision model)
  + [DeepseekVLConfig](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLConfig) configuration class: [DeepseekVLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLForConditionalGeneration) (DeepseekVL model)
  + [DeepseekVLHybridConfig](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridConfig) configuration class: [DeepseekVLHybridForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridForConditionalGeneration) (DeepseekVLHybrid model)
  + [Emu3Config](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3Config) configuration class: [Emu3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ForConditionalGeneration) (Emu3 model)
  + [EvollaConfig](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaConfig) configuration class: [EvollaForProteinText2Text](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaForProteinText2Text) (Evolla model)
  + [Florence2Config](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2Config) configuration class: [Florence2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2ForConditionalGeneration) (Florence2 model)
  + [FuyuConfig](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuConfig) configuration class: [FuyuForCausalLM](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuForCausalLM) (Fuyu model)
  + [Gemma3Config](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config) configuration class: [Gemma3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration) (Gemma3ForConditionalGeneration model)
  + [Gemma3nConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nConfig) configuration class: [Gemma3nForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForConditionalGeneration) (Gemma3nForConditionalGeneration model)
  + [GitConfig](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitConfig) configuration class: [GitForCausalLM](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitForCausalLM) (GIT model)
  + [Glm4vConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vConfig) configuration class: [Glm4vForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vForConditionalGeneration) (GLM4V model)
  + [Glm4vMoeConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeConfig) configuration class: [Glm4vMoeForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeForConditionalGeneration) (GLM4VMOE model)
  + [GotOcr2Config](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2Config) configuration class: [GotOcr2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ForConditionalGeneration) (GOT-OCR2 model)
  + [Idefics2Config](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Config) configuration class: [Idefics2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ForConditionalGeneration) (Idefics2 model)
  + [Idefics3Config](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3Config) configuration class: [Idefics3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3ForConditionalGeneration) (Idefics3 model)
  + [IdeficsConfig](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsConfig) configuration class: [IdeficsForVisionText2Text](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsForVisionText2Text) (IDEFICS model)
  + [InstructBlipConfig](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipConfig) configuration class: [InstructBlipForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration) (InstructBLIP model)
  + [InternVLConfig](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLConfig) configuration class: [InternVLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLForConditionalGeneration) (InternVL model)
  + [JanusConfig](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusConfig) configuration class: [JanusForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusForConditionalGeneration) (Janus model)
  + [Kosmos2Config](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2Config) configuration class: [Kosmos2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2ForConditionalGeneration) (KOSMOS-2 model)
  + [Kosmos2\_5Config](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5Config) configuration class: [Kosmos2\_5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5ForConditionalGeneration) (KOSMOS-2.5 model)
  + [Llama4Config](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Config) configuration class: [Llama4ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForConditionalGeneration) (Llama4 model)
  + [LlavaConfig](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaConfig) configuration class: [LlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaForConditionalGeneration) (LLaVa model)
  + [LlavaNextConfig](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextConfig) configuration class: [LlavaNextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextForConditionalGeneration) (LLaVA-NeXT model)
  + [LlavaNextVideoConfig](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoConfig) configuration class: [LlavaNextVideoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoForConditionalGeneration) (LLaVa-NeXT-Video model)
  + [LlavaOnevisionConfig](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionConfig) configuration class: [LlavaOnevisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionForConditionalGeneration) (LLaVA-Onevision model)
  + [Mistral3Config](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Config) configuration class: [Mistral3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration) (Mistral3 model)
  + [MllamaConfig](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig) configuration class: [MllamaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForConditionalGeneration) (Mllama model)
  + [Ovis2Config](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Config) configuration class: [Ovis2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ForConditionalGeneration) (Ovis2 model)
  + [PaliGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaConfig) configuration class: [PaliGemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration) (PaliGemma model)
  + [PerceptionLMConfig](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMConfig) configuration class: [PerceptionLMForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMForConditionalGeneration) (PerceptionLM model)
  + [Pix2StructConfig](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructConfig) configuration class: [Pix2StructForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructForConditionalGeneration) (Pix2Struct model)
  + [PixtralVisionConfig](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralVisionConfig) configuration class: [LlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaForConditionalGeneration) (Pixtral model)
  + [Qwen2VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLConfig) configuration class: [Qwen2VLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLForConditionalGeneration) (Qwen2VL model)
  + [Qwen2\_5\_VLConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLConfig) configuration class: [Qwen2\_5\_VLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLForConditionalGeneration) (Qwen2\_5\_VL model)
  + [ShieldGemma2Config](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2Config) configuration class: [Gemma3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration) (Shieldgemma2 model)
  + [SmolVLMConfig](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMConfig) configuration class: [SmolVLMForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMForConditionalGeneration) (SmolVLM model)
  + [UdopConfig](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopConfig) configuration class: [UdopForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopForConditionalGeneration) (UDOP model)
  + [VipLlavaConfig](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaConfig) configuration class: [VipLlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaForConditionalGeneration) (VipLlava model)
  + [VisionEncoderDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) configuration class: [VisionEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) (Vision Encoder decoder model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a image-text-to-text modeling head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForImageTextToText

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForImageTextToText.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a image-text-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **aria** — [AriaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaForConditionalGeneration) (Aria model)
* **aya\_vision** — [AyaVisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionForConditionalGeneration) (AyaVision model)
* **blip** — [BlipForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blip#transformers.BlipForConditionalGeneration) (BLIP model)
* **blip-2** — [Blip2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration) (BLIP-2 model)
* **chameleon** — [ChameleonForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/chameleon#transformers.ChameleonForConditionalGeneration) (Chameleon model)
* **cohere2\_vision** — [Cohere2VisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/cohere2_vision#transformers.Cohere2VisionForConditionalGeneration) (Cohere2Vision model)
* **deepseek\_vl** — [DeepseekVLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl#transformers.DeepseekVLForConditionalGeneration) (DeepseekVL model)
* **deepseek\_vl\_hybrid** — [DeepseekVLHybridForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/deepseek_vl_hybrid#transformers.DeepseekVLHybridForConditionalGeneration) (DeepseekVLHybrid model)
* **emu3** — [Emu3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/emu3#transformers.Emu3ForConditionalGeneration) (Emu3 model)
* **evolla** — [EvollaForProteinText2Text](/docs/transformers/v4.56.2/en/model_doc/evolla#transformers.EvollaForProteinText2Text) (Evolla model)
* **florence2** — [Florence2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/florence2#transformers.Florence2ForConditionalGeneration) (Florence2 model)
* **fuyu** — [FuyuForCausalLM](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuForCausalLM) (Fuyu model)
* **gemma3** — [Gemma3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration) (Gemma3ForConditionalGeneration model)
* **gemma3n** — [Gemma3nForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForConditionalGeneration) (Gemma3nForConditionalGeneration model)
* **git** — [GitForCausalLM](/docs/transformers/v4.56.2/en/model_doc/git#transformers.GitForCausalLM) (GIT model)
* **glm4v** — [Glm4vForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vForConditionalGeneration) (GLM4V model)
* **glm4v\_moe** — [Glm4vMoeForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeForConditionalGeneration) (GLM4VMOE model)
* **got\_ocr2** — [GotOcr2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/got_ocr2#transformers.GotOcr2ForConditionalGeneration) (GOT-OCR2 model)
* **idefics** — [IdeficsForVisionText2Text](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsForVisionText2Text) (IDEFICS model)
* **idefics2** — [Idefics2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ForConditionalGeneration) (Idefics2 model)
* **idefics3** — [Idefics3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics3#transformers.Idefics3ForConditionalGeneration) (Idefics3 model)
* **instructblip** — [InstructBlipForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration) (InstructBLIP model)
* **internvl** — [InternVLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/internvl#transformers.InternVLForConditionalGeneration) (InternVL model)
* **janus** — [JanusForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/janus#transformers.JanusForConditionalGeneration) (Janus model)
* **kosmos-2** — [Kosmos2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kosmos-2#transformers.Kosmos2ForConditionalGeneration) (KOSMOS-2 model)
* **kosmos-2.5** — [Kosmos2\_5ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kosmos2_5#transformers.Kosmos2_5ForConditionalGeneration) (KOSMOS-2.5 model)
* **llama4** — [Llama4ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForConditionalGeneration) (Llama4 model)
* **llava** — [LlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaForConditionalGeneration) (LLaVa model)
* **llava\_next** — [LlavaNextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granitevision#transformers.LlavaNextForConditionalGeneration) (LLaVA-NeXT model)
* **llava\_next\_video** — [LlavaNextVideoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_next_video#transformers.LlavaNextVideoForConditionalGeneration) (LLaVa-NeXT-Video model)
* **llava\_onevision** — [LlavaOnevisionForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionForConditionalGeneration) (LLaVA-Onevision model)
* **mistral3** — [Mistral3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration) (Mistral3 model)
* **mllama** — [MllamaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForConditionalGeneration) (Mllama model)
* **ovis2** — [Ovis2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ForConditionalGeneration) (Ovis2 model)
* **paligemma** — [PaliGemmaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration) (PaliGemma model)
* **perception\_lm** — [PerceptionLMForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMForConditionalGeneration) (PerceptionLM model)
* **pix2struct** — [Pix2StructForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/pix2struct#transformers.Pix2StructForConditionalGeneration) (Pix2Struct model)
* **pixtral** — [LlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llava#transformers.LlavaForConditionalGeneration) (Pixtral model)
* **qwen2\_5\_vl** — [Qwen2\_5\_VLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLForConditionalGeneration) (Qwen2\_5\_VL model)
* **qwen2\_vl** — [Qwen2VLForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLForConditionalGeneration) (Qwen2VL model)
* **shieldgemma2** — [Gemma3ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration) (Shieldgemma2 model)
* **smolvlm** — [SmolVLMForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMForConditionalGeneration) (SmolVLM model)
* **udop** — [UdopForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/udop#transformers.UdopForConditionalGeneration) (UDOP model)
* **vipllava** — [VipLlavaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/vipllava#transformers.VipLlavaForConditionalGeneration) (VipLlava model)
* **vision-encoder-decoder** — [VisionEncoderDecoderModel](/docs/transformers/v4.56.2/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) (Vision Encoder decoder model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForImageTextToText

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForImageTextToText.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForImageTextToText.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForImageTextToText.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

## Time Series

### AutoModelForTimeSeriesPrediction

### class transformers.AutoModelForTimeSeriesPrediction

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2059)

( \*args \*\*kwargs  )

This is a generic model class that will be instantiated as one of the model classes of the library (with a time-series prediction head) when created
with the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) class method or the [from\_config()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_config) class
method.

This class cannot be instantiated directly using `__init__()` (throws an error).

#### from\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L424)

( \*\*kwargs  )

Parameters

* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The model class to instantiate is selected based on the configuration class:
  + [TimesFmConfig](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmConfig) configuration class: [TimesFmModelForPrediction](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmModelForPrediction) (TimesFm model)
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

Instantiates one of the model classes of the library (with a time-series prediction head) from a configuration.

Note:
Loading a model from its configuration file does **not** load the model weights. It only affects the
model’s configuration. Use [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to load the model weights.

Examples:


```
>>> from transformers import AutoConfig, AutoModelForTimeSeriesPrediction

>>> # Download configuration from huggingface.co and cache.
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
>>> model = AutoModelForTimeSeriesPrediction.from_config(config)
```

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/auto_factory.py#L468)

( \*model\_args \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
* **model\_args** (additional positional arguments, *optional*) —
  Will be passed along to the underlying model `__init__()` method.
* **config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) —
  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:
  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (*dict[str, torch.Tensor]*, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (e.g., not try downloading the model).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.
* **trust\_remote\_code** (`bool`, *optional*, defaults to `False`) —
  Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
  should only be set to `True` for repositories you trust and in which you have read the code, as it will
  execute code present on the Hub on your local machine.
* **code\_revision** (`str`, *optional*, defaults to `"main"`) —
  The specific revision to use for the code on the Hub, if the code leaves in a different repository than
  the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based
  system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier
  allowed by git.
* **kwargs** (additional keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate one of the model classes of the library (with a time-series prediction head) from a pretrained model.

The model class to instantiate is selected based on the `model_type` property of the config object (either
passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it’s missing, by
falling back to using pattern matching on `pretrained_model_name_or_path`:

* **timesfm** — [TimesFmModelForPrediction](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmModelForPrediction) (TimesFm model)

The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
deactivated). To train the model, you should first set it back in training mode with `model.train()`

Examples:


```
>>> from transformers import AutoConfig, AutoModelForTimeSeriesPrediction

>>> # Download model and configuration from huggingface.co and cache.
>>> model = AutoModelForTimeSeriesPrediction.from_pretrained("google-bert/bert-base-cased")

>>> # Update configuration during loading
>>> model = AutoModelForTimeSeriesPrediction.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
>>> model.config.output_attentions
True

>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
>>> config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
>>> model = AutoModelForTimeSeriesPrediction.from_pretrained(
...     "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
... )
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/auto.md)
