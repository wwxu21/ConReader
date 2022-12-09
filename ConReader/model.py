from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaAttention,
    RobertaPreTrainedModel,
    RobertaIntermediate,
    RobertaOutput,
    RobertaLayer,
)
from transformers.file_utils import (
    is_remote_url,
    cached_path,
    is_torch_tpu_available,
    hf_bucket_url,
)
from transformers.modeling_outputs import (
    ModelOutput
)
from transformers.modeling_utils import (
    TF_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    WEIGHTS_NAME,
    apply_chunking_to_forward,
)
from transformers.configuration_utils import PretrainedConfig
import os
from typing import Optional, Union, Tuple
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from torch import nn
from transformers.utils import logging
import re
logger = logging.get_logger(__name__)
from torch.nn import CrossEntropyLoss, MarginRankingLoss, MultiMarginLoss
from dataclasses import dataclass
class ContractQA(RobertaPreTrainedModel):
    def __init__(self, config,):
        super().__init__(config)
        self.config = config
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.model_c = RobertaModel(config)
        self.fusion_layer = RobertaLayer(config)
        self.t_element = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.t_cls = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        self.memory = torch.zeros(config.query_size + 1, config.msize, config.hidden_size * 2)  # 1 for padding
        self.memory_mask = torch.zeros(config.query_size + 1, config.msize, dtype=torch.long)
        self.query_size = config.query_size + 1
        self.msize = config.msize
        self.reserved = config.reserved
    def forward(
            self,
            definition_ids=None,
            definition_type_ids=None,
            definition_attention_mask=None,
            contract_definition_map=None,
            input_ids_c=None,
            attention_mask_c=None,
            token_type_ids_c=None,
            eval_ids=None,
            label_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions_c=None,
            end_positions_c=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions_c (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions_c (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, doc_num, doc_size = input_ids_c.size()
        total_loss = None
        if token_type_ids_c is not None:
            token_type_ids_c = token_type_ids_c.view([batch_size * doc_num, doc_size])
        else:
            token_type_ids_c = None

        outputs_c = self.model_c(
            input_ids_c.view([batch_size * doc_num, doc_size]),
            attention_mask=attention_mask_c.view([batch_size * doc_num, doc_size]),
            token_type_ids=token_type_ids_c,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs_c = outputs_c[0].view([batch_size, doc_num, doc_size, -1])
        outputs_cls = outputs_c[:, :, 0, :]
        # add context relation
        context = self.model_c.embeddings(inputs_embeds=outputs_cls)
        context = context.unsqueeze(1).expand([batch_size, doc_num, doc_num, -1])
        outputs_c[:, :, -self.reserved:-self.reserved + doc_num, :] = context
        attention_mask_c[:,:,-self.reserved:-self.reserved + doc_num] = 1
        attention_mask_c[:, range(doc_num), range(-self.reserved, -self.reserved + doc_num, 1)] = 0
        # add similar element relation
        if start_positions_c is not None and end_positions_c is not None and label_ids is not None:
            few_shot = nn.functional.embedding(label_ids, self.memory).view(batch_size * doc_num, self.msize, -1)
            few_shot_mask = nn.functional.embedding(label_ids, self.memory_mask).view(batch_size * doc_num, self.msize)
        else:
            few_shot = self.memory.view(self.query_size * self.msize, -1).expand(batch_size * doc_num, self.query_size * self.msize, -1)
            few_shot_mask = self.memory_mask.view(self.query_size * self.msize, -1).expand(batch_size * doc_num, self.query_size * self.msize, -1)
        query = self.t_cls(outputs_cls.view(batch_size * doc_num, -1))
        key = self.t_element(few_shot)
        sim = torch.cosine_similarity(query.unsqueeze(1), key, dim=2)
        sorted_sim, sorted_index = torch.sort(sim, dim=1, descending=True)
        few_shot_index = sorted_index[:,0]

        few_shot = few_shot[range(batch_size * doc_num), few_shot_index].view(batch_size, doc_num, 2, -1)
        few_shot_mask = few_shot_mask[range(batch_size * doc_num), few_shot_index].view(batch_size, doc_num)
        outputs_c[:, :, -self.reserved + doc_num + 1:-self.reserved + doc_num + 3, :] = few_shot
        attention_mask_c[:, :, -self.reserved + doc_num + 1] = few_shot_mask
        attention_mask_c[:, :, -self.reserved + doc_num + 2] = few_shot_mask
        # add term-explanation relation
        if contract_definition_map.size(2) != 0:
            output_definition = self.model_c(
                definition_ids,
                definition_attention_mask,
                definition_type_ids,
            )
            output_definition_cls = output_definition[0][:,0,:]
            output_definition_cls[0] = 0 # padding for non-definition word
            definition_enhancement = nn.functional.embedding(contract_definition_map, output_definition_cls)
            if definition_enhancement.size(2) >= self.reserved - doc_num - 3:
                outputs_c[:, :, -self.reserved + doc_num + 3:, :] = definition_enhancement[:,:,:self.reserved - doc_num - 3,:]
                attention_mask_c[:, :, -self.reserved + doc_num + 3:] = 1
            else:
                outputs_c[:, :, -self.reserved + doc_num + 3: -self.reserved + doc_num + 3 + definition_enhancement.size(2), :] = definition_enhancement
                attention_mask_c[:, :, -self.reserved + doc_num + 3: -self.reserved + doc_num + 3 + definition_enhancement.size(2)] = 1

        attention_mask_c = self.invert_attention_mask(attention_mask_c.view([batch_size * doc_num, doc_size]))
        fusion_output = self.fusion_layer(
            hidden_states=outputs_c.view([batch_size * doc_num, doc_size, -1]),
            attention_mask=attention_mask_c,
        )
        sequence_output = fusion_output[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions_c is not None and end_positions_c is not None and label_ids is not None:

            # If we are on multi-GPU, split add a dimension

            if len(start_positions_c.size()) > 1:
                start_positions_c = start_positions_c.view(-1)
                element_mask = start_positions_c.ne(0) #element exist
            if len(end_positions_c.size()) > 1:
                end_positions_c = end_positions_c.view(-1)
            if len(label_ids.size()) > 1:
                label_ids = label_ids.view(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions_c.clamp_(0, ignored_index)
            end_positions_c.clamp_(0, ignored_index)
            loss_mask = eval_ids.ne(-1).view(-1)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits[loss_mask], start_positions_c[loss_mask])
            end_loss = loss_fct(end_logits[loss_mask], end_positions_c[loss_mask])

            # add similar clauses relation
            margin_loss_fct = MultiMarginLoss()
            margin_loss = 0
            batch_start = sequence_output[range(batch_size * doc_num), start_positions_c][element_mask]
            batch_end = sequence_output[range(batch_size * doc_num), end_positions_c][element_mask]
            element_state = torch.cat([batch_start, batch_end], dim=1)
            if batch_start.size(0) != 0 and batch_end.size(0) != 0:
                memory_pos_mask = few_shot_mask.view(-1)[element_mask].eq(1)
                neg_label_ids = (torch.randint(self.memory.size(0),[1]).type_as(label_ids) + label_ids) % self.memory.size(0)
                few_shot_index_neg = few_shot_index
                memory_neg_mask = nn.functional.embedding(neg_label_ids, self.memory_mask)[range(batch_size * doc_num), few_shot_index_neg][element_mask].eq(1)
                memory_mask = memory_pos_mask * memory_neg_mask #memeory is not null
                if torch.sum(memory_mask) != 0:
                    query_ori = query[element_mask][memory_mask].clone().detach()
                    memory_pos = key[range(batch_size * doc_num), few_shot_index][element_mask][memory_mask]
                    memory_neg = self.t_element(nn.functional.embedding(neg_label_ids, self.memory)[range(batch_size * doc_num), few_shot_index_neg][element_mask][memory_mask])
                    memory_sim_pos = torch.cosine_similarity(query_ori, memory_pos, dim=1).unsqueeze(1)
                    memory_sim_neg = torch.cosine_similarity(query_ori, memory_neg, dim=1).unsqueeze(1)
                    memory_sim = torch.cat([memory_sim_pos, memory_sim_neg], dim=1)
                    margin_loss = margin_loss_fct(memory_sim, torch.zeros(memory_sim_pos.size(0)).type_as(start_positions_c))
                label_ids = label_ids[element_mask]
                label_unique = torch.unique(label_ids)
                for label_one in label_unique:
                    element_index_one = torch.where(label_ids == label_one)[0]
                    save_num = min(element_index_one.size(0), self.msize)
                    self.memory[label_one][:save_num] = element_state.clone().detach()[:save_num]
                    self.memory_mask[label_one][:save_num] = 1
                    temp = self.memory[label_one][save_num:].clone()
                    self.memory[label_one][self.msize - save_num:] = self.memory[label_one][:save_num].clone()
                    self.memory[label_one][:self.msize - save_num] = temp
                    temp = self.memory_mask[label_one][save_num:].clone()
                    self.memory_mask[label_one][self.msize - save_num:] = self.memory_mask[label_one][:save_num].clone()
                    self.memory_mask[label_one][:self.msize - save_num] = temp
            total_loss = (start_loss + end_loss) / 2 + margin_loss
        if not return_dict:
            output = (start_logits, end_logits)
            return ((total_loss,) + output) if total_loss is not None else output

        return ContractModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated). To
        train the model, you should first set it back in training mode with ``model.train()``.

        The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments ``config`` and ``state_dict``).
            model_args (sequence of positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            config (:obj:`Union[PretrainedConfig, str, os.PathLike]`, `optional`):
                Can be either:

                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                    - a string or path valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            state_dict (:obj:`Dict[str, torch.Tensor]`, `optional`):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~transformers.PreTrainedModel.save_pretrained` and
                :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            mirror(:obj:`str`, `optional`, defaults to :obj:`None`):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
                automatically loaded:

                    - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                      underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                      initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
                      ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's ``__init__`` function.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.

        Examples::

            >>> from transformers import BertConfig, BertModel
            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = BertModel.from_pretrained('bert-base-uncased')
            >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
            >>> model = BertModel.from_pretrained('./test/saved_model/')
            >>> # Update configuration during loading.
            >>> model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> assert model.config.output_attentions == True
            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
            >>> config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            >>> model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                )
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
                    f"at '{resolved_archive_file}'"
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module: nn.Module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    True,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ""
            model_to_load = model
            has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + "."
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)
            if start_prefix == "":
                load(model_to_load, prefix=start_prefix)
                model.memory_mask = torch.load(os.path.join(pretrained_model_name_or_path, "memory_mask.bin"))
                model.memory = torch.load(os.path.join(pretrained_model_name_or_path, "memory.bin"))
            else:
                load(model_to_load.model_c, prefix=start_prefix)
            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
                ]
                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

            # Some models may have keys that are not in the state by design, removing them before needlessly warning
            # the user.
            if cls._keys_to_ignore_on_load_missing is not None:
                for pat in cls._keys_to_ignore_on_load_missing:
                    missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

            if cls._keys_to_ignore_on_load_unexpected is not None:
                for pat in cls._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                    f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                    f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                    f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
                    f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                    f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
                )
            else:
                logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
            if len(missing_keys) > 0:
                logger.warning(
                    f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                    f"and are newly initialized: {missing_keys}\n"
                    f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
                )
            else:
                logger.info(
                    f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                    f"If your task is similar to the task the model of the checkpoint was trained on, "
                    f"you can already use {model.__class__.__name__} for predictions without further training."
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        if hasattr(config, "xla_device") and config.xla_device and is_torch_tpu_available():
            import torch_xla.core.xla_model as xm

            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model.to(xm.xla_device())

        return model

class FusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.is_decoder:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        batch_size, doc_num, doc_size, hidde_size = hidden_states.size()
        self_attention_outputs = self.attention(
            hidden_states.view([batch_size * doc_num, doc_size, hidde_size]),
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output.view([batch_size, doc_num * doc_size, hidde_size]),
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


@dataclass
class ContractModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    cache: tuple = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None