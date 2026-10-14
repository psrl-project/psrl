from psrl.utils.converter.model_mappings import ParameterMapping, register_model


# NOTE(lhy): Megatron Bridge handles name transformation, so this mapping only
# carries bridged model metadata.
@register_model(["Megatron"])
class BridgedMegatronParameterMapping(ParameterMapping):
    """Parameter mapping for Megatron model after Megatron-Bridge."""

    def __init__(self, config):
        super().__init__(config)
        # NOTE(lhy): Keeping `lm_head` separate avoids sharing its sharding logic
        # with the embedding layer.
        self.original_tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)

    def disable_tie_word_embeddings(self):
        self.config.tie_word_embeddings = False

    def get_mappings(self):
        raise ValueError(
            "BridgedMegatronParameterMapping is not used for name transformation, please use Megatron-Bridge instead"
        )
