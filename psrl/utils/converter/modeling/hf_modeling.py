from psrl.utils.converter.model_mappings import ParameterMapping, register_model


@register_model(["HF", "HuggingFace"])
class HFParameterMapping(ParameterMapping):
    """
    Parameter mapping for HuggingFace and FSDP models.

    HF/FSDP state dicts are already in the target (HF) format, so no name
    transformation is needed. This mapping exists solely to carry model_info
    (num_heads, num_kv_heads, head_size) into the converter for QKV reshaping.
    """

    def get_mappings(self):
        """
        Return an empty mapping list.

        HF parameters are already in the target format. No name transformation
        is required.
        """
        return []
