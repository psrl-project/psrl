from psrl.utils.converter.model_mappings import ParameterMapping, register_model


@register_model(["FSDP", "FSDP1", "FSDP2", "HSDP"])
class FSDPParameterMapping(ParameterMapping):
    """
    Parameter mapping for FSDP/FSDP2 models.

    FSDP state dicts are already in the target (HF) format, so no name
    transformation is needed. This mapping exists solely to carry model_info
    (num_heads, num_kv_heads, head_size) into the converter for QKV reshaping.
    """

    def get_mappings(self):
        """
        Return an empty mapping list.

        FSDP parameters are already in the HF equivalent format. No name
        transformation is required.
        """
        return []
