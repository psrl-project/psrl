from contextlib import contextmanager


def apply_tms_patch():
    from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

    _TAG_DEFAULT = "default"

    @contextmanager
    def _with_region_config_patch(self, tag: str, enable_cpu_backup: bool):
        # assert not self._binary_wrapper.cdll.tms_get_interesting_region()
        original_enable_cpu_backup = self._binary_wrapper.cdll.tms_get_enable_cpu_backup()
        original_interesting_region = self._binary_wrapper.cdll.tms_get_interesting_region()

        self._binary_wrapper.set_config(tag=tag, interesting_region=True, enable_cpu_backup=enable_cpu_backup)
        try:
            yield
        finally:
            assert self._binary_wrapper.cdll.tms_get_interesting_region(), (
                "TMS region must remain active while restoring its configuration."
            )
            self._binary_wrapper.set_config(
                tag=_TAG_DEFAULT,
                interesting_region=original_interesting_region,
                enable_cpu_backup=original_enable_cpu_backup,
            )

    _TorchMemorySaverImpl._with_region_config = _with_region_config_patch
