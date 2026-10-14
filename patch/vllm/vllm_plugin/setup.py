from setuptools import find_packages, setup

setup(
    name="vllm-psrl-patches",
    version="0.0.1",
    description="Clean vLLM modifications via the plugin system",
    packages=find_packages(),
    install_requires=[
        "packaging>=20.0",
    ],
    # Register with vLLM's plugin system
    entry_points={"vllm.general_plugins": ["custom_patches = vllm_patches:register_patches"]},
    python_requires=">=3.10",
)
