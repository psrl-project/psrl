from setuptools import find_packages, setup

setup(
    name="psrl",
    version="0.0.1",
    package_dir={"": "."},
    packages=find_packages(where="."),
    package_data={
        "psrl": [
            "trainer/config/*.yaml",
            "trainer/config/**/*.yaml",
            "eval/config/*.yaml",
            "eval/config/**/*.yaml",
        ],
    },
    include_package_data=True,
    entry_points={
        "hydra.searchpath": [
            "psrl = psrl.trainer.config.hydra_plugins.psrl_searchpath:PSRLSearchPathPlugin",
        ],
    },
)
