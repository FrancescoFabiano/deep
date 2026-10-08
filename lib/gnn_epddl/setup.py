import pathlib

from setuptools import find_packages, setup

CWD = pathlib.Path(__file__).absolute().parent


setup(
    name="gnn_epddl",
    version="0.1.0",
    description="Learned sibling ranker for the epistemic planner deep: instances, data, training, ONNX export",
    long_description=(CWD / "README.md").read_text(encoding="utf-8")
    if (CWD / "README.md").exists()
    else "Learned sibling ranker for the epistemic planner deep",
    license="GPLv3",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"gnn_epddl": ["vocab.json", "generators/templates/*.epddl"]},
    install_requires=["torch", "torch-geometric", "numpy", "onnx", "onnxruntime"],
    include_package_data=True,
)
