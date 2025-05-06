import pathlib

from setuptools import find_packages, setup


def get_version():
    """Gets the vmas version."""
    path = CWD / "vni" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version out in __init__.py")


CWD = pathlib.Path(__file__).absolute().parent


setup(
    name="epistemic_rl",
    version="0.1.0",
    description="Epistemic Reinforcement Learning",
    long_description=open("README.md").read(),
    url="https://github.com/Giovannibriglia/epistemic_rl",
    license="GPLv3",
    author="Giovanni Briglia",
    author_email="giovanni.briglia@unimore.it",
    packages=find_packages(),
    install_requires=["torch"],
    include_package_data=True,
)
