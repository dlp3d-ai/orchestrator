import os
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install


def compile_proto():
    """Common function for compiling proto files."""
    # Add protoc/bin to the front of environment variables
    protoc_path = os.path.join(os.path.dirname(__file__), "protoc", "bin")
    sys.path.insert(0, protoc_path)
    proto_files = [
        "orchestrator/data_structures/speech2motion_v3.proto",
        "orchestrator/data_structures/audio2face_v1.proto",
        "orchestrator/data_structures/orchestrator_v4.proto",
    ]
    try:
        for proto_file in proto_files:
            if os.path.exists(proto_file):
                print(f"Compiling proto file: {proto_file}")
                result = subprocess.run(
                    ["protoc", "--python_out=.", proto_file], check=True, capture_output=True, text=True
                )
                print(f"Successfully compiled proto file: {proto_file}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
            else:
                print(f"Warning: Proto file not found at {proto_file}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to compile proto file: {e.stderr}")
    except FileNotFoundError:
        print("Warning: protoc command not found. Please install protobuf compiler if you need to compile proto files.")
    except Exception as e:
        print(f"Warning: Unexpected error during proto compilation: {str(e)}")


class CustomBuildPy(build_py):
    """Custom build command that compiles proto files before building.

    This class extends the standard build_py command to automatically compile
    protocol buffer files before building the package.
    """

    def run(self):
        # First compile proto files
        compile_proto()
        # Then execute original build
        build_py.run(self)


class CustomInstall(install):
    """Custom install command that compiles proto files after installation.

    This class extends the standard install command to automatically compile
    protocol buffer files after installing the package.
    """

    def run(self):
        # First execute original installation
        install.run(self)
        # Then compile proto files
        compile_proto()


class CustomDevelop(develop):
    """Custom develop command that compiles proto files after development
    installation.

    This class extends the standard develop command to automatically compile
    protocol buffer files after installing the package in development mode.
    """

    def run(self):
        # First execute original development mode installation
        develop.run(self)
        # Then compile proto files
        compile_proto()


def readme():
    """Read and return the contents of README.md file.

    Returns:
        str: The content of the README.md file as a string.
    """
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


def get_version():
    """Extract version information from the version.py file.

    Returns:
        str: The version string defined in orchestrator/version.py.
    """
    version_file = "orchestrator/version.py"
    scope = {}
    with open(version_file, "r", encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"), scope)
    return scope["__version__"]


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file.

        Args:
            line (str): A single line from the requirements file.

        Yields:
            dict: Dictionary containing parsed package information with keys:
                - 'line': Original line content
                - 'package': Package name
                - 'version': Version specification tuple (operator, version)
                - 'platform_deps': Platform-specific dependencies (optional)
        """
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        """Parse a requirements file and yield package information.

        Args:
            fpath (str): Path to the requirements file to parse.

        Yields:
            dict: Dictionary containing parsed package information for each
                non-comment, non-empty line in the file.
        """
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        """Generate package requirement strings from the requirements file.

        Yields:
            str: Formatted package requirement string that can be used
                with pip or setuptools.
        """
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


setup(
    name="orchestrator",
    version=get_version(),
    description="",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    keywords="",
    url="https://gitlab.bj.sensetime.com/zeotrope/her/apis/orchestrator",
    packages=find_packages(exclude=("configs", "tools", "demo")),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    license="Apache License 2.0",
    install_requires=parse_requirements("requirements.txt"),
    zip_safe=False,
    cmdclass={
        "build_py": CustomBuildPy,
        "install": CustomInstall,
        "develop": CustomDevelop,
    },
)
