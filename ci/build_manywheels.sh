#!/bin/bash
set -e -x

# Collect the pythons
pys=(/opt/python/*/bin)

# Filter out Python 2.7 and 3.4
pys=(${pys[@]//*27*/})
pys=(${pys[@]//*34*/})

# Compile wheels
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install -r /io/requirements.txt
    "${PYBIN}/pip" wheel /io/sdist/*.tar.gz -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/$package_name-*.whl; do
    auditwheel repair --plat $PLAT "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/python" -m pip install $package_name --no-index -f /io/wheelhouse
    "${PYBIN}/pytest" /io/tests
done
