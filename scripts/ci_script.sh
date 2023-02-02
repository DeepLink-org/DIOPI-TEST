# !/bin/bash
set -e

case $1 in
  py-lint)
    (echo "py-lint" && flake8 --ignore=E501,F841 python/conformance/diopi_functions.py \
       && flake8 --ignore=E501,F401 --exclude=python/conformance/diopi_functions.py) \
    || exit -1;;
  cpp-lint)
    (echo "cpp-lint" && cpplint --linelength=160 --filter=-runtime/references,-legal/copyright \
      --filter=-runtime/printf,-runtime/int,-whitespace/indent,-build/namespaces --recursive impl/ \
      && cpplint --linelength=240 --filter=-build/header_guard --recursive diopirt/ ) \
    || exit -1;;
    *)
    echo -e "[ERROR] Incorrect option:" $1;

esac
exit 0