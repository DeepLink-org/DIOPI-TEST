# !/bin/bash
set -e

cd third_party/mmcv_diopi
export PYTHONPATH=${PWD}:$PYTHONPATH
cd tests/test_ops
python -m pytest test_active_rotated_filter.py
python -m pytest test_assign_score_withk.py
python -m pytest test_bbox.py
python -m pytest test_deform_roi_pool.py
python -m pytest test_knn.py
python -m pytest test_convex_iou.py
python -m pytest test_min_area_polygons.py
python -m pytest test_prroi_pool.py
python -m pytest test_chamfer_distance.py
python -m pytest test_border_align.py
cd ../../../../
