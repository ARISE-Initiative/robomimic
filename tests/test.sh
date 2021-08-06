#!/bin/bash

echo "running tests for bc..."
python test_bc.py
echo "running tests for hbc..."
python test_hbc.py
echo "running tests for iris..."
python test_iris.py
echo "running tests for bcq..."
python test_bcq.py
echo "running tests for cql..."
python test_cql.py
echo "running tests for scripts..."
python test_scripts.py
echo "running tests for examples..."
python test_examples.py
