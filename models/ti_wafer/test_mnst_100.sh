#!/usr/bin/env sh
./build/tools/caffe test --iterations=3 --model=models/ti_wafer/net_mnist_100-200-100_test.prototxt --weights=Data/ti_wafer_mnist_100-200-100_final_iter_20000.caffemodel
