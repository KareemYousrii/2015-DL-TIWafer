#!/usr/bin/env sh

./build/tools/caffe train --solver=models/ti_wafer_outputNeurons/solver_mnist_25-50-25.prototxt
./build/tools/caffe train --solver=models/ti_wafer_outputNeurons/solver_mnist_25-100-50.prototxt
./build/tools/caffe train --solver=models/ti_wafer_outputNeurons/solver_mnist_50-100-25.prototxt
./build/tools/caffe train --solver=models/ti_wafer_outputNeurons/solver_mnist_50-100-50.prototxt
./build/tools/caffe train --solver=models/ti_wafer_outputNeurons/solver_mnist_50-100-75.prototxt
./build/tools/caffe train --solver=models/ti_wafer_outputNeurons/solver_mnist_75-100-50.prototxt
./build/tools/caffe train --solver=models/ti_wafer_outputNeurons/solver_mnist_75-150-75.prototxt
./build/tools/caffe train --solver=models/ti_wafer_outputNeurons/solver_mnist_100-200-100.prototxt
