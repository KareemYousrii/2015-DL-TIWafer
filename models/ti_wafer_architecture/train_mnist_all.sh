#!/usr/bin/env sh

./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_3-2.prototxt >> Data/output_mnist_3-2
./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_3-3.prototxt >> Data/output_mnist_3-3
./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_3-4.prototxt >> Data/output_mnist_3-4
./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_3-5.prototxt >> Data/output_mnist_3-5

./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_4-2.prototxt >> Data/output_mnist_4-2
./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_4-3.prototxt >> Data/output_mnist_4-3
./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_4-4.prototxt >> Data/output_mnist_4-4
./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_4-5.prototxt >> Data/output_mnist_4-5

./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_5-2.prototxt >> Data/output_mnist_5-2
./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_5-3.prototxt >> Data/output_mnist_5-3
./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_5-4.prototxt >> Data/output_mnist_5-4
./build/tools/caffe train --solver=models/ti_wafer/solver_mnist_5-5.prototxt >> Data/output_mnist_5-5
