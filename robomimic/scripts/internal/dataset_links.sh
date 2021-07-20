#!/bin/bash

# lift - ph
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/image.hdf5

# lift - mh
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/mh/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/mh/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/mh/image.hdf5

# lift - mg
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/mg/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/mg/low_dim_sparse.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/mg/low_dim_dense.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/mg/image_sparse.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/mg/image_dense.hdf5


# can - ph
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/image.hdf5

# can - mh
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/mh/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/mh/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/mh/image.hdf5

# can - mg
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/mg/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/mg/low_dim_sparse.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/mg/low_dim_dense.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/mg/image_sparse.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/mg/image_dense.hdf5

# can - paired
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/image.hdf5


# square - ph
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/image.hdf5

# square - mh
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/mh/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/mh/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/mh/image.hdf5


# transport - ph
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/transport/ph/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/transport/ph/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/transport/ph/image.hdf5

# transport - mh
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/transport/mh/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/transport/mh/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/transport/mh/image.hdf5


# tool hang - ph
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/tool_hang/ph/demo.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/tool_hang/ph/low_dim.hdf5
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/tool_hang/ph/image.hdf5


# lift_real - ph
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift_real/ph/demo.hdf5

# can_real - ph
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/can_real/ph/demo.hdf5

# tool_hang_real
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/tool_hang_real/ph/demo.hdf5




# model zoo links
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/lift/bc_rnn/lift_ph_low_dim_epoch_1000_succ_100.pth
curl -O https://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/lift/bc_rnn/lift_ph_image_epoch_500_succ_100.pth

curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/can/bc_rnn/can_ph_low_dim_epoch_1150_succ_100.pth
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/can/bc_rnn/can_ph_image_epoch_300_succ_100.pth

curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/square/bc_rnn/square_ph_low_dim_epoch_1850_succ_84.pth
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/square/bc_rnn/square_ph_image_epoch_540_succ_78.pth

curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/transport/bc_rnn/transport_ph_low_dim_epoch_1000_succ_78.pth
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/transport/bc_rnn/transport_ph_image_epoch_580_succ_70.pth

curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/tool_hang/bc_rnn/tool_hang_ph_low_dim_epoch_2000_succ_14.pth
curl -O http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/tool_hang/bc_rnn/tool_hang_ph_image_epoch_440_succ_74.pth

