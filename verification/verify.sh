#!/bin/bash
julia --color=yes ./verification/verify_MNIST.jl $1 $2 $3 $4 | tee -a ./verification/logs/$1.log
