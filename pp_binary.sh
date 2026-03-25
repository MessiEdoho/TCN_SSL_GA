#!/bin/bash

srun \
  --time=10-00:00:00 \
  --partition=cs \
  --cpus-per-task=8 \
  --job-name=preprocessing_binary \
  --pty bash -i