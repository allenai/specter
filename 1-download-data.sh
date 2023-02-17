#!/bin/bash

set -x
# download model
echo "Downloading archive"
wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz
# unpack model
echo "Unpacking archive"
tar -xzvf archive.tar.gz

