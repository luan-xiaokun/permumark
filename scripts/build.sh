#!/bin/bash

# This script builds the C extension for the permumark package.

gcc -shared -fPIC -Wall -Wextra -O3 -o permumark/derangement/derangement.so permumark/derangement/derangement.c
