#!/bin/bash

pytest livelike -v -r a --color yes --doctest-modules --cov ../livelike  --cov-report xml --cov-report term-missing