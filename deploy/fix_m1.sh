#!/usr/bin/env bash
# Run on m1 after git pull:  bash deploy/fix_m1.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/fix_node.sh" m1
