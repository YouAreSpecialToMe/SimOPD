#!/usr/bin/env bash
# Run on m2 after git pull:  bash deploy/fix_m2.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/fix_node.sh" m2
