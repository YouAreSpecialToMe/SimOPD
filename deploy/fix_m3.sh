#!/usr/bin/env bash
# Run on m3 after git pull:  bash deploy/fix_m3.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/fix_node.sh" m3
