#!/bin/bash
set -ex

cd /home/map005/data/eccc-ppp6/gitprojects/SIDRRpy/
rm -f /home/map005/data/eccc-ppp6/gitprojects/SIDRRpy/Lance_SIDRRpy.ppp*
chmod 755 Lance_SIDRRpy.sh
ord_soumet ./Lance_SIDRRpy.sh -cm 10G -w 360 -jn -listing .
