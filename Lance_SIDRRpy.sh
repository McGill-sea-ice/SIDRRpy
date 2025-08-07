#!/bin/bash
set -ex

#. r.load.dot rpn/libs/19.5
. r.load.dot rpn/libs/20220216

#. r.load.dot rpn/utils/19.5
. r.load.dot rpn/utils/20220509

export http_proxy=http://webproxy.science.gc.ca:8888/
export https_proxy=http://webproxy.science.gc.ca:8888/

#Python
. ssmuse-sh -p /fs/ssm/eccc/cmd/cmds/env/python/py39_2022.05.24_rhel-8-icelake-64
#rpnpy
. r.load.dot eccc/mrd/rpn/MIG/ENV/rpnpy/2.1-u2.5  main/opt/intelcomp/inteloneapi-2022.1.2/intelcomp+mpi+mkl

#Not sure is this is really needed
. r.load.dot eccc/mrd/rpn/MIG/GEM/5.1.2

#CONCEPT tools
. ssmuse-sh -d  /fs/ssm/eccc/mrd/rpn/OCEAN/cstint/3.2.12
. ssmuse-sh -d  /fs/ssm/eccc/mrd/rpn/OCEAN/cncpt/5.4.0
. ssmuse-sh -d  /fs/ssm/eccc/mrd/rpn/OCEAN/cncpt-ov/5.4.0

#. ssmuse-sh -x comm/eccc/all/opt/intelcomp/intelpsxe-cluster-19.0.3.199

#. ssmuse-sh -x hpco/exp/openmpi/openmpi-3.1.2--hpcx-2.4.0-mofed-4.6--intel-19.0.3.199

#. ssmuse-sh -x hpco/exp/openmpi-setup/openmpi-setup-0.2

#. ssmuse-sh -d /fs/ssm/eccc/mrd/rpn/utils/19.5

cd /home/map005/data/eccc-ppp6/gitprojects/SIDRRpy
chmod 755 DatasetAnalysis.py
python DatasetAnalysis.py
