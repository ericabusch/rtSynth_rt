#!/usr/bin/env bash

# get commandline params
while test $# -gt 0
do
  case "$1" in
    -h)
      echo "$0 [-p [project] -c <toml_file>] [-ip <local_ip_or_hostname] [--localfiles]"
      exit 0
      ;;
    -c) CFG=$2
      ;;
    -ip) IP=$2
      ;;
    --localfiles) USELOCALFILES=1
      ;;
    -p) PROJECT=$2
      ;;
  esac
  shift
done

if [ -z $PROJECT ]; then
  # USELOCALFILES not set, use remote files
  echo 'must specify a project name with -p option'
  exit 0
fi

# check if experiment file is supplied with -e filename
CFG_PARAM=''
if [ ! -z $CFG ]; then
  CFG_PARAM="-c $CFG"
fi

R_PARAM=''
if [ -z $USELOCALFILES ]; then
  # USELOCALFILES not set, use remote files
  R_PARAM='-x'
fi

pushd web
npm run build
popd

# activate conda python env
source ~/.bashrc
conda deactivate
conda activate rtcloud

if [ -z $IP ]; then
  echo "Warning: no ip address supplied, credentials won't be updated"
else
  bash scripts/make-sslcert.sh -ip $IP
fi


python projects/$PROJECT/webMain.py $R_PARAM $CFG_PARAM
