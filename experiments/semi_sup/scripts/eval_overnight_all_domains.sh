#! /bin/bash

configfile=$1
expid=$2

if [ -z ${configfile} ] || [ -z ${expid} ]; then
    echo "./path_to_this_script.sh configfile expid";
    exit 1;
fi

for domain in calendar blocks socialnetwork publications recipes restaurants housing basketball
do
    echo "evaluating $domain"
    config_args="{\"target_domain\": \"$domain\", \"exp_id\": $expid}"

    # sequential
    tensor2struct eval ${configfile} --config_args "${config_args}"
done
