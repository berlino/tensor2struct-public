#! /bin/bash

configfile=$1

if [ -z ${configfile} ]; then
    echo "./path_to_this_script.sh configfile";
    exit 1;
fi

for domain in calendar blocks socialnetwork publications recipes restaurants housing basketball
do
    echo "preprocessing $domain"
    args="{\"target_domain\": \"$domain\"}"
    tensor2struct preprocess ${configfile} --config_args "$args"
done
