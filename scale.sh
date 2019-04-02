#!/usr/bin/env bash

sleep 15
for i in 4 12 4
do
   ./wrk -t$i -c$i -d320s --timeout 60s -s request.lua http://mlboard-v2.kuberlab:8082/api/v2/tfproxy/
done