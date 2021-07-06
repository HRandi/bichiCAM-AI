#!/bin/sh
for i in ./src/videos/*.mp4;
  do python ./system/object_tracker.py --weights ./system/lib/mr_sic_ind --model yolov4 --video "$i" --output "${i%.*}.avi" --tiny;
#      if [ -f "${i%.*}.mp4" ]; then
#            rm "$i"
#      fi
  done
