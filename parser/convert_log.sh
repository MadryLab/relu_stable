#!/bin/bash
(echo "split,zero_output,linear_in_input,constant_output" > "verification/logs/"$1.csv) && (sed 's/-/\n/g' "verification/logs/"$1.log | grep "split" | sed 's/[^0-9|,]*//g' >> "verification/logs/"$1.csv)
