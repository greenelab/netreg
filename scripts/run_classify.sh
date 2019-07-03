for k in 10 20 50 100 200; do
    cmd="python compress.py -k $k -v"
    echo "Running: $cmd"
    eval $cmd
    cmd="python compress.py -k $k -s -v"
    echo "Running: $cmd"
    eval $cmd
done

# gene list from BioBombe paper, just do these for now
cmd="python classify_mutations.py --v --g TP53 PTEN PIK3CA KRAS TTN"
echo "Running: $cmd"
eval $cmd

# run classification using raw expression features
cmd="python classify_with_raw_expression.py --v --g TP53 PTEN PIK3CA KRAS TTN"
echo "Running: $cmd"
eval $cmd
