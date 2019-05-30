with open('slurm-5778652.out') as f:
    for line in f:
        #fields = line.rstrip().split(',')
        #if fields[2] == 'c' and fields[4]:
        if not line.startswith("torch"):
           print(line.rstrip())
