TESTS=(bmm.json  c1d.json  c2d.json  c3d.json  dil.json  gemm.json  gemv.json  scan.json  t2d.json)

for i in "${TESTS[@]}"
do
	echo "$i"
	python run.py -c configs/${i} 2>&1 > ${i}.log
done 
