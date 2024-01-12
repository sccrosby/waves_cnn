echo "" > out.log
for i in 1 2
do
  echo "$i" >> out.log
  echo "$(date +%s)" >> out.log
  python3 ConvNN/train.py
  echo "$(date +%s)" >> out.log
  echo "" >> out.log
done
