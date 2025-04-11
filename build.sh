pip install .

git clone https://wenhaoli-xmu/lm-profiler.git
cd lm-profiler && pip install -e .
cd ..

python find_spec/bf16_linear_fwd.py
python find_spec/bf16_linear_bwd_da.py
python find_spec/bf16_linear_bwd_db.py

mv bf16_linear_fwd.json ffn_kernel/
mv bf16_linear_bwd_da.json ffn_kernel/
mv bf16_linear_bwd_db.json ffn_kernel/
