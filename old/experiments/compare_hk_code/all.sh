cc-submit sbatch -j cit_hepph_py run_kcore_baseline_py.sh cit_hepph
cc-submit sbatch -j orkut_py run_kcore_baseline_py.sh orkut
cc-submit sbatch -j cen_py run_kcore_baseline_py.sh cen

cc-submit sbatch -j cit_hepph_cpp run_kcore_baseline_cpp.sh cit_hepph
cc-submit sbatch -j orkut_cpp run_kcore_baseline_cpp.sh orkut
cc-submit sbatch -j cen_cpp run_kcore_baseline_cpp.sh cen
