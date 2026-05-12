# DA5400_Bonus_Assignment
This repository is created to Present my work on Data Condensation on HAR data for the Bonus assignment in the subject DA5400
create folder
1.data
inside your project folder and store all of your data there 
2.install all requirements to run this pipeline 
3.Runs a small, fast version to verify the pipeline works:
  "python3 condtsc_pipeline.py --data-dir data --epochs 2 --spc 2 --max-train-per-class 200 --eval-epochs 2 --batch-size 32"
4.Start with this before scaling up:
  "python condtsc_pipeline.py --data-dir data --epochs 200 --spc 10 --max-train-per-class 2000 --eval-epochs 100 --batch-size 128 --run-full-baseline"
5.For a heavier paper-like run, increase --epochs, --m-real, and remove or raise --max-train-per-class.
6.Outputs are written to outputs/, including the condensed dataset as .npz.
