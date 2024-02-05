#!/bin/sh


echo "Model: GCN2_HBPU"
echo "=========================================================================================================="

echo "Cora"
echo "===="
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u0.0 --alpha=0.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u0.1 --alpha=0.1 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u0.2 --alpha=0.2 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u0.5 --alpha=0.5 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u1.0 --alpha=1 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u2.0 --alpha=2 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u5.0 --alpha=5 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u10.0 --alpha=10 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u20.0 --alpha=20 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u50.0 --alpha=50 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-Cora-public-Adam-u100.0 --alpha=100 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/



echo "CiteSeer"
echo "========"
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u0.0 --alpha=0.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u0.1 --alpha=0.1 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u0.5 --alpha=0.5 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u1.0 --alpha=1.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u2.0 --alpha=2.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u5.0 --alpha=5.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u10.0 --alpha=10.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u20.0 --alpha=20.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u50.0 --alpha=50.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u100.0 --alpha=100.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u200.0 --alpha=200.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u500.0 --alpha=500.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-CiteSeer-public-Adam-u1000.0 --alpha=1000.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/

echo "PubMed"
echo "======"
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u0.0 --alpha=0.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u0.1 --alpha=0.1 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u0.2 --alpha=0.2 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u0.5 --alpha=0.5 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u1.0 --alpha=1.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u2.0 --alpha=2.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u5.0 --alpha=5.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u10.0 --alpha=10.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u20.0 --alpha=20.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u50.0 --alpha=50.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u100.0 --alpha=100.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u200.0 --alpha=200.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u500.0 --alpha=100.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-PubMed-public-Adam-u1000.0 --alpha=100.0 --regularization=uniformity --hyperbolicity=1 --path_results=results/plot1/


