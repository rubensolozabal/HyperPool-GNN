#!/bin/sh

echo "Model: GCN2_HBP"
echo "=========================================================================================================="

echo "Cora"
echo "===="
python main.py --dataset=Cora --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h0.01-Cora-public-Adam --hyperbolicity=0.01 --path_results=results/table4/
# python main.py --dataset=Cora --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h0.1-Cora-public-Adam --hyperbolicity=0.1 --path_results=results/table4/
# python main.py --dataset=Cora --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h1.0-Cora-public-Adam --hyperbolicity=1 --path_results=results/table4/
# python main.py --dataset=Cora --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h10-Cora-public-Adam --hyperbolicity=10 --path_results=results/table4/
python main.py --dataset=Cora --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h100-Cora-public-Adam --hyperbolicity=100 --path_results=results/table4/


echo "CiteSeer"
echo "========"
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h0.01-CiteSeer-public-Adam --hyperbolicity=0.01 --path_results=results/table4/
# python main.py --dataset=CiteSeer --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h0.1-CiteSeer-public-Adam --hyperbolicity=0.1 --path_results=results/table4/
# python main.py --dataset=CiteSeer --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h1.0-CiteSeer-public-Adam --hyperbolicity=1.0 --path_results=results/table4/
# python main.py --dataset=CiteSeer --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h10-CiteSeer-public-Adam --hyperbolicity=10.0 --path_results=results/table4/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h100-CiteSeer-public-Adam --hyperbolicity=100.0 --path_results=results/table4/


echo "PubMed"
echo "======"
python main.py --dataset=PubMed --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h0.01-PubMed-public-Adam --hyperbolicity=0.01 --path_results=results/table4/
# python main.py --dataset=PubMed --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h0.1-PubMed-public-Adam --hyperbolicity=0.1 --path_results=results/table4/
# python main.py --dataset=PubMed --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h1.0-PubMed-public-Adam --hyperbolicity=1.0 --path_results=results/table4/
# python main.py --dataset=PubMed --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h10-PubMed-public-Adam --hyperbolicity=10.0 --path_results=results/table4/
python main.py --dataset=PubMed --split=public --model=GCN2_HBP --optimizer=Adam --logger=GCN2_HBP-h100-PubMed-public-Adam --hyperbolicity=100.0 --path_results=results/table4/



echo "Model: GCN2_HBPU"
echo "=========================================================================================================="

echo "Cora"
echo "===="
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h0.01-Cora-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=0.01 --path_results=results/table4/
# python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h0.1-Cora-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=0.1 --path_results=results/table4/
# python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h1.0-Cora-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=1 --path_results=results/table4/
# python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h10-Cora-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=10 --path_results=results/table4/
python main.py --dataset=Cora --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h100-Cora-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=100 --path_results=results/table4/


echo "CiteSeer"
echo "========"
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h0.01-CiteSeer-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=0.01 --path_results=results/table4/
# python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h0.1-CiteSeer-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=0.1 --path_results=results/table4/
# python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h1.0-CiteSeer-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=1 --path_results=results/table4/
# python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h10-CiteSeer-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=10 --path_results=results/table4/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h100-CiteSeer-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=100 --path_results=results/table4/

echo "PubMed"
echo "======"
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h0.01-PubMed-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=0.01 --path_results=results/table4/
# python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h0.1-PubMed-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=0.1 --path_results=results/table4/
# python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h1.0-PubMed-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=1 --path_results=results/table4/
# python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h10-PubMed-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=10 --path_results=results/table4/
python main.py --dataset=PubMed --split=public --model=GCN2_HBPU --optimizer=Adam --logger=GCN2_HBPU-h100-PubMed-public-Adam --alpha=0.1 --regularization=uniformity --hyperbolicity=100 --path_results=results/table4/


echo "Model: GCN2_HLBP"
echo "=========================================================================================================="

echo "Cora"
echo "===="
python main.py --dataset=Cora --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h0.01-Cora-public-Adam --hyperbolicity=0.01 --path_results=results/table4/
# python main.py --dataset=Cora --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h0.1-Cora-public-Adam --hyperbolicity=0.1 --path_results=results/table4/
# python main.py --dataset=Cora --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h1.0-Cora-public-Adam --hyperbolicity=1 --path_results=results/table4/
# python main.py --dataset=Cora --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h10-Cora-public-Adam --hyperbolicity=10 --path_results=results/table4/
python main.py --dataset=Cora --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h100-Cora-public-Adam --hyperbolicity=100 --path_results=results/table4/

echo "CiteSeer"
echo "========"
python main.py --dataset=CiteSeer --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h0.01-CiteSeer-public-Adam --hyperbolicity=0.01 --path_results=results/table4/
# python main.py --dataset=CiteSeer --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h0.1-CiteSeer-public-Adam --hyperbolicity=0.1 --path_results=results/table4/
# python main.py --dataset=CiteSeer --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h1.0-CiteSeer-public-Adam --hyperbolicity=1 --path_results=results/table4/
# python main.py --dataset=CiteSeer --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h10-CiteSeer-public-Adam --hyperbolicity=10 --path_results=results/table4/
python main.py --dataset=CiteSeer --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h100-CiteSeer-public-Adam --hyperbolicity=100 --path_results=results/table4/

echo "PubMed"
echo "======"
python main.py --dataset=PubMed --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h0.01-PubMed-public-Adam --hyperbolicity=0.01 --path_results=results/table4/
# python main.py --dataset=PubMed --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h0.1-PubMed-public-Adam --hyperbolicity=0.1 --path_results=results/table4/
# python main.py --dataset=PubMed --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h1.0-PubMed-public-Adam --hyperbolicity=1 --path_results=results/table4/
# python main.py --dataset=PubMed --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h10-PubMed-public-Adam --hyperbolicity=10 --path_results=results/table4/
python main.py --dataset=PubMed --split=public --model=GCN2_HLBP --optimizer=Adam --logger=GCN2_HLBP-h100-PubMed-public-Adam --hyperbolicity=100 --path_results=results/table4/
