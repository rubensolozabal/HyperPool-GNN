#!/bin/sh

echo "Model: GCN_HBP"
echo "=========================================================================================================="

echo "Cora"
echo "===="
python main.py --dataset=Cora --split=public --model=GCN_HBP --optimizer=Adam --logger=GCN_HBP-h1-Cora-public-Adam --hyperbolicity=1 --path_results=results/table5/ 

echo "CiteSeer"
echo "========"
python main.py --dataset=CiteSeer --split=public --model=GCN_HBP --optimizer=Adam --logger=GCN_HBP-h1-CiteSeer-public-Adam --hyperbolicity=1 --path_results=results/table5/


echo "PubMed"
echo "======"
python main.py --dataset=PubMed --split=public --model=GCN_HBP --optimizer=Adam --logger=GCN_HBP-h1-PubMed-public-Adam --hyperbolicity=1 --path_results=results/table5/



echo "Model: SAGE_HBP"
echo "=========================================================================================================="

echo "Cora"
echo "===="
python main.py --dataset=Cora --split=public --model=SAGE_HBP --optimizer=Adam --logger=SAGE_HBP-h1-Cora-public-Adam --hyperbolicity=1 --path_results=results/table5/

echo "CiteSeer"
echo "========"
python main.py --dataset=CiteSeer --split=public --model=SAGE_HBP --optimizer=Adam --logger=SAGE_HBP-h1-CiteSeer-public-Adam --hyperbolicity=1 --path_results=results/table5/


echo "PubMed"
echo "======"
python main.py --dataset=PubMed --split=public --model=SAGE_HBP --optimizer=Adam --logger=SAGE_HBP-h1-PubMed-public-Adam --hyperbolicity=1 --path_results=results/table5/


