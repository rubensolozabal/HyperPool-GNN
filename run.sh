#!/bin/sh

echo "Model: GCN"
echo "=========================================================================================================="


echo "Cora"
echo "===="
# python main.py --dataset=Cora --split=public --model=GCN --optimizer=Adam --logger=GCN-Cora-public-Adam
# python main.py --dataset=Cora --split=full --model=GCN --optimizer=Adam --logger=GCN-Cora-full-Adam
# python main.py --dataset=Cora --split=complete --model=GCN --optimizer=Adam --logger=GCN-Cora-complete-Adam

echo "CiteSeer"
echo "========"
# python main.py --dataset=CiteSeer --split=public --model=GCN --optimizer=Adam --logger=GCN-CiteSeer-public-Adam
# python main.py --dataset=CiteSeer --split=full --model=GCN --optimizer=Adam --logger=GCN-CiteSeer-full-Adam
# python main.py --dataset=CiteSeer --split=complete --model=GCN --optimizer=Adam --logger=GCN-CiteSeer-complete-Adam


echo "PubMed"
echo "======"
# python main.py --dataset=PubMed --split=public --model=GCN --optimizer=Adam --logger=GCN-PubMed-public-Adam
# python main.py --dataset=PubMed --split=full --model=GCN --optimizer=Adam --logger=GCN-PubMed-full-Adam
# python main.py --dataset=PubMed --split=complete --model=GCN --optimizer=Adam --logger=GCN-PubMed-complete-Adam



echo "Model: GCN2"
echo "=========================================================================================================="

echo "Cora"
echo "===="

# python main.py --dataset=Cora --split=public --model=GCN2 --optimizer=Adam --logger=GCN2-Cora-public-Adam
python main.py --dataset=Cora --split=full --model=GCN2 --optimizer=Adam --logger=GCN2-Cora-full-Adam
python main.py --dataset=Cora --split=complete --model=GCN2 --optimizer=Adam --logger=GCN2-Cora-complete-Adam

echo "CiteSeer"
echo "========"
# python main.py --dataset=CiteSeer --split=public --model=GCN2 --optimizer=Adam --logger=GCN2-CiteSeer-public-Adam

echo "PubMed"
echo "======"
# python main.py --dataset=PubMed --split=public --model=GCN2 --optimizer=Adam --logger=GCN2-PubMed-public-Adam