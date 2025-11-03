Este repositório contém um pipeline completo de treinamento, teste e avaliação de modelos de segmentação de imagens 
médicas panorâmicas, incluindo validação cruzada, Grid Search de hiperparâmetros e avaliação em datasets externos.

Funcionalidades

1 - Treinamento com K-Fold + Grid Search
Suporte a múltiplas arquiteturas: UNet, UNetPlusPlus, UNet3Plus, AttentionUNet e WNet.
Busca de melhores hiperparâmetros (learning_rate, batch_size, optimizer) via ParameterGrid.
Validação com Dice para seleção do melhor modelo por fold.
Salva checkpoints de cada fold.

2 -Treinamento final
Treinamento do modelo completo com a melhor configuração obtida no Grid Search.
Salva pesos finais do modelo treinado.

3-Avaliação em datasets externos
Avaliação de modelos finais em datasets externos.
Geração de métricas: Accuracy, Specificity, Sensitivity, E-measure, MAE, IoU, Dice.
Salvamento das máscaras previstas (limitadas para evitar sobrecarga).

4-Teste estatístico de Wilcoxon
Comparação entre modelos usando a métrica Dice.
Determina se a diferença entre modelos é estatisticamente significativa.
