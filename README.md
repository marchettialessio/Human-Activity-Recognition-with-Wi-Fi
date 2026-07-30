# Human Activity Recognition with Wi-Fi

Riconoscimento di attività umane (**AR**) e identificazione della persona (**PI**) da spettri Doppler
ricavati dal CSI Wi-Fi, con pretraining **contrastivo self-supervised** e fine-tuning a due teste.

## Come eseguire su Google Colab

I notebook stanno in [`src/`](src/) e scaricano **dataset e pesi da soli**, dalla
[release `assets-v1`](https://github.com/marchettialessio/Human-Activity-Recognition-with-Wi-Fi/releases/tag/assets-v1)
di questo repo. Non serve alcun login, né montare Google Drive, né configurare niente.

1. Apri il notebook in Colab e imposta il runtime su **GPU** (`Runtime > Change runtime type > T4`).
2. Esegui le celle in ordine.

La prima cella scarica ed estrae il dataset; la cella dei checkpoint scarica i pesi pre-addestrati.
Entrambe verificano la dimensione esatta dei file e riprendono i download interrotti (`wget -c`),
così un'interruzione di rete non lascia un file troncato.

> I file **non** sono su Google Drive: l'endpoint di download pubblico di Drive applica una quota
> per-file sui byte serviti e blocca gli IP dei datacenter Colab con
> `FileURLRetrievalError: Cannot retrieve the public link of the file`. Gli asset di una release
> GitHub non hanno quota di download.

### Notebook

| notebook | contenuto |
|---|---|
| [`src/SHARP.ipynb`](src/main.ipynb) | baseline supervisionata, porting di `CSI_network.py` in PyTorch |
| [`src/Self-Supervised_Contrastive.ipynb`](src/Self-Supervised_Contrastive.ipynb) | pretraining contrastivo multi-view + fine-tuning a 2 teste (AR + PI) |
| [`src/Supervised_Contrastive.ipynb`](src/Supervised_Contrastive.ipynb) | pretraining contrastivo, chiavi positive : sample con stessa label + fine-tuning con singola head per AR |

Con i pesi scaricati, `Self-Supervised_Contrastive.ipynb` e  `Supervised_Contrastive.ipynb`**saltano l'addestramento** e passano diretti alla valutazione:
il pretraining e il fine-tuning riprendono dall'ultima epoca. Per riaddestrare da zero, svuota la cartella dei checkpoint e commenta la sezione per scaricare i pesi, rispettivamente nelle celle 16 e 12.

Per far sopravvivere i checkpoint a una disconnessione di Colab, imposta `USE_OWN_DRIVE = True`
nella cella dei checkpoint: monta il tuo Drive e salva in `MyDrive/HAR_checkpoints`.

## Pipeline per Self-Supervised Contrastive

1. **Generazione dataset** — spettro Doppler per antenna, media per colonna rimossa, split temporale
   60/20/20 per attività con gap anti-overlap, finestre da 340 colonne.
2. **Pretraining contrastivo** — le 4 antenne della stessa finestra sono viste naturali di uno stesso
   evento. Loss NT-Xent multi-positive (forma SupCon): i positivi di ogni vista sono le altre 3 viste
   della stessa finestra, i negativi tutte le viste delle altre finestre del batch. Nessuna etichetta.
3. **Fine-tuning a 2 teste** — fase A linear probe con backbone congelato, fase B full fine-tune a LR
   ridotto. Loss `CE(AR) + λ·CE(PI)`, early stopping su `AR + PI` di validation.
4. **Valutazione** — fusione delle decisioni per media delle softmax sulle 4 antenne della stessa
   acquisizione. Il set `S6` è tenuto fuori dal training e serve solo da test di generalizzazione AR.

## Pipeline per Supervised Contrastive

1. **Generazione dataset** — spettro Doppler per antenna, media per colonna rimossa, split temporale
   60/20/20 per attività con gap anti-overlap, finestre da 340 colonne.
2. **Pretraining contrastivo** — positivi definiti dalle finestre con la stessa etichetta di attività; le altre classi sono negativi.
3. **Fine-tuning a singola head per AR** — backbone contrastivo congelato, poi fine-tuning della classificazione con Cross-Entropy.
4. **Valutazione** — accuratezza e confusion matrix sul test, con fusione delle quattro antenne della stessa acquisizione.

## Asset della release

| file | dimensione | contenuto |
|---|---|---|
| `doppler_traces.zip` | 799 MB | spettri Doppler |
| `contrastive_resnet.torch` | 260 MB | backbone ResNet34, pretraining contrastivo, 100 epoche |
| `twohead_resnet.torch` | 259 MB | fine-tuning 2 teste, checkpoint riavviabile |
| `twohead_resnet_best.torch` | 86 MB | migliori pesi su validation |
| `supcon_checkpoint.torch` | 210 MB | pesi della backbone per Supervised_Contrastive |
| `classification_head_best.torch` | 2.1 MB | migliori pesi su validation per la head di Supervised_Contrastive |

## Crediti

Il dataset Doppler proviene da **SHARP** (Meneghello et al.), ridistribuito nella release solo per
riproducibilità. Il codice di generazione del dataset in `src/` è un refactoring del codice SHARP
originale (`CSI_doppler_create_dataset_train.py` / `_test.py`, `CSI_network.py`) da TensorFlow a PyTorch.
