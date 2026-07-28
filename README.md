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
| [`src/main.ipynb`](src/main.ipynb) | baseline supervisionata, porting di `CSI_network.py` in PyTorch |
| [`src/contrastive.ipynb`](src/contrastive.ipynb) | pretraining contrastivo multi-view + fine-tuning a 2 teste (AR + PI) |

Con i pesi scaricati, `contrastive.ipynb` **salta l'addestramento** e passa diretto alla valutazione:
il pretraining riprende da epoca 100/100 e il fine-tuning da fase `full` epoca 15. Per riaddestrare
da zero, svuota la cartella dei checkpoint.

Per far sopravvivere i checkpoint a una disconnessione di Colab, imposta `USE_OWN_DRIVE = True`
nella cella dei checkpoint: monta il tuo Drive e salva in `MyDrive/HAR_checkpoints`.

## Pipeline

1. **Generazione dataset** — spettro Doppler per antenna, media per colonna rimossa, split temporale
   60/20/20 per attività con gap anti-overlap, finestre da 340 colonne.
2. **Pretraining contrastivo** — le 4 antenne della stessa finestra sono viste naturali di uno stesso
   evento. Loss NT-Xent multi-positive (forma SupCon): i positivi di ogni vista sono le altre 3 viste
   della stessa finestra, i negativi tutte le viste delle altre finestre del batch. Nessuna etichetta.
3. **Fine-tuning a 2 teste** — fase A linear probe con backbone congelato, fase B full fine-tune a LR
   ridotto. Loss `CE(AR) + λ·CE(PI)`, early stopping su `AR + PI` di validation.
4. **Valutazione** — fusione delle decisioni per media delle softmax sulle 4 antenne della stessa
   acquisizione. Il set `S6` è tenuto fuori dal training e serve solo da test di generalizzazione AR.

## Asset della release

| file | dimensione | contenuto |
|---|---|---|
| `doppler_traces.zip` | 799 MB | spettri Doppler |
| `contrastive_resnet.torch` | 260 MB | backbone ResNet34, pretraining contrastivo, 100 epoche |
| `twohead_resnet.torch` | 259 MB | fine-tuning 2 teste, checkpoint riavviabile |
| `twohead_resnet_best.torch` | 86 MB | migliori pesi su validation |

## Crediti

Il dataset Doppler proviene da **SHARP** (Meneghello et al.), ridistribuito nella release solo per
riproducibilità. Il codice di generazione del dataset in `src/` è un refactoring del codice SHARP
originale (`CSI_doppler_create_dataset_train.py` / `_test.py`, `CSI_network.py`) da TensorFlow a PyTorch.
