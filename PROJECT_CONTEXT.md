# PROJECT_CONTEXT.md — Immunogold-Dots-Project

> Documento di handoff generato a partire dalla cronologia completa di sviluppo con Claude (Anthropic). Scopo: fornire a un nuovo agente AI (o a un umano) tutto il contesto necessario per proseguire il progetto senza dover rileggere l'intera conversazione originale.

---

## 1. Panoramica del Progetto

**Obiettivo principale**: costruire una pipeline di deep learning per il conteggio automatico di particelle immunogold in immagini TEM (Transmission Electron Microscopy), eliminando la necessità di conteggio manuale su ImageJ.

**Contesto biologico/applicativo**:
- Studio del canale meccanosensibile **Piezo1** su membrana plasmatica, in monostrati cellulari **U87 wild-type** e **U87 KO-Piezo1**.
- Marcatura immunogold: Ab primario anti-Piezo1 ECD (rabbit, 15939-1-AP Proteintech, 1:400) + Ab secondario Nanogold-Fab' goat anti-rabbit 1.4nm (41C642, 1:100), seguito da enhancement GOLDENHANCE™ (silver/gold, ~8 min di reazione).
- Protocollo **pre-embedding** (non post-embedding): introduce variabilità morfologica nei puntini (forma irregolare, dimensione non uniforme) — una sfida nota e ricorrente per il rilevamento automatico.
- Acquisizione TEM (microscopio JEOL JEM-1400 Flash), 144 immagini raw ad alta risoluzione, distribuite su 4 esperimenti (2 date × 2 gruppi cellulari: WT/KO), a ingrandimenti diversi tra loro.
- Il progetto è nato dopo aver abbandonato uno strumento esistente ("Gold Digger") per incompatibilità di versione Python e pesi del modello non disponibili — la pipeline è stata ricostruita da zero.

**Scopo finale del modello IA**: due obiettivi analitici distinti, in sequenza:
1. **Conteggio dei singoli puntini gold** — focus attuale, in corso.
2. **Classificazione delle configurazioni** (monomero / trimero / artefatto) — fase futura, non ancora iniziata. Approccio previsto: clustering spaziale non supervisionato (probabile DBSCAN) con soglie di distanza motivate fisicamente dalla geometria del linkage anticorpo-gold, poiché non esistono ancora esempi di ground truth confermati per questa discriminazione.

---

## 2. Architettura e Scelte Tecniche

**Stack**: Python, **PyTorch**, Google Colab + Google Drive (storage e ambiente di esecuzione).

**Approccio di modeling scelto**: **density map regression** (kernel gaussiano), preferito a object detection o semantic segmentation per la sua robustezza a puntini sovrapposti/ravvicinati.

**Pipeline concettuale end-to-end**:
raw images (144, TEM) → calibrazione scala → tiling in patch → annotazione (ImageJ) →
generazione density map target → split train/val/test → baseline classico (rif.) →
costruzione CNN encoder-decoder → training → valutazione (MAE) → [futuro: clustering monomero/trimero]


**Architettura CNN — U-Net compatta** (dettagli completi in `scripts/modello.py`):
- Input: patch 256×256, 1 canale (scala di grigi). Output: density map, stessa risoluzione, 1 canale, non-negativa.
- 3 livelli di encoder (non 4-5 come uno standard U-Net biomedico) — scelta deliberata per il dataset ridotto (10 immagini di train).
- Progressione canali: `canali_base=16` → Encoder 1→16→32→64 (risoluzione 256→128→64) → Bottleneck 64→128 (32×32) → Decoder 128→64→32→16 (risoluzione 32→64→128→256) → Output 1×1 conv + `Softplus`.
- Ogni blocco convoluzionale = 2× (Conv2d 3×3, padding=1 + ReLU).
- Skip connections per **concatenazione** (non somma) tra encoder e decoder, alla stessa risoluzione spaziale.
- Upsampling nel decoder: `nn.Upsample(bilinear)` + conv 1×1 di riduzione canali (per rendere prevedibile il numero di canali prima della concatenazione con lo skip).
- Parametri allenabili totali: **449.489**.
- Loss: `nn.MSELoss()`. Optimizer: `Adam`, `lr=1e-3`.

**Baseline classico (termine di paragone pre-CNN)**: `skimage.feature.blob_log` (Laplacian of Gaussian), scelto su Hough Circle Transform perché riusa direttamente `sigma_px` già calcolato per classe di calibrazione (finestra di ricerca vincolata a `sigma_px * (1 ± fattore_range)`). Vedi sezione 4 per risultati.

**Librerie chiave**: `torch`, `torch.nn`/`torch.nn.functional`, `torchvision.transforms.functional` (per rotazioni in augmentation), `numpy`, `pandas`, `PIL.Image`, `scipy.ndimage` (`gaussian_filter`, `uniform_filter`, `binary_dilation`), `skimage.feature.blob_log`, `matplotlib.pyplot`.

---

## 3. Struttura dei Dati e Annotazioni

**Struttura cartelle su Drive** (`/content/drive/MyDrive/progetto_immunogold/`):
raw_images/ 144 immagini TEM originali
annotations/ CSV di calibrazione, statistiche, split, log (vedi sotto)
points/ export ImageJ grezzi per immagine intera (PRE-tiling)
patches/
images/ patch 256×256 ritagliate, con overlap
coordinates/ CSV coordinate locali per patch (POST-tiling, x_local/y_local)
density_maps/ target .npy generati (float32)
checkpoints/ pesi modello + record best val loss
logs/ project_log.txt (log append-only)
scripts/ common.py, modello.py
notebooks/ tutti i notebook numerati


⚠️ **Attenzione a non confondere** `annotations/points/` (coordinate a livello di immagine intera, pre-tiling) con `patches/coordinates/` (coordinate locali per singola patch, post-tiling) — due cartelle diverse con scopo diverso, causa di un bug già risolto in passato.

**Convenzione di naming patch**: `{nome_immagine_originale}_x{offset_x}_y{offset_y}.png` (es. `SA-MAG_X10k_JEM-1400 Flash_22440_x0_y0.png`) — **non** `_patch_XX_YY` come inizialmente ipotizzato (bug corretto). Il file di coordinate corrispondente ha lo stesso nome base + `_points.csv`, la density map + `.npy`.

**Calibrazione (`annotations/calibrazione.csv`)**: una riga per immagine raw (non per classe), colonne `filename` (con estensione `.jpg`), `filepath`, `esperimento`, `gruppo_ingrandimento`, `nm_per_pixel`, `classe_calibrazione`. 23 combinazioni esperimento×gruppo di ingrandimento sono state clusterizzate (tolleranza 3%) in **10 classi di calibrazione** vere, basate sul rapporto nm/pixel. `filename` è chiave di join sicura (verificato: nessuna collisione tra esperimenti diversi).

**Statistiche dimensione puntini (`annotations/statistiche_raggio_per_classe.csv`)**: una riga per classe, con `raggio_medio_nm` (range ~4.9nm classe_2 a 28.24nm classe_10, outlier) e `cv_percento` (12.7%–33.6%, coerente con la morfologia irregolare da protocollo pre-embedding).

**Sigma per classe (`annotations/sigma_per_classe.csv`, generato)**: `sigma_px = raggio_medio_nm / nm_per_pixel`, calcolato per classe — non un valore globale, perché classi diverse hanno ingrandimenti (nm/pixel) diversi anche a parità di dimensione fisica del puntino.

**Mappa patch → classe → sigma (`annotations/mappa_patch.csv`, generato)**: join patch → immagine sorgente (via `filename`, stem-matching per gestire estensioni diverse) → classe di calibrazione → sigma.

**Split train/val/test (`annotations/split_immagini.csv`, generato)**: split a **livello di immagine**, non di patch (evita data leakage). Regola algoritmica: classi con 1 sola immagine annotata → tutta in train; classi con 2 immagini → 1 train + 1 val/test (bilanciato con contatore incrementale); classi con ≥3 immagini → 1 val + 1 test + resto in train. Risultato attuale: **10 immagini train / 4 val / 3 test** (su 17 immagini annotate finora, su 144 totali).

**Stato annotazione**: solo **17 immagini su 144** annotate finora (strategia incrementale deliberata, non tutte le 144 a priori). Il tiling in patch è stato eseguito **solo** su queste 17 immagini → **1342 patch totali**, di cui **793 nel train set** (dopo filtro per disponibilità di density map).

---

## 4. Stato Attuale dei Lavori ed Esperimenti

### Completato e verificato
- **Calibrazione scala** (10 classi, 23 combinazioni) e **misurazione diametri puntini** per classe.
- **Tiling** (256×256, overlap 32px) sulle 17 immagini annotate.
- **Generazione density map** (Fase 4): 1342 patch → 1342 `.npy` generati. Controllo di sanità (`somma_mappa` vs `n_punti_attesi`, tolleranza 5%): 1056/1342 (78.7%) entro tolleranza; i restanti concentrati su patch a bassa densità (n=1-3 punti, perdita di massa gaussiana ai bordi — comportamento atteso, non bug); un solo caso anomalo isolato (somma≈0 con 1 punto atteso, verosimilmente coordinata fuori dai limiti della patch).
- **Split train/val/test** (Fase 5): completato, 10/4/3 immagini.
- **Baseline classico** (Fase 6, `blob_log`): completato con tuning iterativo.
  - Scoperto e risolto un **artefatto sistematico**: la scale bar/barra info del software del microscopio (fascia nera uniforme) veniva rilevata come falsi blob. Un primo tentativo di mascheramento diretto sull'immagine (azzeramento pixel) ha **peggiorato** il MAE creando un bordo artificiale netto, a sua volta rilevato come blob — corretto filtrando i blob **dopo** il rilevamento (per posizione, con maschera dilatata), non modificando l'immagine in input al detector.
  - **Risultato finale baseline: MAE = 1.857, bias = -0.562** (sottostima sistematica, verosimilmente per fusione di particelle ravvicinate in un unico blob — limite strutturale dei blob detector classici) su 315 patch di validation, con `threshold=0.175`, `fattore_range=0.3`, `num_sigma=5`.
  - **Questo MAE=1.857 è il numero di riferimento che la CNN deve battere.**
  - Log di tuning accumulati in `baseline_tuning_log.csv` e `baseline_mae_per_classe_log.csv` (breakdown per classe).
- **Architettura CNN** (Fase 7): costruita e verificata pezzo per pezzo (ogni blocco testato isolatamente prima dell'assemblaggio finale), salvata in `scripts/modello.py`. Verifica end-to-end con tensore casuale: shape e non-negatività confermate.

### In corso / parzialmente completato
- **Training CNN** (Fase 8, notebook `10_training.ipynb`): Dataset PyTorch, augmentation, DataLoader, loss/optimizer scritti e testati. Training loop scritto con checkpointing.
  - **Bug critico #1 (risolto in codice, fix non ancora validato con un run completo)**: il primo run di training è collassato a predire ~0 ovunque (val loss ~0.00000 già dall'epoca 2). **Causa**: i valori target delle density map sono minuscoli (picco ~0.002-0.01, kernel gaussiano che somma a 1 spalmato su decine di pixel) → l'MSE di "predici sempre zero" è già ~1e-7, segnale di gradiente trascurabile per imparare i pattern reali; aggravato dalla `Softplus` finale, la cui derivata è quasi nulla per input molto negativi (necessari per avvicinarsi a 0). **Fix**: `FATTORE_SCALA = 100.0` applicato al target **al momento del caricamento** nel `Dataset.__getitem__` (i file `.npy` su disco restano invariati/non scalati) — deve essere ridiviso ovunque si visualizzi/calcoli un conteggio fisico (titoli dei plot, futuro calcolo MAE). Un nuovo run con questo fix è stato appena lanciato, esito non ancora confermato a fine conversazione.
  - **Bug #2 (risolto)**: `migliore_val_loss` si resettava a `infinito` a ogni riavvio del runtime Colab, rischiando di sovrascrivere un checkpoint buono di una sessione precedente con uno peggiore di una sessione nuova. **Fix**: persistenza in `checkpoints/record_val_loss.json`, caricato all'inizio di ogni run invece di reinizializzare sempre a infinito.
  - **Problema aperto a fine conversazione**: `matplotlib.pyplot` appena aggiunto a `common.py`, ma verosimilmente dimenticato di aggiungerlo alla lista `__all__` → `NameError: name 'plt' is not defined` nelle celle di visualizzazione. Diagnosi indicata ma non confermata da verifica diretta del file (accesso Drive momentaneamente non disponibile a fine sessione).
- **GPU**: il training va eseguito con GPU attiva (Runtime > Change runtime type > GPU) — per diverse fasi precedenti il device è rimasto erroneamente su CPU senza conseguenze pratiche (preprocessing/baseline non ne avevano bisogno), ma per il training è necessaria.

### Non ancora iniziato
- Calcolo formale del MAE della CNN sul validation set (nella stessa formula del baseline, per confronto diretto).
- Valutazione finale sul **test set** (da fare una sola volta, a modello e iperparametri finalizzati).
- Estensione del checkpointing a stato optimizer + epoca (per vero resume, se servissero run lunghi non presidiati).
- Uso della funzione di rilevamento artefatti/scale-bar come filtro qualità dati **prima** del training CNN (idea accantonata per il futuro, non ancora implementata in questo contesto).
- Fase di classificazione monomero/trimero/artefatto (clustering DBSCAN).

---

## 5. Mappa del Codice e dei File

| File | Ruolo |
|---|---|
| `scripts/common.py` | Stato condiviso del progetto: dizionario `paths` (tutte le cartelle), `device` (cuda/cpu), `log()` (scrive su `project_log.txt` + stampa), import di terze parti centralizzati e ri-esportati via `__all__` (usare `from common import *` in ogni notebook). **Attenzione**: ogni nuova libreria aggiunta va anche aggiunta a `__all__`, altrimenti `NameError` silenzioso a valle. |
| `scripts/modello.py` | Tutte le classi dell'architettura U-Net (`BloccoConv`, `LivelloEncoder`, `Encoder`, `Bottleneck`, `LivelloDecoder`, `Decoder`, `LivelloOutput`, `UNetImmunogold`). Importare con `from modello import UNetImmunogold`. |
| `notebooks/06_density_maps.ipynb` | Fase 4: calcolo sigma per classe, mappatura patch→classe→sigma, generazione e salvataggio density map `.npy`, storico controlli di sanità. |
| `notebooks/07_split_dataset.ipynb` | Fase 5: split train/val/test a livello immagine, algoritmo di bilanciamento per classi con pochi esempi. |
| `notebooks/08_baseline_blob_detector.ipynb` | Fase 6: baseline classico `blob_log`, rilevamento/filtro artefatto scale-bar, tuning iperparametri, log MAE (totale e per classe). |
| `notebooks/09_architettura_encoder_decoder.ipynb` | Fase 7: costruzione e verifica pezzo-per-pezzo dell'architettura U-Net (contenuto ora duplicato/centralizzato anche in `modello.py`). |
| `notebooks/10_training.ipynb` | Fase 8 (in corso): Dataset PyTorch, augmentation, DataLoader, loss/optimizer, training loop con checkpointing, diagnostica (curva di loss, feedback visivo). |
| Notebook fasi 0-3 (setup ambiente, calibrazione/annotazione, tiling) | Completati in fasi precedenti del progetto (naming originale previsto: `00_setup_ambiente`, `01_calibrazione_annotazione`, `02_tiling`), non ricostruiti in dettaglio in questo documento — fare riferimento diretto ai file su Drive per il codice esatto. |

**CSV/dati generati chiave** (tutti in `annotations/`): `calibrazione.csv`, `statistiche_raggio_per_classe.csv`, `sigma_per_classe.csv`, `mappa_patch.csv`, `split_immagini.csv`, `sanita_storico.csv`, `baseline_blob_dettaglio.csv` (sovrascritto ad ogni run), `baseline_tuning_log.csv` (accumulato), `baseline_mae_per_classe_log.csv` (accumulato).

---

## 6. Istruzioni e Regole di Dominio

- **Lingua**: tutto il codice (commenti, nomi variabili, print, markdown) è in **italiano**. Mantenere questa convenzione.
- **Percorsi**: mai hardcodati — sempre `os.path.join(paths['...'], ...)` con `paths` da `common.py`.
- **Chiave di join** tra immagine e metadati: `filename` (verificato univoco, non `filepath`).
- **Formato density map**: sempre `.npy` (float32), mai PNG/JPG — la quantizzazione distruggerebbe la proprietà "somma della mappa = conteggio puntini", segnale di training essenziale.
- **Log storici** (sanità, tuning): pattern append/concat su CSV esistente, non sovrascrittura — eccetto tabelle di dettaglio "ultima run" (es. `baseline_blob_dettaglio.csv`), che è corretto sovrascrivere.
- **Isolamento dei notebook**: ogni notebook = kernel separato, nessuno stato condiviso in memoria tra notebook diversi. Qualunque dato prodotto in un notebook e necessario in un altro va salvato su file (CSV/`.npy`/checkpoint) e ricaricato esplicitamente.
- **Modificare una cella ≠ eseguirla**: dopo ogni modifica a una classe/funzione già usata altrove nel notebook (es. `DatasetImmunogold`), rieseguire **anche** le celle a valle che istanziano oggetti da essa dipendenti (es. `dataset_train`, poi `train_loader`) — il riferimento Python non si auto-aggiorna.
- **Apertura standard di ogni notebook**: mount Drive (condizionale) → `sys.path.append(.../scripts)` → `importlib.reload(common)` → `from common import *`.
- **Split train/val/test**: sempre a livello di immagine originale, mai di patch, per evitare data leakage.
- **Metodo di lavoro atteso dall'utente (Samuele)**: preferisce costruire ed eseguire il codice **passo per passo**, verificando ogni piccolo pezzo (shape dei tensori, test su singolo caso) prima di scalare all'intero dataset. Vuole spiegazioni del "perché", non solo del "cosa". Fa domande di verifica frequenti e si aspetta risposte precise e puntuali (es. "in quale cella esatta devo mettere questo codice"), non genériche. Ha accesso Google Drive collegabile all'agente AI per verifica diretta di file/notebook reali (via connettore MCP), preferibile a fidarsi solo di screenshot incollati.

---

## 7. Roadmap e Prossimi Passi

1. **Sbloccare `common.py`**: confermare/correggere l'aggiunta di `plt` alla lista `__all__`; rieseguire la cella di apertura in tutti i notebook aperti.
2. **Validare il fix di `FATTORE_SCALA`**: riavviare runtime con GPU attiva, eseguire `10_training.ipynb` per intero (ordine di dipendenza rigoroso: Fase0 → import modello → Dataset → Augmentation/istanza dataset → DataLoader → Loss/Optimizer → Training) con `NUM_EPOCHS=5` come smoke test. Verificare: (a) la val loss scende chiaramente sotto la linea di riferimento "predici sempre zero"; (b) gli output visivi mostrano struttura reale (non uniformi), con somme predette vicine a quelle target.
3. Se il fix funziona: alzare `NUM_EPOCHS` per un run di training serio (il record di best val loss ora persiste correttamente tra riavvii).
4. **Implementare il calcolo MAE della CNN** sul validation set, con la stessa formula del baseline (`|somma(density_map_predetta)/FATTORE_SCALA - n_punti_annotati|`, aggregato), per confronto diretto con **MAE baseline = 1.857**.
5. Iterare su architettura/iperparametri (canali base, batch size, learning rate, numero epoche, eventuale loss ausiliaria basata sul conteggio) se il MAE della CNN non supera il baseline.
6. (Opzionale, se servono run lunghi non presidiati) Estendere il checkpointing a stato optimizer + epoca per vero resume.
7. Proseguire l'annotazione incrementale oltre le 17 immagini attuali (su 144 totali), specialmente per le classi di calibrazione oggi rappresentate da 1-2 sole immagini.
8. Valutare l'uso della funzione di rilevamento scale-bar/artefatti come filtro di qualità dati **prima** dell'inclusione delle patch nel training set della CNN.
9. **Valutazione finale sul test set** — una sola volta, solo a modello e iperparametri ormai fissati.
10. (Fase futura, fuori scope finché il conteggio non è solido) Fase di classificazione monomero/trimero/artefatto via clustering spaziale non supervisionato (DBSCAN), con soglie di distanza motivate dalla geometria del linkage anticorpo-gold.