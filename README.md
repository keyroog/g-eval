
# **G-EVAL: Valutazione Avanzata della Qualità dei Dialoghi**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

G-EVAL è un framework avanzato per la valutazione automatica della qualità dei dialoghi generati utilizzando metriche basate su modelli di linguaggio di grandi dimensioni (LLM). Questo repository offre strumenti per l'analisi di dataset, il calcolo delle metriche e la generazione di visualizzazioni dei risultati.

---

## **Caratteristiche Principali**
- 🚀 **Valutazione Multi-Livello**: Supporta sia valutazioni a livello di turno (turn-level) che di dialogo completo (dialog-level).
- 📊 **Analisi Completa**: Generazione di correlazioni, grafici di distribuzione e heatmap per analisi approfondite.
- 🛠️ **Altamente Personalizzabile**: Possibilità di utilizzare template personalizzati per diverse modalità di valutazione.
- 📂 **Dataset Supportati**: Include integrazione con i dataset FED e TC_USR.

---

## **Installazione**

1. Clona il repository:
   ```bash
   git clone https://github.com/tuo-username/g-eval.git
   cd g-eval
   ```

2. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

3. Configura la tua API key (se necessario):
   - Crea un file `.env` nella directory principale.
   - Aggiungi la tua chiave API:
     ```
     OPENAI_API_KEY=la_tua_chiave_api
     ```

---

## **Dataset**
- **FED**: Feedback Evaluation Dataset per dialoghi open-domain.
- **TC_USR**: Dataset focalizzato su dialoghi con valutazioni di engagingness e naturalness.

Assicurati che i dataset siano salvati nella cartella `datasets/`.

---

## **Modalità di Esecuzione**

### 1. **Valutazione del Dataset TC_USR**
Esegui la valutazione di un sottoinsieme del dataset TC_USR:
```bash
python main.py --mode tc_usr                --input_file datasets/tc_usr_data.json                --single_template_path prompts/tc_usr_single_response.txt                --full_template_path prompts/tc_usr_single_response.txt                --output_file results/tc/results_tc_usr.json                --num_records 1
```

### 2. **Valutazione del Dataset FED**
Esegui la valutazione di un sottoinsieme del dataset FED:
```bash
python main.py --mode fed                --input_file datasets/fed_data.json                --single_template_path prompts/fed_single_response.txt                --full_template_path prompts/fed_full_dialogue.txt                --output_file results/fed/results_fed.json                --num_records 2
```

### 3. **Analisi dei Risultati**
Analizza i risultati di una valutazione completata e genera visualizzazioni:
```bash
python main.py --mode results                --input_file results/fed/results_fed_all.json                --output_file results/analyze_output
```

---

## **Struttura del Repository**

- `datasets/`: Contiene i dataset di input (es. FED, TC_USR).
- `prompts/`: Template per i prompt di valutazione.
- `results/`: Output delle valutazioni e analisi.
- `main.py`: Script principale per eseguire le valutazioni e analisi.
- `evaluators/`: Moduli per il preprocessing e la valutazione dei dati.

---

## **Esempi di Output**
### Grafici Generati
#### Scatter Plot
![Scatter Plot](results/tc/scatter_plot.png)

#### Distribuzione delle Differenze
![Istogramma delle Differenze](results/tc/difference_histogram.png)

#### Distribuzione dei Punteggi
![Boxplot](results/tc/boxplot.png)

#### Correlazioni
![Bar Plot delle Correlazioni](results/tc/correlation_bar_plot.png)


---

## **Contatti**
Per domande o suggerimenti:
- Email: **s.sirica2000@gmail.com**
- LinkedIn: **[Salvatore Sirica](https://www.linkedin.com/in/salvatore-sirica-823325208/)**

---
