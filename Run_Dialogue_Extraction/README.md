# Dialogue Extraction and Analysis

This folder contains **batch files (.bat)** and environment definitions to run the full dialogue extraction and semantic analysis pipeline for the **Friends Event Segmentation project**.

The workflow includes:
1. Creating the **WhisperX environment**  
2. Transcribing videos into aligned transcripts  
3. Running semantic analysis on transcripts  
4. Cleaning and combining data  
5. Running KDE analysis and generating timeline graphs  
6. Running permutation tests  

---

## 📂 Folder Contents

- `Create_whisperxenv.bat` → Creates the `whisperx` environment (for transcription + semantics)  
- `whisperx_env.yml` → Conda environment specification for WhisperX + semantics  
- `Run_Transcription_And_Semantics.bat` → Runs both transcription and semantic analysis scripts  
- `run_kde_pipeline.bat` → Runs the full analysis pipeline (clean → combine → KDE → timeline graph)  
- `Run_Permutation.bat` → Runs the permutation testing script  
- `README.md` → This guide  

---

## ⚙️ Environment Setup

This workflow uses **two separate Conda environments**:

- **`whisperx`** → For transcription (WhisperX) and semantic analysis (SentenceTransformers)  
- **`psytask`** → For event segmentation analysis (cleaning, combining, KDE, permutation)  

### 1. Create the WhisperX environment
Run once to set up WhisperX:

```bash
Run Create_whisperxenv.bat
````

This will create a Conda environment named `whisperx` from `whisperx_env.yml`.

### 2. Make sure you also have the `psytask` environment
## IF YOU HAVE ALREADY CREATED THE PSYTASK ENVIRONMENT IGNORE THIS STEP

This environment is defined in `Tasks\taskScripts\resources\environment.yml`

```bash
conda env create -f Tasks\taskScripts\resources\environment.yml #Creates Psytask environment
```

---

## ▶️ Usage

### Step 1. Transcription + Semantics (WhisperX environment)

Transcribe videos and compute semantic similarity:

```bash
Run Run_Transcription_And_Semantics.bat #Change video directory and output folder within the scripts, found here: Analysis\event_seg analysis\dialogue_extraction 
```

This will:

* Run `extract_dialogue.py` → produces `*_aligned.csv` transcripts
* Run `semantics.py` → produces semantic timecourses + plots

Outputs are saved in:

```
Analysis\event_seg analysis\Analyzed_Data\Transcripts
Analysis\event_seg analysis\Analyzed_Data\Semantics
```

---

### Step 2. Clean + Combine + KDE + Timeline Graph (Psytask environment)

Run the KDE pipeline:

```bash
Run run_kde_pipeline.bat
```

This will:

1. Run `clean_data.py` → cleans raw event segmentation data
2. Run `combine_csv.py` → merges cleaned CSV files
3. Run `KDE.py` → performs kernel density estimation on boundaries
4. Run `Timeline_graph.py` → generates time-aligned graphs

Outputs are saved in:

```
Analysis\event_seg analysis\Analyzed_Data\Kernal
```


### Step 3. Permutation Testing (Psytask environment)

Run the permutation test analysis:

```bash
Run Run_Permutation.bat
```

This will:

* Load semantic + boundary data
* Perform circular-shift permutation testing
* Save null distributions, plots, and summary CSVs

Outputs are saved in:

```
Analysis\event_seg analysis\Analyzed_Data\Semantics\Perm
```

## ✅ Full Workflow Summary

1. **Set up environments**

   * `Create_whisperxenv.bat` (creates `whisperx`)
   * Make sure `psytask` is already created

2. **Run dialogue extraction**

   * `Run_Transcription_And_Semantics.bat`

3. **Run KDE pipeline**

   * `run_kde_pipeline.bat`

4. **Run permutation test**

   * `Run_Permutation.bat`

At the end of this workflow you will have:

* Aligned transcripts
* Semantic similarity timecourses
* Cleaned + combined event segmentation data
* KDE and timeline graphs
* Permutation test results

---

## 📌 Notes

* Always use the `.bat` files provided here — they handle activating the correct environment and running scripts in order.
* GPU is recommended for WhisperX transcription (CPU fallback is possible but slower).
* If you update script paths or add new scripts, make sure to also update the corresponding `.bat` file.
* All outputs are saved into `Analysis\event_seg analysis\Analyzed_Data\...` subfolders.


