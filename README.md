# Task-Battery
Run the .bat scripts in order on your first run. These will automatically set up a compatible Anaconda environment.

Once the environment has been set up, you shouldn't need to repeat the setup process again. 

The #3 .bat script will run the battery and automatically activate the python environment.

In addition to the task battery, there is also a **dialogue extraction and semantic analysis pipeline** located in `Run_Dialogue_Extraction/`.  
- Run `Create_whisperxenv.bat` once to set up the `whisperx` environment.  
- Then use the provided `.bat` files to:  
  1. Transcribe videos and run semantic analysis (`Run_Transcription_And_Semantics.bat`)  
  2. Clean, combine, and run KDE + timeline analysis (`run_kde_pipeline.bat`)  
  3. Run permutation testing (`Run_Permutation.bat`)  

---

REQUIREMENTS: 

Anaconda

---

TODO:

### Refactoring
  - Create library of common functions
  - Rearrange the filesystem to make task scripts accessible at a higher level
  - Get rid off all unnecessary code (a lot)
  - Unit testing all functions
  - Document everything 
  - Create low level task schema (abstract class)

### New features
  - Create common config file which is human readable
  - Have a script to edit the config file through a gui
  - Modernize the data collection (SQL)?
  - Set new data collection to run in parallel to not interrupt current collection
  - SQL server??? 
