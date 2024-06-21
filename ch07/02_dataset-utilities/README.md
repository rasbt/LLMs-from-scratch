# Chapter 7: Finetuning to Follow Instructions

This folder contains utility code that can be used for preparing an instruction dataset.

Install the additional package requirements via:

```bash
pip install -r requirements-extra.txt
```





### Finding Near Duplicates

The `find-near-duplicates.py` function can be used to identify duplicates and near-duplicates in an instruction dataset. For example,



```bash
python find-near-duplicates.py --json_file instruction-examples.json
```

```
scikit-learn version: 1.3.1


==================================================
Searching 'instruction' for duplicates ...
==================================================
Duplicate pair found with similarity 0.94:
1. Edit the following sentence to make it more formal.
2. Edit the sentence to make it more formal.

Duplicate pair found with similarity 1.00:
1. Name a dwarf planet in our solar system.
2. Name a dwarf planet in our solar system.

Duplicate pair found with similarity 0.91:
1. Change the sentences from active voice to passive voice.
2. Change the sentence from passive to active voice.



==================================================
Searching 'input' for duplicates ...
==================================================
No duplicates found


==================================================
Searching 'output' for duplicates ...
==================================================
Duplicate pair found with similarity 1.00:
1. One dwarf planet in our solar system is Pluto.
2. One dwarf planet in our solar system is Pluto.


```

&nbsp;
You can use the `--threshold` setting with a value between 0 and 1 to decrease or increase the sensitivity.
The default threshold is 0.9.



&nbsp;
 ## Creating Passive Voice Entries

 - The [create-passive-voice-entries.ipynb](create-passive-voice-entries.ipynb) notebook uses OpenAI's GPT-4 to create "passive voice" entries for an instruction dataset, as shown in the example below

 ```python
 {  
    'instruction': 'Identify the verb in the following sentence',
    'input': 'The cat sleeps on the couch.',
    'output': 'The verb in the sentence is "sleeps."',
    'output_2': 'The sentence is "sleeps."'   #  <---- Newly created entry
 }  
 ```
